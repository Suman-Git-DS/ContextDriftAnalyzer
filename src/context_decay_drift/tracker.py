"""Unified drift tracker — the main entry point for context-decay-drift.

This is the single wrapper for any LLM pipeline. It combines:
- Drift detection (semantic or lexical strategies)
- Drift explanation (why did it drift?)
- Context management (original context + session summaries)
- Persistence (.session_memory file)
- On-demand or always-on scoring modes

Usage:
    from context_decay_drift import DriftTracker

    tracker = DriftTracker(
        system_prompt="You are a Python tutor...",
        mode="always",           # or "ondemand"
        persist=True,            # saves to .session_memory
        max_summary_sessions=3,  # keep last 3 session summaries
    )

    # After each LLM call:
    result = tracker.record_turn(
        user_message="How do loops work?",
        assistant_response="Loops iterate over sequences..."
    )
    print(result.drift.score)       # 85.2
    print(result.drift.explanation) # "Context well-preserved..."

    # On-demand mode:
    tracker = DriftTracker(system_prompt="...", mode="ondemand")
    tracker.record_turn(user_msg, assistant_msg)  # no scoring
    report = tracker.check()  # explicitly request drift check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from context_decay_drift.context.explainer import DriftExplainer
from context_decay_drift.context.manager import ContextManager
from context_decay_drift.core.analyzer import DriftAnalyzer
from context_decay_drift.core.scorer import DriftScore, DriftVerdict
from context_decay_drift.core.session import FewShotExample, Session
from context_decay_drift.persistence.session_memory import (
    SessionMemoryData,
    SessionMemoryStore,
)
from context_decay_drift.strategies.base import BaseStrategy
from context_decay_drift.utils.markdown import strip_markdown


@dataclass
class TurnResult:
    """Result of recording a conversation turn.

    Attributes:
        user_message: The user message that was recorded.
        assistant_response: The assistant response that was recorded.
        drift: DriftScore if mode is "always", None if "ondemand".
        explanation: Human-readable drift explanation (if available).
        managed_context: The full managed context string.
    """

    user_message: str
    assistant_response: str
    drift: Optional[DriftScore]
    explanation: Optional[str]
    managed_context: str

    def to_dict(self) -> dict:
        return {
            "user_message": self.user_message,
            "assistant_response": self.assistant_response[:200] + "..."
            if len(self.assistant_response) > 200
            else self.assistant_response,
            "drift": self.drift.to_dict() if self.drift else None,
            "explanation": self.explanation,
        }


@dataclass
class DriftReport:
    """Full drift report from an on-demand check.

    Attributes:
        drift: The DriftScore.
        explanation: Human-readable drift explanation.
        session_number: Current session number.
        total_turns: Cumulative turns across all sessions.
        managed_context: The current managed context string.
        context_token_estimate: Estimated tokens in managed context.
    """

    drift: DriftScore
    explanation: str
    session_number: int
    total_turns: int
    managed_context: str
    context_token_estimate: int

    def to_dict(self) -> dict:
        return {
            "score": round(self.drift.score, 2),
            "verdict": self.drift.verdict.value,
            "is_effective": self.drift.is_effective,
            "needs_reset": self.drift.needs_reset,
            "explanation": self.explanation,
            "session_number": self.session_number,
            "total_turns": self.total_turns,
            "context_token_estimate": self.context_token_estimate,
            "strategy_scores": {
                k: round(v, 2)
                for k, v in self.drift.strategy_scores.items()
            },
        }


class DriftTracker:
    """Unified drift tracker for any LLM pipeline.

    Args:
        system_prompt: The system prompt / instructions.
        few_shot_examples: Optional few-shot (user, assistant) pairs.
        mode: "always" scores every turn, "ondemand" only scores on check().
        strategies: List of drift strategies. Default uses keyword + token overlap.
        decay_rate: Exponential decay factor (0-1]. Default 0.95.
        window_size: Recent turns to evaluate. 0 = all. Default 5.
        persist: Whether to persist session memory to disk. Default False.
        persist_path: Path for .session_memory file. Default ".session_memory".
        max_summary_sessions: Number of past session summaries to retain. Default 3.
        summarize_fn: Custom function to summarize sessions. Default: extractive.
        explain_fn: Custom function to explain drift. Default: rule-based local.
        strip_md: Strip markdown before embedding. Default True.
        frozen: Freeze context (no modifications). Default False.
    """

    def __init__(
        self,
        system_prompt: str,
        few_shot_examples: Optional[list[FewShotExample]] = None,
        mode: str = "always",
        strategies: Optional[list[BaseStrategy]] = None,
        decay_rate: float = 0.95,
        window_size: int = 5,
        persist: bool = False,
        persist_path: str = ".session_memory",
        max_summary_sessions: int = 3,
        summarize_fn: Optional[Callable[[str], str]] = None,
        explain_fn: Optional[Callable[[str, str, float], str]] = None,
        strip_md: bool = True,
        frozen: bool = False,
    ):
        if mode not in ("always", "ondemand"):
            raise ValueError("mode must be 'always' or 'ondemand'")

        self.mode = mode
        self.strip_md = strip_md

        # Session
        self.session = Session(
            system_prompt=system_prompt,
            few_shot_examples=few_shot_examples or [],
        )

        # Analyzer
        self.analyzer = DriftAnalyzer(
            strategies=strategies,
            decay_rate=decay_rate,
            window_size=window_size,
        )

        # Explainer
        self.explainer = DriftExplainer(explain_fn=explain_fn)

        # Context manager
        self.context_manager = ContextManager(
            original_context=self.session.initial_context,
            max_summary_sessions=max_summary_sessions,
            summarize_fn=summarize_fn,
            frozen=frozen,
        )

        # Persistence
        self._persist = persist
        self._store = SessionMemoryStore(path=persist_path) if persist else None
        self._session_number = 1

        # Load persisted state if available
        if self._persist and self._store and self._store.exists():
            self._load_from_memory()

    def record_turn(
        self,
        user_message: str,
        assistant_response: str,
    ) -> TurnResult:
        """Record a conversation turn.

        In "always" mode, computes drift score and explanation.
        In "ondemand" mode, just records the turn (no scoring overhead).

        Args:
            user_message: What the user said.
            assistant_response: What the assistant replied.

        Returns:
            TurnResult with drift info (if mode is "always").
        """
        self.session.add_user_message(user_message)

        # Optionally strip markdown before recording for drift analysis
        clean_response = (
            strip_markdown(assistant_response) if self.strip_md
            else assistant_response
        )
        self.session.add_assistant_message(clean_response)

        drift = None
        explanation = None

        if self.mode == "always":
            drift = self.analyzer.analyze(self.session)
            explanation = self.explainer.explain(
                self.session.initial_context,
                clean_response,
                drift.score,
            )

            # Persist drift history
            if self._persist and self._store:
                self._save_to_memory(drift, explanation)

        managed_context = self.context_manager.build_managed_context()

        return TurnResult(
            user_message=user_message,
            assistant_response=assistant_response,
            drift=drift,
            explanation=explanation,
            managed_context=managed_context,
        )

    def check(self) -> DriftReport:
        """On-demand drift check. Works in both modes.

        Computes current drift score, generates explanation, and
        returns a full DriftReport.
        """
        drift = self.analyzer.analyze(self.session)

        recent_text = ""
        if self.session.assistant_turns:
            recent_text = self.session.assistant_turns[-1].content

        explanation = self.explainer.explain(
            self.session.initial_context,
            recent_text,
            drift.score,
        )

        if self._persist and self._store:
            self._save_to_memory(drift, explanation)

        return DriftReport(
            drift=drift,
            explanation=explanation,
            session_number=self._session_number,
            total_turns=self.session.turn_count,
            managed_context=self.context_manager.build_managed_context(),
            context_token_estimate=self.context_manager.estimate_token_count(),
        )

    def end_session(self) -> Optional[DriftReport]:
        """End the current session, summarize it, and start a new one.

        The current session is summarized and added to the context manager.
        Session turns are cleared but original context is preserved.

        Returns:
            Final DriftReport for the session, or None if no turns.
        """
        if not self.session.turns:
            return None

        # Get final drift score
        report = self.check()

        # Summarize the session
        session_text = self.session.get_full_text()
        if not self.context_manager.frozen:
            self.context_manager.add_session_summary(
                session_text=session_text,
                session_number=self._session_number,
                turn_count=self.session.turn_count,
                final_drift_score=report.drift.score,
            )

        # Reset session for next one
        self.session.reset()
        self._session_number += 1

        # Persist
        if self._persist and self._store:
            self._save_to_memory(report.drift, report.explanation)

        return report

    def get_managed_context(self) -> str:
        """Get the current managed context string.

        Use this as the system message for your LLM calls. It contains
        the original context plus summaries of past sessions.
        """
        return self.context_manager.build_managed_context()

    def freeze_context(self) -> None:
        """Freeze context — no more session summaries can be added."""
        self.context_manager.freeze()

    def unfreeze_context(self) -> None:
        """Unfreeze context to allow modifications."""
        self.context_manager.unfreeze()

    def clear_history(self) -> None:
        """Remove all session summaries. Original context preserved."""
        self.context_manager.clear_history()

    def reset(self) -> None:
        """Full reset — clear everything including persisted memory."""
        self.session.reset()
        self.context_manager.clear_history()
        self._session_number = 1
        if self._persist and self._store:
            self._store.delete()

    @property
    def session_number(self) -> int:
        return self._session_number

    @property
    def turn_count(self) -> int:
        return self.session.turn_count

    @property
    def is_frozen(self) -> bool:
        return self.context_manager.frozen

    def _save_to_memory(self, drift: DriftScore, explanation: str) -> None:
        """Persist current state to .session_memory."""
        if not self._store:
            return

        data = self._store.load() if self._store.exists() else SessionMemoryData()
        data.original_context = self.session.initial_context
        data.context_frozen = self.context_manager.frozen
        data.session_count = self._session_number
        data.total_turns = self.session.turn_count
        data.session_summaries = self.context_manager.export_summaries()
        data.drift_history.append({
            "turn": self.session.turn_count,
            "session": self._session_number,
            "score": round(drift.score, 2),
            "verdict": drift.verdict.value,
            "explanation": explanation,
        })
        # Keep last 100 drift history entries
        data.drift_history = data.drift_history[-100:]

        if self.session.assistant_turns:
            data.last_response_text = self.session.assistant_turns[-1].content

        self._store.save(data)

    def _load_from_memory(self) -> None:
        """Restore state from .session_memory file."""
        if not self._store:
            return

        data = self._store.load()
        if data.session_count > 0:
            self._session_number = data.session_count + 1
        if data.session_summaries:
            self.context_manager.load_summaries(data.session_summaries)
        if data.context_frozen:
            self.context_manager.freeze()
