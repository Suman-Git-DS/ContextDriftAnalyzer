"""Unified drift tracker — the main entry point for context-drift-analyzer.

This is the single wrapper for any LLM pipeline. It combines:
- Drift detection (semantic or lexical strategies)
- Drift explanation (why did it drift?)
- Context management (original context + session summaries)
- Persistence (.session_memory file)
- On-demand or always-on scoring modes

Usage:
    from context_drift_analyzer import DriftTracker

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

from context_drift_analyzer.context.explainer import DriftExplainer
from context_drift_analyzer.context.manager import ContextManager
from context_drift_analyzer.core.analyzer import DriftAnalyzer
from context_drift_analyzer.core.scorer import DriftScore, DriftVerdict
from context_drift_analyzer.core.session import FewShotExample, Session
from context_drift_analyzer.persistence.session_memory import (
    SessionMemoryData,
    SessionMemoryStore,
)
from context_drift_analyzer.strategies.base import BaseStrategy
from context_drift_analyzer.utils.markdown import strip_markdown

# Default off-topic marker. When the LLM prefixes a response with this,
# we strip everything after "I can help you with" (the redirect portion)
# so that only the off-topic acknowledgement is scored for drift.
DEFAULT_OFF_TOPIC_MARKER = "This is off-topic"


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
        off_topic_marker: Phrase the LLM uses to prefix off-topic responses.
            When detected, the redirect portion ("I can help you with...")
            is stripped before scoring so it doesn't inflate the drift score.
            Default: "This is off-topic". Set to None to disable.
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
        off_topic_marker: Optional[str] = DEFAULT_OFF_TOPIC_MARKER,
    ):
        if mode not in ("always", "ondemand"):
            raise ValueError("mode must be 'always' or 'ondemand'")

        self.mode = mode
        self.strip_md = strip_md
        self.off_topic_marker = off_topic_marker.lower() if off_topic_marker else None

        # Strip off-topic instruction from system prompt before using it as
        # the scoring reference. The instruction ("If user asks something
        # unrelated... respond with 'This is off-topic...'") dilutes the
        # reference embedding with off-topic semantics, causing on-topic
        # responses to score low. We keep the full prompt for the LLM but
        # score against the clean version.
        scoring_prompt = self._strip_off_topic_instruction(system_prompt)

        # Session (uses cleaned prompt as the reference for drift scoring)
        self.session = Session(
            system_prompt=scoring_prompt,
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
        self._current_user_msg = ""  # Track current user message for persistence

        # Load persisted state if available
        if self._persist and self._store and self._store.exists():
            self._load_from_memory()

    def _strip_off_topic_instruction(self, prompt: str) -> str:
        """Remove off-topic handling instructions from the system prompt.

        The instruction ("If a user asks something unrelated... respond with
        'This is off-topic...'") adds off-topic semantics to the reference
        embedding, making on-topic responses score artificially low.

        Returns the cleaned prompt for use as the scoring reference.
        """
        if not self.off_topic_marker:
            return prompt

        import re
        # Match sentences that instruct the LLM how to handle off-topic questions
        # e.g. "If a user asks something unrelated to banking, respond with: '...'"
        marker_first_word = re.escape(self.off_topic_marker.split()[0])
        pattern = (
            r"[.!?]?\s*[Ii]f\s+(?:a\s+)?user\s+asks?\s+(?:something\s+)?(?:unrelated|off[- ]topic)"
            r".*?" + marker_first_word +
            r".*?['\"]\.?\s*"
        )
        cleaned = re.sub(pattern, " ", prompt, flags=re.IGNORECASE | re.DOTALL)

        # Fallback: if the marker phrase itself appears in the prompt, remove
        # everything from "If a/the user asks" to the end of that sentence block
        if self.off_topic_marker in cleaned.lower():
            # Remove from "If" instruction through the marker and rest of sentence
            pattern2 = r"[Ii]f\s+.*?" + re.escape(self.off_topic_marker) + r"[^.]*\.?"
            cleaned = re.sub(pattern2, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        return re.sub(r"\s+", " ", cleaned).strip()

    def _strip_off_topic_redirect(self, response: str) -> str:
        """Strip redirect boilerplate from off-topic responses.

        When the LLM correctly refuses an off-topic question and redirects
        ("This is off-topic... I can help you with savings, loans..."),
        the redirect text contains on-topic keywords that inflate the drift
        score. This method detects the off-topic marker and strips the
        redirect, leaving only the off-topic acknowledgement for scoring.

        Returns the original response if no off-topic marker is detected.
        """
        if not self.off_topic_marker:
            return response

        lower = response.lower()
        if self.off_topic_marker not in lower:
            return response

        # Find where the redirect starts — common patterns:
        # "I can help you with...", "I can assist with...",
        # "However, I can...", "but I can help..."
        import re
        # Split at the point where the assistant starts redirecting
        redirect_patterns = [
            r"(?:however|but)?\s*,?\s*i\s+can\s+(?:help|assist)\s+(?:you\s+)?with",
            r"(?:however|but)?\s*,?\s*i(?:'m|\s+am)\s+(?:here|designed|able)\s+to\s+(?:help|assist)",
            r"is\s+there\s+anything\s+(?:else\s+)?(?:i\s+can|banking)",
            r"(?:please\s+)?let\s+me\s+know\s+(?:how|if)\s+i\s+can",
            r"for\s+(?:example|instance)\s*,?\s*i\s+can",
            r"i\s+(?:can|could)\s+(?:however|instead)\s+help",
        ]

        best_cut = len(response)
        for pattern in redirect_patterns:
            match = re.search(pattern, lower)
            if match and match.start() < best_cut:
                best_cut = match.start()

        stripped = response[:best_cut].strip()
        # If we stripped almost everything, return a minimal off-topic marker
        if len(stripped) < 10:
            return "This is off-topic."
        return stripped

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
        self._current_user_msg = user_message
        self.session.add_user_message(user_message)

        # Optionally strip markdown before recording for drift analysis
        clean_response = (
            strip_markdown(assistant_response) if self.strip_md
            else assistant_response
        )

        # Strip off-topic redirect boilerplate before scoring
        scoring_response = self._strip_off_topic_redirect(clean_response)
        self.session.add_assistant_message(scoring_response)

        drift = None
        explanation = None

        if self.mode == "always":
            drift = self.analyzer.analyze(self.session)
            explanation = self.explainer.explain(
                self.session.initial_context,
                clean_response,
                drift.score,
                user_message=user_message,
            )

            # Persist exchange with full Q&A
            if self._persist and self._store:
                self._save_exchange(
                    user_message, assistant_response,
                    drift, explanation,
                )

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
        returns a full DriftReport. Does NOT write to memory (read-only).
        """
        drift = self.analyzer.analyze(self.session)

        recent_text = ""
        if self.session.assistant_turns:
            recent_text = self.session.assistant_turns[-1].content

        recent_user = ""
        if self.session.user_turns:
            recent_user = self.session.user_turns[-1].content

        explanation = self.explainer.explain(
            self.session.initial_context,
            recent_text,
            drift.score,
            user_message=recent_user,
        )

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

        # Compute final drift score directly (not via check() to avoid double-save)
        drift = self.analyzer.analyze(self.session)
        recent_text = ""
        if self.session.assistant_turns:
            recent_text = self.session.assistant_turns[-1].content
        recent_user = ""
        if self.session.user_turns:
            recent_user = self.session.user_turns[-1].content
        explanation = self.explainer.explain(
            self.session.initial_context, recent_text, drift.score,
            user_message=recent_user,
        )

        report = DriftReport(
            drift=drift,
            explanation=explanation,
            session_number=self._session_number,
            total_turns=self.session.turn_count,
            managed_context=self.context_manager.build_managed_context(),
            context_token_estimate=self.context_manager.estimate_token_count(),
        )

        # Summarize the session
        session_text = self.session.get_full_text()
        if not self.context_manager.frozen:
            self.context_manager.add_session_summary(
                session_text=session_text,
                session_number=self._session_number,
                turn_count=self.session.turn_count,
                final_drift_score=report.drift.score,
            )

        # Finalize session in memory (add summary, mark completed)
        if self._persist and self._store:
            self._finalize_session_in_memory(
                summary=self.context_manager.summaries[-1].summary
                if self.context_manager.summaries else "",
                final_drift_score=report.drift.score,
            )

        # Reset session for next one
        self.session.reset()
        self.analyzer.reset_floor()
        self._session_number += 1

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
        self.analyzer.reset_floor()
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

    def _save_exchange(
        self,
        user_message: str,
        assistant_response: str,
        drift: DriftScore,
        explanation: str,
    ) -> None:
        """Save a Q&A exchange to the current active session in memory."""
        if not self._store:
            return

        data = self._store.load() if self._store.exists() else SessionMemoryData()
        data.original_context = self.session.initial_context
        data.context_frozen = self.context_manager.frozen
        data.session_count = self._session_number

        # Find or create the active session
        active_session = None
        for s in data.sessions:
            if s.get("session_number") == self._session_number and s.get("status") == "active":
                active_session = s
                break

        if active_session is None:
            active_session = {
                "session_number": self._session_number,
                "status": "active",
                "exchanges": [],
                "summary": None,
                "final_drift_score": None,
            }
            data.sessions.append(active_session)

        # Append the exchange
        active_session["exchanges"].append({
            "exchange": self.session.turn_count,
            "user": user_message,
            "assistant": assistant_response[:500],  # Cap to avoid huge files
            "score": round(drift.score, 2),
            "verdict": drift.verdict.value,
            "explanation": explanation,
        })

        self._store.save(data)

    def _finalize_session_in_memory(
        self, summary: str, final_drift_score: float,
    ) -> None:
        """Mark the current session as completed with summary."""
        if not self._store:
            return

        data = self._store.load() if self._store.exists() else SessionMemoryData()
        data.original_context = self.session.initial_context
        data.context_frozen = self.context_manager.frozen
        data.session_count = self._session_number

        # Find the active session and finalize it
        for s in data.sessions:
            if s.get("session_number") == self._session_number and s.get("status") == "active":
                s["status"] = "completed"
                s["summary"] = summary
                s["final_drift_score"] = round(final_drift_score, 2)
                break

        self._store.save(data)

    def _load_from_memory(self) -> None:
        """Restore state from .session_memory file."""
        if not self._store:
            return

        data = self._store.load()

        # Figure out the next session number from completed sessions
        if data.sessions:
            max_session = max(s.get("session_number", 0) for s in data.sessions)
            self._session_number = max_session + 1
        elif data.session_count > 0:
            self._session_number = data.session_count + 1

        # Load session summaries into context manager
        summaries = []
        for s in data.sessions:
            if s.get("status") == "completed" and s.get("summary"):
                summaries.append({
                    "session_number": s["session_number"],
                    "summary": s["summary"],
                    "exchange_count": len(s.get("exchanges", [])),
                    "final_drift_score": s.get("final_drift_score", 0),
                })
        if summaries:
            self.context_manager.load_summaries(summaries)

        if data.context_frozen:
            self.context_manager.freeze()
