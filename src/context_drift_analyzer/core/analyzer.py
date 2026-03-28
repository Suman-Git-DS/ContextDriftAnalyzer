"""Central drift analyzer that orchestrates strategies and produces scores."""

from __future__ import annotations

from typing import Optional

from context_drift_analyzer.core.scorer import DriftScore, DriftVerdict
from context_drift_analyzer.core.session import Session
from context_drift_analyzer.strategies.base import BaseStrategy
from context_drift_analyzer.strategies.composite import CompositeStrategy
from context_drift_analyzer.strategies.keyword import KeywordStrategy
from context_drift_analyzer.strategies.token_overlap import TokenOverlapStrategy


def _build_default_strategy() -> BaseStrategy:
    """Build the best available default strategy.

    Prefers SentenceTransformerStrategy (semantic) if sentence-transformers
    is installed, otherwise falls back to keyword + token overlap (lexical).
    """
    try:
        from context_drift_analyzer.strategies.sentence_transformer import (
            SentenceTransformerStrategy,
        )
        # Test that the import actually works (not just the module but the dep)
        from sentence_transformers import SentenceTransformer  # noqa: F401

        return SentenceTransformerStrategy()
    except ImportError:
        return CompositeStrategy(
            [
                KeywordStrategy(),
                TokenOverlapStrategy(),
            ]
        )


class DriftAnalyzer:
    """Analyzes context drift in LLM conversations.

    The analyzer accepts one or more strategies that each produce a sub-score.
    Scores are combined (weighted average) into a single 0-100 drift score.

    If no strategies are provided, automatically uses SentenceTransformerStrategy
    when sentence-transformers is installed (semantic, accurate), or falls back
    to keyword + token overlap (lexical, zero dependencies).

    Args:
        strategies: List of strategies to use. If None, auto-detects best available.
        decay_rate: Exponential decay factor applied per turn (0-1).
            Lower values mean faster decay. Default 0.95.
        window_size: Number of recent assistant turns to consider
            for drift calculation. Default 5 (0 = use all turns).
    """

    def __init__(
        self,
        strategies: Optional[list[BaseStrategy]] = None,
        decay_rate: float = 0.95,
        window_size: int = 5,
    ):
        if decay_rate <= 0 or decay_rate > 1:
            raise ValueError("decay_rate must be in (0, 1]")
        if window_size < 0:
            raise ValueError("window_size must be >= 0")

        self.decay_rate = decay_rate
        self.window_size = window_size
        self._floor_score: Optional[float] = None

        if strategies is not None:
            self.strategy = (
                strategies[0]
                if len(strategies) == 1
                else CompositeStrategy(strategies)
            )
        else:
            self.strategy = _build_default_strategy()

    def analyze(self, session: Session) -> DriftScore:
        """Compute drift score for the current state of a session.

        Returns a DriftScore with the combined score and per-strategy breakdown.
        """
        if not session.assistant_turns:
            return DriftScore(
                score=100.0,
                verdict=DriftVerdict.FRESH,
                turn_number=0,
                session_id=session.session_id,
                strategy_scores={},
                metadata={"reason": "no_assistant_turns"},
            )

        # Select the window of assistant turns to evaluate
        assistant_turns = session.assistant_turns
        if self.window_size > 0:
            assistant_turns = assistant_turns[-self.window_size :]

        # Gather the text from the windowed assistant turns
        recent_texts = [t.content for t in assistant_turns]

        # Calculate raw strategy score against the full initial context
        # (system prompt + few-shot examples)
        raw_score, strategy_scores = self.strategy.score(
            system_prompt=session.initial_context,
            assistant_responses=recent_texts,
        )

        # Apply exponential decay based on total turn count
        total_turns = session.turn_count
        decay_factor = self.decay_rate ** (total_turns / 2)
        decayed_score = raw_score * decay_factor

        # Clamp to [0, 100]
        final_score = max(0.0, min(100.0, decayed_score))

        # Score floor: once the score drops, it cannot bounce back up
        # within the same session. This prevents misleading recovery
        # when the assistant uses on-topic keywords in redirect responses.
        if self._floor_score is not None:
            final_score = min(final_score, self._floor_score)
        self._floor_score = final_score

        verdict = DriftVerdict.from_score(final_score)

        return DriftScore(
            score=final_score,
            verdict=verdict,
            turn_number=session.turn_count,
            session_id=session.session_id,
            strategy_scores=strategy_scores,
            metadata={
                "raw_score": round(raw_score, 2),
                "decay_factor": round(decay_factor, 4),
                "decay_rate": self.decay_rate,
                "window_size": self.window_size,
                "turns_evaluated": len(recent_texts),
            },
        )

    def reset_floor(self) -> None:
        """Reset the score floor. Call when starting a new session."""
        self._floor_score = None

    def is_effective(self, session: Session) -> bool:
        """Quick check: is the session context still effective?"""
        return self.analyze(session).is_effective

    def needs_reset(self, session: Session) -> bool:
        """Quick check: should the session context be reset?"""
        return self.analyze(session).needs_reset
