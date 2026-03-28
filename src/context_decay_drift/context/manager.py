"""Context window manager for keeping LLM context fresh across sessions.

The core problem: over long conversations, the LLM's context window fills
up and the original instructions get diluted. This module solves it by:

1. Always preserving the original context (system prompt + few-shots)
2. Summarizing past sessions into compact representations
3. Building a managed context = original + session summaries
4. Letting users control how many session summaries to retain

Architecture:
    ┌──────────────────────────────────────────┐
    │           Managed Context Window          │
    ├──────────────────────────────────────────┤
    │  [ALWAYS] Original System Prompt         │
    │  [ALWAYS] Few-Shot Examples              │
    ├──────────────────────────────────────────┤
    │  [CONFIGURABLE] Session 1 Summary        │
    │  [CONFIGURABLE] Session 2 Summary        │
    │  [CONFIGURABLE] Session 3 Summary        │
    ├──────────────────────────────────────────┤
    │  [CURRENT] Live Conversation Turns       │
    └──────────────────────────────────────────┘

The summaries compress past sessions into ~2-3 sentences each, so the
model retains awareness of prior conversations without consuming the
full context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class SessionSummary:
    """A compressed summary of a past session."""

    session_number: int
    summary: str
    turn_count: int
    final_drift_score: float


class ContextManager:
    """Manages the LLM context window to prevent dilution.

    Keeps the original context intact and appends compressed summaries
    of past sessions, so the model always has strong alignment with
    its original instructions.

    Args:
        original_context: The full initial context (system prompt + few-shots).
        max_summary_sessions: Maximum number of past session summaries to keep.
            Older summaries are dropped. Default 3.
        summarize_fn: Optional callable to summarize a session's text.
            Signature: (session_text: str) -> str
            If not provided, uses a simple extractive fallback.
        frozen: If True, context history cannot be modified. Default False.
    """

    def __init__(
        self,
        original_context: str,
        max_summary_sessions: int = 3,
        summarize_fn: Optional[Callable[[str], str]] = None,
        frozen: bool = False,
    ):
        self.original_context = original_context
        self.max_summary_sessions = max_summary_sessions
        self._summarize_fn = summarize_fn or self._default_summarize
        self._summaries: list[SessionSummary] = []
        self._frozen = frozen

    @property
    def frozen(self) -> bool:
        """Whether the context is frozen (no modifications allowed)."""
        return self._frozen

    def freeze(self) -> None:
        """Freeze the context — no more summaries can be added or removed."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the context to allow modifications."""
        self._frozen = False

    @property
    def summaries(self) -> list[SessionSummary]:
        """Current list of session summaries."""
        return list(self._summaries)

    def add_session_summary(
        self,
        session_text: str,
        session_number: int,
        turn_count: int,
        final_drift_score: float,
    ) -> SessionSummary:
        """Summarize a completed session and add it to the history.

        Args:
            session_text: Full text of the session to summarize.
            session_number: Session number for tracking.
            turn_count: Number of turns in the session.
            final_drift_score: The drift score at end of session.

        Returns:
            The created SessionSummary.

        Raises:
            RuntimeError: If context is frozen.
        """
        if self._frozen:
            raise RuntimeError(
                "Context is frozen. Call unfreeze() to allow modifications."
            )

        summary_text = self._summarize_fn(session_text)
        summary = SessionSummary(
            session_number=session_number,
            summary=summary_text,
            turn_count=turn_count,
            final_drift_score=final_drift_score,
        )
        self._summaries.append(summary)

        # Trim to max_summary_sessions (drop oldest)
        if len(self._summaries) > self.max_summary_sessions:
            self._summaries = self._summaries[-self.max_summary_sessions :]

        return summary

    def clear_history(self) -> None:
        """Remove all session summaries. Original context is preserved.

        Raises:
            RuntimeError: If context is frozen.
        """
        if self._frozen:
            raise RuntimeError(
                "Context is frozen. Call unfreeze() to allow modifications."
            )
        self._summaries.clear()

    def build_managed_context(self) -> str:
        """Build the full managed context for the LLM.

        Returns the original context + session summaries combined into
        a single string ready to use as the system message.
        """
        parts = [self.original_context]

        if self._summaries:
            parts.append("\n--- Previous Session Context ---")
            for s in self._summaries:
                parts.append(
                    f"[Session {s.session_number} | {s.turn_count} turns | "
                    f"drift: {s.final_drift_score:.1f}/100]: {s.summary}"
                )

        return "\n".join(parts)

    def estimate_token_count(self) -> int:
        """Rough estimate of token count for the managed context.

        Uses the ~4 chars per token approximation.
        """
        text = self.build_managed_context()
        return len(text) // 4

    def load_summaries(self, summaries_data: list[dict]) -> None:
        """Load summaries from persisted data (e.g., .session_memory file)."""
        self._summaries = [
            SessionSummary(**s) for s in summaries_data
        ]

    def export_summaries(self) -> list[dict]:
        """Export summaries for persistence."""
        return [
            {
                "session_number": s.session_number,
                "summary": s.summary,
                "turn_count": s.turn_count,
                "final_drift_score": s.final_drift_score,
            }
            for s in self._summaries
        ]

    @staticmethod
    def _default_summarize(text: str) -> str:
        """Simple extractive summarization fallback.

        Takes the first and last sentences as a crude summary.
        For production use, provide a proper summarize_fn (e.g., LLM-based).
        """
        sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
        if not sentences:
            return ""
        if len(sentences) <= 2:
            return ". ".join(sentences) + "."

        # Take first 2 and last sentence
        selected = sentences[:2] + [sentences[-1]]
        return ". ".join(selected) + "."
