"""Session management for tracking conversation history and drift over time."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """A single conversation turn (one Q&A exchange = one turn)."""

    role: str  # "user", "assistant", or "system"
    content: str
    turn_number: int  # The exchange number (1-based)


@dataclass
class FewShotExample:
    """A few-shot example pair used as initial context."""

    user: str
    assistant: str


@dataclass
class Session:
    """Tracks a conversation session with its full initial context and turn history.

    The initial context = system_prompt + few_shot_examples. This is the
    "ground truth" that drift is measured against. Over time, as the
    conversation progresses, responses may drift away from this baseline.

    Args:
        system_prompt: The original system prompt / instructions for the LLM.
        few_shot_examples: Optional list of few-shot (user, assistant) pairs
            that form part of the initial context.
        session_id: Optional identifier. Auto-generated if not provided.
    """

    system_prompt: str
    few_shot_examples: list[FewShotExample] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[Turn] = field(default_factory=list)
    _turn_counter: int = field(default=0, repr=False)

    def add_user_message(self, content: str) -> Turn:
        """Record a user message. Increments the exchange counter."""
        self._turn_counter += 1
        turn = Turn(role="user", content=content, turn_number=self._turn_counter)
        self.turns.append(turn)
        return turn

    def add_assistant_message(self, content: str) -> Turn:
        """Record an assistant response. Same exchange number as the user message."""
        turn = Turn(role="assistant", content=content, turn_number=self._turn_counter)
        self.turns.append(turn)
        return turn

    @property
    def initial_context(self) -> str:
        """The full initial context: system prompt + few-shot examples combined.

        This is the reference text that drift is measured against.
        """
        parts = [self.system_prompt]
        for ex in self.few_shot_examples:
            parts.append(f"User: {ex.user}")
            parts.append(f"Assistant: {ex.assistant}")
        return "\n".join(parts)

    @property
    def turn_count(self) -> int:
        """Total number of Q&A exchanges in the session."""
        return self._turn_counter

    @property
    def exchange_count(self) -> int:
        """Alias for turn_count — number of Q&A exchanges."""
        return self._turn_counter

    @property
    def assistant_turns(self) -> list[Turn]:
        """All assistant responses in order."""
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def user_turns(self) -> list[Turn]:
        """All user messages in order."""
        return [t for t in self.turns if t.role == "user"]

    def get_recent_context(self, n: int = 5) -> list[Turn]:
        """Get the last n turns of conversation."""
        return self.turns[-n:]

    def get_full_text(self) -> str:
        """Format all turns as readable Q&A exchanges."""
        lines = []
        for turn in self.turns:
            if turn.role == "user":
                lines.append(f"Q{turn.turn_number}: {turn.content}")
            elif turn.role == "assistant":
                lines.append(f"A{turn.turn_number}: {turn.content}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all turns but keep the system prompt, few-shots, and session ID."""
        self.turns.clear()
        self._turn_counter = 0
