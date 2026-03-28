"""Session memory persistence via .session_memory JSON files.

Stores the original context fingerprint, session summaries, drift history,
and checkpoint embeddings so drift can be tracked across restarts and deploys.

The .session_memory file is a plain JSON file stored locally. It contains:

- original_context: the full initial context text (system prompt + few-shots)
- original_vector: embedding of the original context
- checkpoint_vector: embedding at last drift check
- checkpoint_text: text at last checkpoint
- last_response_vector: most recent response embedding
- last_response_text: most recent response text
- session_summaries: list of past session summaries (configurable count)
- session_count: total number of sessions
- total_turns: cumulative turn count across all sessions
- context_frozen: true once original fingerprint is set
- drift_history: list of {turn, score, verdict, explanation} entries

Note: .session_memory files should be added to .gitignore. Do not commit
them — they may contain embeddings of your production conversations.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionMemoryData:
    """In-memory representation of the .session_memory file."""

    original_context: str = ""
    original_vector: list[float] = field(default_factory=list)
    checkpoint_vector: list[float] = field(default_factory=list)
    checkpoint_text: str = ""
    last_response_vector: list[float] = field(default_factory=list)
    last_response_text: str = ""
    session_summaries: list[dict] = field(default_factory=list)
    session_count: int = 0
    total_turns: int = 0
    context_frozen: bool = False
    drift_history: list[dict] = field(default_factory=list)


class SessionMemoryStore:
    """Read/write .session_memory JSON files.

    Args:
        path: File path for the .session_memory file.
            Default ".session_memory" in the current directory.
    """

    def __init__(self, path: str = ".session_memory"):
        self.path = Path(path)

    def exists(self) -> bool:
        """Check if a .session_memory file exists."""
        return self.path.exists()

    def load(self) -> SessionMemoryData:
        """Load session memory from disk. Returns empty data if file missing."""
        if not self.path.exists():
            return SessionMemoryData()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return SessionMemoryData(**{
                k: v for k, v in raw.items()
                if k in SessionMemoryData.__dataclass_fields__
            })
        except (json.JSONDecodeError, TypeError):
            return SessionMemoryData()

    def save(self, data: SessionMemoryData) -> None:
        """Save session memory to disk."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(asdict(data), f, indent=2, ensure_ascii=False)

    def delete(self) -> bool:
        """Delete the .session_memory file. Returns True if deleted."""
        if self.path.exists():
            self.path.unlink()
            return True
        return False

    def ensure_gitignore(self) -> None:
        """Add .session_memory to .gitignore if not already present."""
        gitignore = self.path.parent / ".gitignore"
        pattern = self.path.name

        if gitignore.exists():
            content = gitignore.read_text(encoding="utf-8")
            if pattern in content:
                return
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write(f"\n{pattern}\n")
        else:
            gitignore.write_text(f"{pattern}\n", encoding="utf-8")
