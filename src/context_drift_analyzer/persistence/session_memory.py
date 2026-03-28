"""Session memory persistence via .session_memory JSON files.

Stores a full audit trail of every conversation organized by session.
Each session contains all Q&A exchanges with their drift scores.

Structure:
    {
      "original_context": "System prompt + few-shot examples",
      "session_count": 2,
      "context_frozen": false,
      "sessions": [
        {
          "session_number": 1,
          "status": "completed",
          "exchanges": [
            {
              "exchange": 1,
              "user": "What credit cards do you offer?",
              "assistant": "We offer Acme Rewards...",
              "score": 95.0,
              "verdict": "fresh",
              "explanation": "Context is well-preserved..."
            }
          ],
          "summary": "Topics discussed: credit cards, home loans.",
          "final_drift_score": 85.7
        }
      ]
    }

Note: .session_memory files should be added to .gitignore. Do not commit
them — they may contain content from your conversations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionMemoryData:
    """In-memory representation of the .session_memory file.

    Fields:
        original_context: System prompt + few-shot examples (the ground truth).
        session_count: Total sessions (completed + active).
        context_frozen: Whether context modifications are locked.
        sessions: Full audit trail — list of session dicts, each containing
            exchanges with user/assistant messages and drift scores.
    """

    original_context: str = ""
    session_count: int = 0
    context_frozen: bool = False
    sessions: list[dict] = field(default_factory=list)


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

            # Backward compatibility: convert old format to new
            if "drift_history" in raw and "sessions" not in raw:
                raw = self._migrate_old_format(raw)

            valid_fields = SessionMemoryData.__dataclass_fields__
            return SessionMemoryData(**{
                k: v for k, v in raw.items()
                if k in valid_fields
            })
        except (json.JSONDecodeError, TypeError):
            return SessionMemoryData()

    def save(self, data: SessionMemoryData) -> None:
        """Save session memory to disk."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "original_context": data.original_context,
                    "session_count": data.session_count,
                    "context_frozen": data.context_frozen,
                    "sessions": data.sessions,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

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

    @staticmethod
    def _migrate_old_format(raw: dict) -> dict:
        """Convert old flat drift_history format to new sessions format."""
        sessions = []
        # Group old drift_history by session number
        history = raw.get("drift_history", [])
        by_session: dict[int, list] = {}
        for entry in history:
            sn = entry.get("session", 1)
            by_session.setdefault(sn, []).append(entry)

        for sn in sorted(by_session.keys()):
            entries = by_session[sn]
            exchanges = []
            for e in entries:
                exchanges.append({
                    "exchange": e.get("exchange", e.get("turn", 0)),
                    "user": "",
                    "assistant": "",
                    "score": e.get("score", 0),
                    "verdict": e.get("verdict", ""),
                    "explanation": e.get("explanation", ""),
                })
            final_score = entries[-1].get("score", 0) if entries else 0
            sessions.append({
                "session_number": sn,
                "status": "completed",
                "exchanges": exchanges,
                "summary": "(migrated from old format)",
                "final_drift_score": final_score,
            })

        # Also migrate session_summaries if present
        for s in raw.get("session_summaries", []):
            sn = s.get("session_number", 0)
            for sess in sessions:
                if sess["session_number"] == sn:
                    sess["summary"] = s.get("summary", "")
                    break

        return {
            "original_context": raw.get("original_context", ""),
            "session_count": raw.get("session_count", len(sessions)),
            "context_frozen": raw.get("context_frozen", False),
            "sessions": sessions,
        }
