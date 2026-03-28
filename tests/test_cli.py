"""Tests for CLI commands."""

import json
import os
import tempfile
import pytest

from context_drift_analyzer.cli.main import cmd_status, cmd_reset, cmd_history, cmd_freeze, cmd_unfreeze
from context_drift_analyzer.persistence.session_memory import SessionMemoryData, SessionMemoryStore


class FakeArgs:
    """Minimal args object for CLI tests."""
    def __init__(self, file_path, last=20):
        self.file = file_path
        self.last = last


class TestCLI:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, ".session_memory")

    def _create_memory(self, **kwargs):
        store = SessionMemoryStore(path=self.path)
        data = SessionMemoryData(**kwargs)
        store.save(data)

    def test_status_no_file(self, capsys):
        cmd_status(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "No session memory" in captured.out

    def test_status_with_file(self, capsys):
        self._create_memory(
            session_count=2,
            context_frozen=False,
            sessions=[
                {
                    "session_number": 1,
                    "status": "completed",
                    "exchanges": [
                        {"exchange": 1, "user": "Hello", "assistant": "Hi",
                         "score": 95.0, "verdict": "fresh", "explanation": "Good"},
                        {"exchange": 2, "user": "Bye", "assistant": "See ya",
                         "score": 72.5, "verdict": "moderate", "explanation": "Some drift"},
                    ],
                    "summary": "Topics discussed: Hello; Bye.",
                    "final_drift_score": 72.5,
                },
            ],
        )
        cmd_status(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "Session 1" in captured.out
        assert "completed" in captured.out
        assert "72.5" in captured.out

    def test_reset(self, capsys):
        self._create_memory(session_count=1)
        cmd_reset(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "Deleted" in captured.out
        assert not os.path.exists(self.path)

    def test_reset_no_file(self, capsys):
        cmd_reset(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "No session memory" in captured.out

    def test_history(self, capsys):
        self._create_memory(
            sessions=[{
                "session_number": 1,
                "status": "completed",
                "exchanges": [
                    {"exchange": 1, "user": "What is Python?", "assistant": "A language",
                     "score": 90.0, "verdict": "fresh", "explanation": "On topic"},
                    {"exchange": 2, "user": "Tell me about weather", "assistant": "Sunny today",
                     "score": 65.0, "verdict": "moderate", "explanation": "Drifting"},
                ],
                "summary": "Discussed Python and weather.",
                "final_drift_score": 65.0,
            }],
        )
        cmd_history(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "90.0" in captured.out
        assert "65.0" in captured.out
        assert "fresh" in captured.out
        assert "What is Python?" in captured.out

    def test_history_empty(self, capsys):
        self._create_memory()
        cmd_history(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "No drift history" in captured.out

    def test_freeze(self, capsys):
        self._create_memory(context_frozen=False)
        cmd_freeze(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "frozen" in captured.out.lower()

        store = SessionMemoryStore(path=self.path)
        data = store.load()
        assert data.context_frozen is True

    def test_unfreeze(self, capsys):
        self._create_memory(context_frozen=True)
        cmd_unfreeze(FakeArgs(self.path))
        captured = capsys.readouterr()
        assert "unfrozen" in captured.out.lower()

        store = SessionMemoryStore(path=self.path)
        data = store.load()
        assert data.context_frozen is False
