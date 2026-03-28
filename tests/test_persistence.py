"""Tests for session memory persistence."""

import json
import os
import pytest
import tempfile

from context_drift_analyzer.persistence.session_memory import (
    SessionMemoryData,
    SessionMemoryStore,
)


class TestSessionMemoryStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, ".session_memory")
        self.store = SessionMemoryStore(path=self.path)

    def test_exists_false_initially(self):
        assert self.store.exists() is False

    def test_save_and_load(self):
        data = SessionMemoryData(
            original_context="You are a tutor.",
            session_count=2,
            context_frozen=True,
            sessions=[
                {
                    "session_number": 1,
                    "status": "completed",
                    "exchanges": [
                        {
                            "exchange": 1,
                            "user": "What is Python?",
                            "assistant": "Python is a programming language.",
                            "score": 95.0,
                            "verdict": "fresh",
                            "explanation": "Context well-preserved.",
                        }
                    ],
                    "summary": "Topics discussed: What is Python?.",
                    "final_drift_score": 95.0,
                }
            ],
        )
        self.store.save(data)
        assert self.store.exists() is True

        loaded = self.store.load()
        assert loaded.original_context == "You are a tutor."
        assert loaded.session_count == 2
        assert loaded.context_frozen is True
        assert len(loaded.sessions) == 1
        assert loaded.sessions[0]["exchanges"][0]["user"] == "What is Python?"

    def test_load_missing_file(self):
        data = self.store.load()
        assert data.session_count == 0
        assert data.original_context == ""

    def test_load_corrupted_file(self):
        with open(self.path, "w") as f:
            f.write("not json{{{")
        data = self.store.load()
        assert data.session_count == 0

    def test_delete(self):
        self.store.save(SessionMemoryData())
        assert self.store.delete() is True
        assert self.store.exists() is False

    def test_delete_missing(self):
        assert self.store.delete() is False

    def test_saved_file_is_valid_json(self):
        self.store.save(SessionMemoryData(original_context="test"))
        with open(self.path, "r") as f:
            parsed = json.load(f)
        assert parsed["original_context"] == "test"

    def test_sessions_roundtrip(self):
        data = SessionMemoryData(
            session_count=1,
            sessions=[{
                "session_number": 1,
                "status": "active",
                "exchanges": [
                    {"exchange": 1, "user": "Hello", "assistant": "Hi there",
                     "score": 95.0, "verdict": "fresh", "explanation": "Good"},
                ],
                "summary": None,
                "final_drift_score": None,
            }],
        )
        self.store.save(data)
        loaded = self.store.load()
        assert len(loaded.sessions) == 1
        assert loaded.sessions[0]["status"] == "active"
        assert loaded.sessions[0]["exchanges"][0]["assistant"] == "Hi there"

    def test_backward_compat_old_format(self):
        """Old files with drift_history should be migrated to sessions."""
        old_data = {
            "original_context": "You are a tutor.",
            "session_count": 2,
            "drift_history": [
                {"turn": 1, "session": 1, "score": 95.0, "verdict": "fresh", "explanation": "Good"},
                {"turn": 2, "session": 1, "score": 80.0, "verdict": "mild", "explanation": "Mild drift"},
                {"turn": 1, "session": 2, "score": 90.0, "verdict": "fresh", "explanation": "Good"},
            ],
        }
        with open(self.path, "w") as f:
            json.dump(old_data, f)

        loaded = self.store.load()
        assert len(loaded.sessions) == 2
        assert loaded.sessions[0]["session_number"] == 1
        assert len(loaded.sessions[0]["exchanges"]) == 2
        assert loaded.sessions[1]["session_number"] == 2
