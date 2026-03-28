"""Tests for session memory persistence."""

import json
import os
import pytest
import tempfile

from context_decay_drift.persistence.session_memory import (
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
            total_turns=10,
            context_frozen=True,
            session_summaries=[{"session_number": 1, "summary": "Talked about Python"}],
            drift_history=[{"turn": 5, "score": 75.0, "verdict": "mild"}],
        )
        self.store.save(data)
        assert self.store.exists() is True

        loaded = self.store.load()
        assert loaded.original_context == "You are a tutor."
        assert loaded.session_count == 2
        assert loaded.total_turns == 10
        assert loaded.context_frozen is True
        assert len(loaded.session_summaries) == 1
        assert len(loaded.drift_history) == 1

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

    def test_roundtrip_with_vectors(self):
        data = SessionMemoryData(
            original_vector=[0.1, 0.2, 0.3],
            last_response_vector=[0.4, 0.5, 0.6],
            last_response_text="Hello world",
        )
        self.store.save(data)
        loaded = self.store.load()
        assert loaded.original_vector == [0.1, 0.2, 0.3]
        assert loaded.last_response_text == "Hello world"
