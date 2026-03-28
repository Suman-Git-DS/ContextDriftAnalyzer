"""Tests for context window manager."""

import pytest
from context_decay_drift.context.manager import ContextManager, SessionSummary


class TestContextManager:
    def test_initial_managed_context(self):
        cm = ContextManager(original_context="You are a Python tutor.")
        assert cm.build_managed_context() == "You are a Python tutor."

    def test_add_session_summary(self):
        cm = ContextManager(original_context="System prompt.")
        summary = cm.add_session_summary(
            session_text="We discussed variables and functions in Python.",
            session_number=1,
            turn_count=6,
            final_drift_score=85.0,
        )
        assert summary.session_number == 1
        assert summary.turn_count == 6

        ctx = cm.build_managed_context()
        assert "System prompt." in ctx
        assert "Session 1" in ctx
        assert "drift: 85.0" in ctx

    def test_max_summary_sessions(self):
        cm = ContextManager(original_context="Test", max_summary_sessions=2)

        for i in range(5):
            cm.add_session_summary(
                session_text=f"Session {i} content.",
                session_number=i + 1,
                turn_count=4,
                final_drift_score=80.0,
            )

        assert len(cm.summaries) == 2
        # Should keep the last 2
        assert cm.summaries[0].session_number == 4
        assert cm.summaries[1].session_number == 5

    def test_clear_history(self):
        cm = ContextManager(original_context="Test")
        cm.add_session_summary("Content", 1, 4, 90.0)
        cm.clear_history()
        assert len(cm.summaries) == 0
        # Original context preserved
        assert "Test" in cm.build_managed_context()

    def test_freeze_prevents_add(self):
        cm = ContextManager(original_context="Test", frozen=True)
        with pytest.raises(RuntimeError, match="frozen"):
            cm.add_session_summary("Content", 1, 4, 90.0)

    def test_freeze_prevents_clear(self):
        cm = ContextManager(original_context="Test", frozen=True)
        with pytest.raises(RuntimeError, match="frozen"):
            cm.clear_history()

    def test_freeze_unfreeze(self):
        cm = ContextManager(original_context="Test")
        cm.freeze()
        assert cm.frozen is True
        cm.unfreeze()
        assert cm.frozen is False
        cm.add_session_summary("Content", 1, 4, 90.0)
        assert len(cm.summaries) == 1

    def test_estimate_token_count(self):
        cm = ContextManager(original_context="A" * 400)
        # ~400 chars / 4 = ~100 tokens
        assert cm.estimate_token_count() == 100

    def test_export_import_summaries(self):
        cm = ContextManager(original_context="Test")
        cm.add_session_summary("Python loops and functions.", 1, 5, 82.0)
        cm.add_session_summary("Error handling discussion.", 2, 3, 71.0)

        exported = cm.export_summaries()
        assert len(exported) == 2

        cm2 = ContextManager(original_context="Test")
        cm2.load_summaries(exported)
        assert len(cm2.summaries) == 2
        assert cm2.summaries[0].session_number == 1

    def test_default_summarize(self):
        cm = ContextManager(original_context="Test")
        summary = cm.add_session_summary(
            session_text="First sentence here. Second sentence. Third sentence. Fourth sentence. Last one.",
            session_number=1,
            turn_count=5,
            final_drift_score=80.0,
        )
        # Default summarizer takes first 2 + last sentence
        assert "First sentence here" in summary.summary
        assert "Last one" in summary.summary

    def test_custom_summarize_fn(self):
        cm = ContextManager(
            original_context="Test",
            summarize_fn=lambda text: f"SUMMARY: {text[:20]}",
        )
        summary = cm.add_session_summary("Hello world test content", 1, 4, 90.0)
        assert summary.summary.startswith("SUMMARY:")
