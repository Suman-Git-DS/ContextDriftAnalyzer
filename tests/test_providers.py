"""Tests for provider wrappers."""

import pytest
from context_drift_analyzer.providers.generic import GenericDriftTracker
from context_drift_analyzer.providers.base import DriftAwareResponse
from context_drift_analyzer.core.scorer import DriftScore, DriftVerdict


SYSTEM_PROMPT = (
    "You are a Python programming tutor. Help students learn Python concepts "
    "including variables, functions, loops, classes, and error handling."
)


class TestGenericDriftTracker:
    def test_initial_drift_is_fresh(self):
        tracker = GenericDriftTracker(system_prompt=SYSTEM_PROMPT)
        drift = tracker.get_drift()
        assert drift.score == 100.0
        assert drift.verdict == DriftVerdict.FRESH

    def test_record_turn(self):
        tracker = GenericDriftTracker(system_prompt=SYSTEM_PROMPT)
        drift = tracker.record_turn(
            user_message="How do variables work?",
            assistant_response=(
                "Python variables store values. Functions use variables. "
                "Classes define variable scope."
            ),
        )
        assert 0 <= drift.score <= 100
        assert drift.turn_number == 1  # 1 Q&A exchange

    def test_drift_decreases_with_off_topic(self):
        tracker = GenericDriftTracker(
            system_prompt=SYSTEM_PROMPT, decay_rate=0.92
        )

        # On-topic turn
        d1 = tracker.record_turn(
            "Explain Python functions",
            "Functions in Python are defined with def. "
            "They help organize code into reusable blocks. "
            "Variables, loops, and classes are key Python concepts.",
        )

        # Many off-topic turns
        for i in range(8):
            tracker.record_turn(
                f"Tell me about sports team {i}",
                f"Team {i} won the championship last year. "
                "The crowd was thrilled with the final score.",
            )

        d_final = tracker.get_drift()
        assert d_final.score < d1.score

    def test_reset_session(self):
        tracker = GenericDriftTracker(system_prompt=SYSTEM_PROMPT)
        tracker.record_turn("Q", "A")
        tracker.reset_session()
        drift = tracker.get_drift()
        assert drift.score == 100.0

    def test_record_user_message_only(self):
        tracker = GenericDriftTracker(system_prompt=SYSTEM_PROMPT)
        tracker.record_user_message("Hello")
        assert tracker.session.turn_count == 1
        # Drift is still fresh since no assistant response
        drift = tracker.get_drift()
        assert drift.score == 100.0

    def test_record_assistant_response_only(self):
        tracker = GenericDriftTracker(system_prompt=SYSTEM_PROMPT)
        drift = tracker.record_assistant_response(
            "Python variables and functions are core concepts."
        )
        # No user message recorded, so exchange number is 0
        assert drift.turn_number == 0


class TestDriftAwareResponse:
    def test_properties(self):
        drift = DriftScore(
            score=75.5,
            verdict=DriftVerdict.MILD,
            turn_number=4,
            session_id="test",
        )
        resp = DriftAwareResponse(
            response={"raw": "data"},
            content="Hello world",
            drift=drift,
        )
        assert resp.drift_score == 75.5
        assert resp.drift_verdict == "mild"
        assert resp.content == "Hello world"
        assert resp.response == {"raw": "data"}

    def test_to_dict(self):
        drift = DriftScore(
            score=60.0,
            verdict=DriftVerdict.MODERATE,
            turn_number=6,
            session_id="test",
        )
        resp = DriftAwareResponse(response=None, content="Hi", drift=drift)
        d = resp.to_dict()
        assert d["content"] == "Hi"
        assert d["drift"]["score"] == 60.0
        assert d["drift"]["verdict"] == "moderate"

    def test_repr(self):
        drift = DriftScore(
            score=80.0,
            verdict=DriftVerdict.MILD,
            turn_number=2,
            session_id="test",
        )
        resp = DriftAwareResponse(response=None, content="Hi", drift=drift)
        r = repr(resp)
        assert "80.0" in r
        assert "mild" in r
