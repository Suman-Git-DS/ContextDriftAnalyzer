"""Tests for the unified DriftTracker."""

import os
import tempfile
import pytest

from context_decay_drift import DriftTracker, FewShotExample, DriftVerdict


SYSTEM_PROMPT = (
    "You are a Python programming tutor. Help students learn Python concepts "
    "including variables, functions, loops, classes, and error handling. "
    "Always provide code examples and explain step by step."
)


class TestDriftTrackerAlwaysMode:
    def test_basic_turn(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, mode="always")
        result = tracker.record_turn(
            "How do loops work?",
            "Python loops iterate using for and while. Example: for i in range(5): print(i)"
        )
        assert result.drift is not None
        assert 0 <= result.drift.score <= 100
        assert result.explanation is not None
        assert len(result.explanation) > 0

    def test_drift_increases_off_topic(self):
        tracker = DriftTracker(
            system_prompt=SYSTEM_PROMPT, mode="always", decay_rate=0.92
        )

        r1 = tracker.record_turn(
            "Explain functions",
            "Python functions use def keyword. Example: def greet(name): return name. "
            "Functions, variables, loops, classes, error handling, code examples."
        )

        for _ in range(6):
            tracker.record_turn(
                "Tell me about cooking",
                "Pasta carbonara needs eggs, parmesan, pancetta. "
                "Mix and serve on a plate with fresh pepper."
            )

        r_final = tracker.check()
        assert r_final.drift.score < r1.drift.score

    def test_explanation_provided(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, mode="always")
        result = tracker.record_turn(
            "What is the weather?",
            "Today is sunny with a high of 75 degrees."
        )
        assert result.explanation is not None

    def test_managed_context_returned(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT)
        result = tracker.record_turn("Hi", "Hello!")
        assert SYSTEM_PROMPT in result.managed_context


class TestDriftTrackerOndemandMode:
    def test_no_drift_in_ondemand(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, mode="ondemand")
        result = tracker.record_turn("Hi", "Hello!")
        assert result.drift is None
        assert result.explanation is None

    def test_check_works_in_ondemand(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, mode="ondemand")
        tracker.record_turn("Hi", "Hello!")
        report = tracker.check()
        assert report.drift is not None
        assert 0 <= report.drift.score <= 100
        assert report.explanation is not None

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            DriftTracker(system_prompt="test", mode="invalid")


class TestDriftTrackerSessions:
    def test_end_session(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT)
        tracker.record_turn("Hi", "Hello, let me help with Python!")
        tracker.record_turn("Explain vars", "Variables store data in Python.")

        report = tracker.end_session()
        assert report is not None
        assert report.drift.score > 0
        assert tracker.session_number == 2
        assert tracker.turn_count == 0  # Reset after end_session

    def test_end_empty_session(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT)
        assert tracker.end_session() is None

    def test_session_summaries_in_context(self):
        tracker = DriftTracker(
            system_prompt=SYSTEM_PROMPT, max_summary_sessions=2
        )

        tracker.record_turn("Explain loops", "Loops iterate with for and while in Python.")
        tracker.end_session()

        tracker.record_turn("Explain classes", "Classes are blueprints for objects in Python.")
        tracker.end_session()

        ctx = tracker.get_managed_context()
        assert "Session 1" in ctx
        assert "Session 2" in ctx


class TestDriftTrackerPersistence:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem_path = os.path.join(self.tmpdir, ".session_memory")

    def test_persist_creates_file(self):
        tracker = DriftTracker(
            system_prompt=SYSTEM_PROMPT,
            persist=True,
            persist_path=self.mem_path,
        )
        tracker.record_turn("Hi", "Hello!")
        assert os.path.exists(self.mem_path)

    def test_persist_and_restore(self):
        # Session 1
        t1 = DriftTracker(
            system_prompt=SYSTEM_PROMPT,
            persist=True,
            persist_path=self.mem_path,
        )
        t1.record_turn("Explain vars", "Variables store data in Python code.")
        t1.end_session()

        # Session 2 — should restore from memory
        # end_session bumps to 2, then save persists session_count=2
        # On load: session_count(2) + 1 = 3 (next session to start)
        t2 = DriftTracker(
            system_prompt=SYSTEM_PROMPT,
            persist=True,
            persist_path=self.mem_path,
        )
        assert t2.session_number >= 2
        ctx = t2.get_managed_context()
        assert "Session 1" in ctx

    def test_reset_deletes_memory(self):
        tracker = DriftTracker(
            system_prompt=SYSTEM_PROMPT,
            persist=True,
            persist_path=self.mem_path,
        )
        tracker.record_turn("Hi", "Hello!")
        tracker.reset()
        assert not os.path.exists(self.mem_path)


class TestDriftTrackerContextControl:
    def test_freeze_context(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT)
        tracker.freeze_context()
        assert tracker.is_frozen is True

    def test_unfreeze_context(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, frozen=True)
        tracker.unfreeze_context()
        assert tracker.is_frozen is False

    def test_clear_history(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT)
        tracker.record_turn("Hi", "Hello!")
        tracker.end_session()
        tracker.clear_history()
        ctx = tracker.get_managed_context()
        assert "Session" not in ctx

    def test_frozen_prevents_session_summary(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, frozen=True)
        tracker.record_turn("Hi", "Hello!")
        # end_session should not add summary when frozen
        report = tracker.end_session()
        assert report is not None
        ctx = tracker.get_managed_context()
        assert "Session" not in ctx


class TestDriftTrackerFewShots:
    def test_few_shots_in_context(self):
        tracker = DriftTracker(
            system_prompt="You are a Python tutor.",
            few_shot_examples=[
                FewShotExample(user="What is a var?", assistant="A variable stores data."),
            ],
        )
        ctx = tracker.get_managed_context()
        assert "variable stores data" in ctx

    def test_to_dict(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT)
        result = tracker.record_turn("Hi", "Hello!")
        d = result.to_dict()
        assert "drift" in d
        assert "explanation" in d


class TestDriftTrackerMarkdownStripping:
    def test_strip_md_enabled(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, strip_md=True)
        result = tracker.record_turn(
            "Show code",
            "Here is code:\n```python\ndef foo(): pass\n```\nDone."
        )
        # The raw response should be in the result
        assert "```python" in result.assistant_response
        # But the session should have stripped markdown
        last_turn = tracker.session.assistant_turns[-1]
        assert "```" not in last_turn.content

    def test_strip_md_disabled(self):
        tracker = DriftTracker(system_prompt=SYSTEM_PROMPT, strip_md=False)
        tracker.record_turn("Show code", "```python\nprint('hi')\n```")
        last_turn = tracker.session.assistant_turns[-1]
        assert "```" in last_turn.content
