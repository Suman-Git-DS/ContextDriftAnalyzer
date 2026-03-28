"""Tests for the DriftAnalyzer."""

import pytest
from context_drift_analyzer.core.analyzer import DriftAnalyzer
from context_drift_analyzer.core.session import Session
from context_drift_analyzer.core.scorer import DriftVerdict
from context_drift_analyzer.strategies.keyword import KeywordStrategy


SYSTEM_PROMPT = (
    "You are a Python programming tutor. Help students learn Python concepts "
    "including variables, functions, loops, classes, and error handling. "
    "Always provide code examples and explain step by step."
)


class TestDriftAnalyzer:
    def test_fresh_session(self):
        """Empty session should return score 100."""
        analyzer = DriftAnalyzer()
        session = Session(system_prompt=SYSTEM_PROMPT)
        result = analyzer.analyze(session)
        assert result.score == 100.0
        assert result.verdict == DriftVerdict.FRESH

    def test_on_topic_stays_high(self):
        """On-topic conversation should maintain a decent score."""
        analyzer = DriftAnalyzer(decay_rate=0.98)
        session = Session(system_prompt=SYSTEM_PROMPT)

        session.add_user_message("How do I define a function in Python?")
        session.add_assistant_message(
            "In Python, you define a function using the def keyword. "
            "Here is a code example that explains step by step: "
            "def greet(name): return f'Hello {name}'. "
            "Functions help organize your Python programming code."
        )

        result = analyzer.analyze(session)
        assert result.score > 30.0
        assert result.verdict != DriftVerdict.CRITICAL

    def test_drift_increases_over_turns(self):
        """Score should decrease as conversation goes off-topic."""
        analyzer = DriftAnalyzer(decay_rate=0.92)
        session = Session(system_prompt=SYSTEM_PROMPT)

        # First few on-topic turns
        session.add_user_message("Explain Python variables")
        session.add_assistant_message(
            "Python variables store values. Functions use variables. "
            "Here are code examples for learning Python step by step."
        )
        score_early = analyzer.analyze(session).score

        # Add many off-topic turns
        for i in range(10):
            session.add_user_message(f"Tell me about cooking recipe {i}")
            session.add_assistant_message(
                f"Recipe {i}: Mix flour and sugar. Bake at 350 degrees. "
                "Add chocolate chips and vanilla extract for flavor."
            )

        score_late = analyzer.analyze(session).score
        assert score_late < score_early

    def test_custom_strategy(self):
        """Analyzer should work with a single custom strategy."""
        analyzer = DriftAnalyzer(strategies=[KeywordStrategy(top_n=10)])
        session = Session(system_prompt=SYSTEM_PROMPT)
        session.add_user_message("Hello")
        session.add_assistant_message("Hello! Let me help you learn Python.")

        result = analyzer.analyze(session)
        assert 0 <= result.score <= 100

    def test_invalid_decay_rate(self):
        with pytest.raises(ValueError, match="decay_rate"):
            DriftAnalyzer(decay_rate=0.0)
        with pytest.raises(ValueError, match="decay_rate"):
            DriftAnalyzer(decay_rate=1.5)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            DriftAnalyzer(window_size=-1)

    def test_is_effective_shortcut(self):
        analyzer = DriftAnalyzer()
        session = Session(system_prompt=SYSTEM_PROMPT)
        assert analyzer.is_effective(session) is True

    def test_needs_reset_shortcut(self):
        analyzer = DriftAnalyzer()
        session = Session(system_prompt=SYSTEM_PROMPT)
        assert analyzer.needs_reset(session) is False

    def test_score_clamped_to_range(self):
        analyzer = DriftAnalyzer()
        session = Session(system_prompt=SYSTEM_PROMPT)
        session.add_user_message("test")
        session.add_assistant_message("test")
        result = analyzer.analyze(session)
        assert 0 <= result.score <= 100

    def test_metadata_populated(self):
        analyzer = DriftAnalyzer(decay_rate=0.9, window_size=3)
        session = Session(system_prompt=SYSTEM_PROMPT)
        session.add_user_message("Hi")
        session.add_assistant_message("Hello")
        result = analyzer.analyze(session)
        assert "raw_score" in result.metadata
        assert "decay_factor" in result.metadata
        assert result.metadata["decay_rate"] == 0.9
        assert result.metadata["window_size"] == 3


class TestDriftProgression:
    """Test realistic multi-turn drift progression."""

    def test_five_session_scenario(self):
        """Simulate the scenario from the spec: score drops over sessions."""
        analyzer = DriftAnalyzer(decay_rate=0.93, window_size=0)
        session = Session(system_prompt=SYSTEM_PROMPT)

        scores = []

        # Session 1: Very on-topic
        session.add_user_message("Teach me about Python variables")
        session.add_assistant_message(
            "Python variables store data values. You create a variable "
            "by assigning a value: x = 5. Variables can hold strings, "
            "numbers, lists. Here's a code example step by step."
        )
        scores.append(analyzer.analyze(session).score)

        # Session 2: Still on topic but starting to wander
        session.add_user_message("What about data types?")
        session.add_assistant_message(
            "Python has several data types: int, float, str, list, dict. "
            "Each type represents different kinds of data in your programs."
        )
        scores.append(analyzer.analyze(session).score)

        # Session 3: Drifting
        session.add_user_message("Can you help me with my resume?")
        session.add_assistant_message(
            "Sure! A good resume should highlight your experience, "
            "skills, and education. Use action verbs and quantify achievements."
        )
        scores.append(analyzer.analyze(session).score)

        # Session 4: More off topic
        session.add_user_message("What's a good dinner recipe?")
        session.add_assistant_message(
            "Try making pasta carbonara! You'll need eggs, parmesan, "
            "pancetta, and spaghetti. Cook the pasta, fry the pancetta."
        )
        scores.append(analyzer.analyze(session).score)

        # Session 5: Completely off topic
        session.add_user_message("Tell me about the weather")
        session.add_assistant_message(
            "Today's forecast shows sunny skies with a high of 75. "
            "Tomorrow brings possible showers in the afternoon."
        )
        scores.append(analyzer.analyze(session).score)

        # Verify monotonically decreasing trend (with some tolerance)
        assert scores[0] > scores[-1], (
            f"First score ({scores[0]:.1f}) should be higher than last ({scores[-1]:.1f})"
        )
        # Score should drop noticeably
        assert scores[-1] < scores[0] * 0.9, (
            f"Expected noticeable drop: {scores[0]:.1f} -> {scores[-1]:.1f}"
        )
