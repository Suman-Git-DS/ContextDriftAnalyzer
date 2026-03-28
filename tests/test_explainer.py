"""Tests for drift explainer."""

import pytest
from context_decay_drift.context.explainer import DriftExplainer


SYSTEM_PROMPT = (
    "You are a Python programming tutor. Help students learn Python concepts "
    "including variables, functions, loops, classes, and error handling."
)


class TestDriftExplainer:
    def setup_method(self):
        self.explainer = DriftExplainer()

    def test_fresh_context(self):
        explanation = self.explainer.explain(
            SYSTEM_PROMPT,
            "Python variables store data. Functions organize code. Loops iterate.",
            score=95.0,
        )
        assert "well-preserved" in explanation.lower() or "align" in explanation.lower()

    def test_mild_drift(self):
        explanation = self.explainer.explain(
            SYSTEM_PROMPT,
            "Python is a great language for data analysis and machine learning.",
            score=80.0,
        )
        assert len(explanation) > 10
        assert "mild" in explanation.lower() or "drift" in explanation.lower() or "topic" in explanation.lower()

    def test_moderate_drift(self):
        explanation = self.explainer.explain(
            SYSTEM_PROMPT,
            "The stock market showed gains today. Tech companies reported earnings.",
            score=60.0,
        )
        assert len(explanation) > 10

    def test_severe_drift(self):
        explanation = self.explainer.explain(
            SYSTEM_PROMPT,
            "Pasta carbonara is made with eggs, parmesan, and pancetta.",
            score=40.0,
        )
        assert "drift" in explanation.lower() or "keyword" in explanation.lower()

    def test_critical_drift(self):
        explanation = self.explainer.explain(
            SYSTEM_PROMPT,
            "The weather forecast shows sunny skies.",
            score=15.0,
        )
        assert "critical" in explanation.lower() or "departed" in explanation.lower()

    def test_custom_explain_fn(self):
        custom_fn = lambda ctx, text, score: f"Custom: score is {score}"
        explainer = DriftExplainer(explain_fn=custom_fn)
        result = explainer.explain("context", "response", 50.0)
        assert result == "Custom: score is 50.0"

    def test_empty_context(self):
        explanation = self.explainer.explain("", "Some response", 50.0)
        assert isinstance(explanation, str)
