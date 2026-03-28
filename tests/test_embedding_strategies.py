"""Tests for embedding-based drift strategies."""

import math
import pytest

from context_drift_analyzer.strategies.embedding_base import EmbeddingStrategy
from context_drift_analyzer.strategies.callable_embedding import CallableEmbeddingStrategy


SYSTEM_PROMPT = (
    "You are a Python programming tutor. Help students learn Python concepts "
    "including variables, functions, loops, classes, and error handling. "
    "Always provide code examples and explain step by step."
)


# --- Fake embedder for deterministic testing ---

def fake_embedder(text: str) -> list[float]:
    """Simple deterministic embedder for testing.

    Maps text to a vector based on presence of key topic words.
    Dimensions: [python, code, function, loop, class, food, weather, sport]
    """
    text_lower = text.lower()
    topics = ["python", "code", "function", "loop", "class", "food", "weather", "sport"]
    vec = []
    for topic in topics:
        # Count occurrences and normalize
        count = text_lower.count(topic)
        vec.append(float(count))
    # Normalize to unit vector
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    else:
        vec = [0.0] * len(topics)
    return vec


class TestEmbeddingBase:
    def test_cosine_similarity_identical(self):
        vec = [1.0, 2.0, 3.0]
        sim = EmbeddingStrategy._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        sim = EmbeddingStrategy._cosine_similarity(vec_a, vec_b)
        assert abs(sim) < 1e-9

    def test_cosine_similarity_opposite(self):
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        sim = EmbeddingStrategy._cosine_similarity(vec_a, vec_b)
        assert abs(sim - (-1.0)) < 1e-9

    def test_cosine_similarity_zero_vector(self):
        vec_a = [0.0, 0.0]
        vec_b = [1.0, 2.0]
        sim = EmbeddingStrategy._cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_cosine_similarity_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            EmbeddingStrategy._cosine_similarity([1.0], [1.0, 2.0])


class TestCallableEmbeddingStrategy:
    def test_on_topic_scores_high(self):
        strategy = CallableEmbeddingStrategy(
            embed_fn=fake_embedder, strategy_name="test"
        )
        responses = [
            "Python functions and classes help you write code. "
            "Use loops to iterate. Python code examples step by step."
        ]
        score, scores = strategy.score(SYSTEM_PROMPT, responses)
        assert score > 50.0
        assert "test" in scores

    def test_off_topic_scores_low(self):
        strategy = CallableEmbeddingStrategy(
            embed_fn=fake_embedder, strategy_name="test"
        )
        responses = [
            "The food at the restaurant was amazing. "
            "Weather forecast shows rain. Sport event tomorrow."
        ]
        score, _ = strategy.score(SYSTEM_PROMPT, responses)
        assert score < 30.0

    def test_empty_responses_returns_100(self):
        strategy = CallableEmbeddingStrategy(embed_fn=fake_embedder)
        score, _ = strategy.score(SYSTEM_PROMPT, [])
        assert score == 100.0

    def test_custom_name(self):
        strategy = CallableEmbeddingStrategy(
            embed_fn=fake_embedder, strategy_name="my_model"
        )
        assert strategy.name == "my_model"

    def test_default_name(self):
        strategy = CallableEmbeddingStrategy(embed_fn=fake_embedder)
        assert strategy.name == "custom_embedding"

    def test_caching(self):
        call_count = 0

        def counting_embedder(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return fake_embedder(text)

        strategy = CallableEmbeddingStrategy(
            embed_fn=counting_embedder, cache_reference=True
        )

        responses = ["Python code functions"]
        strategy.score(SYSTEM_PROMPT, responses)
        first_count = call_count

        strategy.score(SYSTEM_PROMPT, responses)
        # Reference should be cached, so only 1 new call for the response
        assert call_count == first_count + 1  # only response re-embedded

    def test_no_caching(self):
        call_count = 0

        def counting_embedder(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return fake_embedder(text)

        strategy = CallableEmbeddingStrategy(
            embed_fn=counting_embedder, cache_reference=False
        )

        responses = ["Python code"]
        strategy.score(SYSTEM_PROMPT, responses)
        first_count = call_count

        strategy.score(SYSTEM_PROMPT, responses)
        # Both reference and response re-embedded
        assert call_count == first_count + 2

    def test_clear_cache(self):
        strategy = CallableEmbeddingStrategy(
            embed_fn=fake_embedder, cache_reference=True
        )
        strategy.score(SYSTEM_PROMPT, ["Python code"])
        assert len(strategy._ref_cache) > 0

        strategy.clear_cache()
        assert len(strategy._ref_cache) == 0

    def test_drift_increases_over_off_topic_turns(self):
        """Simulate progressive drift — score should drop."""
        strategy = CallableEmbeddingStrategy(embed_fn=fake_embedder)

        on_topic = ["Python functions and loops with code examples in class"]
        off_topic = ["food weather sport food weather sport"]

        score_on, _ = strategy.score(SYSTEM_PROMPT, on_topic)
        score_off, _ = strategy.score(SYSTEM_PROMPT, off_topic)

        assert score_on > score_off


class TestFewShotWithEmbedding:
    """Test that few-shot examples in the initial context improve drift detection."""

    def test_few_shot_context_used(self):
        from context_drift_analyzer import Session, DriftAnalyzer, FewShotExample

        session = Session(
            system_prompt="You are a Python tutor.",
            few_shot_examples=[
                FewShotExample(
                    user="What is a variable?",
                    assistant="A Python variable stores data. Example: x = 5",
                ),
                FewShotExample(
                    user="How do loops work?",
                    assistant="Python loops iterate with for and while. Example: for i in range(10): print(i)",
                ),
            ],
        )

        # initial_context should include both system prompt and few-shots
        ctx = session.initial_context
        assert "Python tutor" in ctx
        assert "variable stores data" in ctx
        assert "loops iterate" in ctx

    def test_few_shot_affects_drift_score(self):
        from context_drift_analyzer import Session, DriftAnalyzer, FewShotExample

        strategy = CallableEmbeddingStrategy(embed_fn=fake_embedder)
        analyzer = DriftAnalyzer(strategies=[strategy], decay_rate=0.99)

        # Session WITH few-shots about Python code
        session_with = Session(
            system_prompt="You are a Python tutor.",
            few_shot_examples=[
                FewShotExample(
                    user="Explain functions",
                    assistant="Python function code class loop example",
                ),
            ],
        )
        # Use a response that's related to programming/tutoring but not
        # identical to the few-shot examples, so the richer reference
        # (with few-shots about functions/loops) gives a different cosine
        borderline_response = (
            "To learn programming you should practice writing code every day "
            "and start with basic concepts like data types and control flow"
        )
        session_with.add_user_message("Hello")
        session_with.add_assistant_message(borderline_response)

        # Session WITHOUT few-shots
        session_without = Session(system_prompt="You are a Python tutor.")
        session_without.add_user_message("Hello")
        session_without.add_assistant_message(borderline_response)

        score_with = analyzer.analyze(session_with).score
        score_without = analyzer.analyze(session_without).score

        # Both sessions use the same response but different reference contexts.
        # The richer context (with few-shots) should produce a different score.
        assert score_with != score_without
