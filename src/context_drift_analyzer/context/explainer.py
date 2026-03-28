"""Drift explainer — generates 1-2 line human-readable explanations of why drift occurred.

Supports two modes:
1. Local (default): Score-based explanation with topic analysis — free, no API calls.
2. Custom callable: Provide your own explanation function (e.g., LLM-powered).
"""

from __future__ import annotations

from typing import Callable, Optional

from context_drift_analyzer.utils.text import extract_keywords, tokenize


def _fuzzy_overlap(set_a: set[str], set_b: set[str]) -> set[str]:
    """Find words that match exactly or share a common stem (prefix >= 4 chars)."""
    matched = set_a & set_b  # exact matches
    remaining_a = set_a - matched
    remaining_b = set_b - matched
    for word_a in remaining_a:
        for word_b in remaining_b:
            # Check if one is a prefix of the other (handles offer/offers, account/accounts, etc.)
            if len(word_a) >= 4 and len(word_b) >= 4:
                if word_a.startswith(word_b[:4]) or word_b.startswith(word_a[:4]):
                    matched.add(word_a)
                    break
    return matched


class DriftExplainer:
    """Generates human-readable explanations for detected drift.

    Args:
        explain_fn: Optional custom explanation function.
            Signature: (original_context: str, recent_text: str, score: float) -> str
            If not provided, uses a local rule-based explainer (no API calls).
    """

    def __init__(
        self,
        explain_fn: Optional[Callable[[str, str, float], str]] = None,
    ):
        self._explain_fn = explain_fn or self._local_explain

    def explain(
        self,
        original_context: str,
        recent_text: str,
        score: float,
        user_message: str = "",
    ) -> str:
        """Generate a 1-2 line explanation of why drift occurred.

        Args:
            original_context: The original system prompt + few-shot context.
            recent_text: The recent assistant response text.
            score: The drift score (0-100).
            user_message: The user's question (used to detect off-topic queries).

        Returns:
            A short explanation string.
        """
        if self._explain_fn is not self._local_explain:
            # Custom explain_fn uses the old 3-arg signature
            return self._explain_fn(original_context, recent_text, score)
        return self._local_explain(original_context, recent_text, score, user_message)

    @staticmethod
    def _local_explain(
        original_context: str,
        recent_text: str,
        score: float,
        user_message: str = "",
    ) -> str:
        """Score-based explanation with topic analysis — no API calls needed."""
        if not recent_text.strip():
            return "No response to evaluate."

        # Extract meaningful topics
        original_topics = set(extract_keywords(original_context, top_n=15))
        response_topics = set(extract_keywords(recent_text, top_n=10))

        # Identify shared and divergent topics (fuzzy matching for plurals, etc.)
        shared = _fuzzy_overlap(original_topics, response_topics)
        new_topics = response_topics - shared

        # Detect whether the user's question is off-topic
        user_off_topic = False
        user_new_topics: list[str] = []
        if user_message.strip():
            user_topics = set(extract_keywords(user_message, top_n=8))
            # Compare user question against BOTH original context AND response topics
            # (the response may contain domain terms not in the short system prompt)
            broad_context_topics = original_topics | response_topics
            user_shared = _fuzzy_overlap(broad_context_topics, user_topics)
            user_new_topics = sorted(user_topics - user_shared)
            # User is off-topic only if their question introduces new topics
            # with zero or near-zero overlap with the broader context
            if not user_shared and user_new_topics:
                user_off_topic = True
            elif len(user_new_topics) >= 2 and not user_shared:
                user_off_topic = True

        # Build explanation considering both user question and assistant response
        if score >= 90:
            if user_off_topic:
                return (
                    f"User asked off-topic question ({', '.join(user_new_topics[:3])}). "
                    f"Assistant stayed on track and redirected to original context."
                )
            return "Context is well-preserved. Responses closely align with original instructions."

        if score >= 75:
            if user_off_topic:
                return (
                    f"User asked off-topic question ({', '.join(user_new_topics[:3])}). "
                    f"Assistant redirected to banking topics, but conversation is starting to drift."
                )
            if shared:
                return f"Mild drift detected, but core topics still present ({', '.join(sorted(shared)[:3])})."
            return "Mild drift: response is related but uses different terminology than the original context."

        if score >= 55:
            if user_off_topic:
                return (
                    f"User went off-topic ({', '.join(user_new_topics[:3])}). "
                    f"Assistant attempted to redirect, but drift is accumulating."
                )
            if shared and new_topics:
                return (
                    f"Moderate drift: some original topics present ({', '.join(sorted(shared)[:3])}), "
                    f"but new topics emerging ({', '.join(sorted(new_topics)[:3])})."
                )
            if new_topics:
                return f"Moderate drift: conversation shifting toward new topics ({', '.join(sorted(new_topics)[:4])})."
            return "Moderate drift: response is loosely related to original context."

        if score >= 35:
            if user_off_topic:
                return (
                    f"User went off-topic ({', '.join(user_new_topics[:3])}). "
                    f"Significant drift: conversation has moved away from original purpose."
                )
            if new_topics:
                return (
                    f"Significant drift: conversation has moved away from original purpose. "
                    f"Now focused on: {', '.join(sorted(new_topics)[:4])}."
                )
            return "Significant drift: response shows weak alignment with original context. Consider resetting."

        # Critical
        if user_off_topic:
            return (
                f"User went off-topic ({', '.join(user_new_topics[:3])}). "
                f"Critical drift: conversation has departed from its original purpose. Reset recommended."
            )
        if new_topics:
            return (
                f"Critical drift: conversation has largely departed from its original purpose. "
                f"Current topics ({', '.join(sorted(new_topics)[:4])}) are unrelated to original context."
            )
        return "Critical drift: response has very low alignment with original context. Reset recommended."
