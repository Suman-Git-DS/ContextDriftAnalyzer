"""Drift explainer — generates 1-2 line human-readable explanations of why drift occurred.

Supports two modes:
1. Local (default): Rule-based explanation using keyword analysis — free, no API calls.
2. Custom callable: Provide your own explanation function (e.g., LLM-powered).
"""

from __future__ import annotations

from typing import Callable, Optional

from context_decay_drift.utils.text import extract_keywords, tokenize


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
    ) -> str:
        """Generate a 1-2 line explanation of why drift occurred.

        Args:
            original_context: The original system prompt + few-shot context.
            recent_text: The recent assistant response text.
            score: The drift score (0-100).

        Returns:
            A short explanation string.
        """
        return self._explain_fn(original_context, recent_text, score)

    @staticmethod
    def _local_explain(
        original_context: str,
        recent_text: str,
        score: float,
    ) -> str:
        """Rule-based local explanation — no API calls needed."""
        original_keywords = set(extract_keywords(original_context, top_n=20))
        response_keywords = set(extract_keywords(recent_text, top_n=20))

        if not original_keywords:
            return "No keywords detected in original context."

        missing = original_keywords - response_keywords
        new_topics = response_keywords - original_keywords

        parts = []

        if score >= 90:
            return "Context is well-preserved. Responses closely align with original instructions."

        if score >= 75:
            if missing:
                top_missing = sorted(missing)[:3]
                parts.append(
                    f"Mild drift: key topics fading — {', '.join(top_missing)} "
                    f"less prominent in recent responses."
                )
            else:
                parts.append("Mild drift detected, but core topics still present.")

        elif score >= 55:
            if missing:
                top_missing = sorted(missing)[:4]
                parts.append(
                    f"Moderate drift: original topics [{', '.join(top_missing)}] "
                    f"are underrepresented."
                )
            if new_topics:
                top_new = sorted(new_topics)[:3]
                parts.append(
                    f"New topics emerged: {', '.join(top_new)}."
                )

        elif score >= 35:
            hit_rate = len(original_keywords & response_keywords) / len(original_keywords) * 100
            parts.append(
                f"Significant drift: only {hit_rate:.0f}% of original context "
                f"keywords present in recent responses."
            )
            if new_topics:
                top_new = sorted(new_topics)[:4]
                parts.append(f"Conversation shifted toward: {', '.join(top_new)}.")

        else:
            hit_rate = len(original_keywords & response_keywords) / len(original_keywords) * 100
            parts.append(
                f"Critical drift: {hit_rate:.0f}% keyword retention. "
                f"The conversation has largely departed from its original purpose."
            )
            if new_topics:
                top_new = sorted(new_topics)[:5]
                parts.append(f"Now discussing: {', '.join(top_new)}.")

        return " ".join(parts) if parts else "Drift detected."
