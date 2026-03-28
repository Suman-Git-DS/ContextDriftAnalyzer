"""Markdown stripping utilities.

Removes markdown formatting before embedding to avoid false-positive
drift caused by formatting changes rather than content changes.
"""

from __future__ import annotations

import re


def strip_markdown(text: str) -> str:
    """Remove markdown formatting from text for cleaner embeddings.

    Strips: code blocks, inline code, headers, bold, italic,
    links, images, tables, blockquotes, horizontal rules.
    """
    # Remove fenced code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", " ", text)

    # Remove inline code (`...`)
    text = re.sub(r"`[^`]+`", " ", text)

    # Remove images ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

    # Remove links [text](url) -> keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove headers (# ... ######)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold (**text** or __text__)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)

    # Remove italic (*text* or _text_)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)

    # Remove blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove table pipes
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"^[-:|\s]+$", "", text, flags=re.MULTILINE)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
