"""Tests for markdown stripping utilities."""

import pytest
from context_decay_drift.utils.markdown import strip_markdown


class TestStripMarkdown:
    def test_code_blocks(self):
        text = "Here is code:\n```python\ndef hello():\n    print('hi')\n```\nDone."
        result = strip_markdown(text)
        assert "def hello" not in result
        assert "Done" in result

    def test_inline_code(self):
        text = "Use the `print()` function to output."
        result = strip_markdown(text)
        assert "`" not in result
        assert "function" in result

    def test_headers(self):
        text = "# Title\n## Subtitle\nContent here."
        result = strip_markdown(text)
        assert "#" not in result
        assert "Title" in result
        assert "Content here" in result

    def test_bold(self):
        text = "This is **bold** and __also bold__."
        result = strip_markdown(text)
        assert "**" not in result
        assert "__" not in result
        assert "bold" in result

    def test_italic(self):
        text = "This is *italic* text."
        result = strip_markdown(text)
        assert "*" not in result
        assert "italic" in result

    def test_links(self):
        text = "Visit [Google](https://google.com) for more."
        result = strip_markdown(text)
        assert "Google" in result
        assert "https://" not in result

    def test_images(self):
        text = "See ![alt text](image.png) here."
        result = strip_markdown(text)
        assert "![" not in result

    def test_blockquotes(self):
        text = "> This is a quote\n> Another line"
        result = strip_markdown(text)
        assert ">" not in result
        assert "quote" in result

    def test_urls(self):
        text = "Check https://example.com/path for details."
        result = strip_markdown(text)
        assert "https://" not in result
        assert "details" in result

    def test_tables(self):
        text = "| Col1 | Col2 |\n|------|------|\n| A | B |"
        result = strip_markdown(text)
        assert "|" not in result
        assert "Col1" in result

    def test_preserves_plain_text(self):
        text = "Just a normal sentence with no formatting."
        result = strip_markdown(text)
        assert result == text

    def test_collapse_whitespace(self):
        text = "Too   many    spaces   here"
        result = strip_markdown(text)
        assert "  " not in result
