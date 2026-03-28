"""Tests for the drop-in client wrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from context_drift_analyzer.wrap import (
    DriftClientWrapper,
    _AnthropicMessagesProxy,
    _OpenAIChatProxy,
    _OpenAICompletionsProxy,
    wrap,
)


# ---------------------------------------------------------------------------
# Helpers – fake clients that look like OpenAI / Anthropic
# ---------------------------------------------------------------------------

def _make_openai_client():
    """Return a mock that mimics openai.OpenAI() shape."""
    choice = SimpleNamespace(message=SimpleNamespace(content="Loops let you repeat code."))
    response = SimpleNamespace(choices=[choice])

    completions = MagicMock()
    completions.create = MagicMock(return_value=response)

    chat = SimpleNamespace(completions=completions)
    client = MagicMock()
    client.chat = chat
    # Make it detectable as OpenAI
    type(client).__name__ = "OpenAI"
    type(client).__module__ = "openai"
    return client


def _make_anthropic_client():
    """Return a mock that mimics anthropic.Anthropic() shape."""
    text_block = SimpleNamespace(type="text", text="Loops let you repeat code.")
    response = SimpleNamespace(content=[text_block])

    messages = MagicMock()
    messages.create = MagicMock(return_value=response)

    client = MagicMock()
    client.messages = messages
    type(client).__name__ = "Anthropic"
    type(client).__module__ = "anthropic"
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWrapFunction:
    def test_returns_wrapper(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="Test prompt.")
        assert isinstance(wrapped, DriftClientWrapper)

    def test_wrap_passes_kwargs(self):
        client = _make_openai_client()
        wrapped = wrap(
            client,
            system_prompt="Test",
            mode="ondemand",
            persist=False,
            max_summary_sessions=5,
        )
        assert wrapped.tracker.mode == "ondemand"


class TestOpenAIProxy:
    def test_detects_openai_client(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        assert hasattr(wrapped, "chat")
        assert isinstance(wrapped.chat, _OpenAIChatProxy)
        assert isinstance(wrapped.chat.completions, _OpenAICompletionsProxy)

    def test_create_calls_original(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        response = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "How do loops work?"}],
        )
        client.chat.completions.create.assert_called_once()

    def test_drift_attached_to_response(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        response = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "How do loops work?"}],
        )
        assert hasattr(response, "_drift")
        assert hasattr(response, "_drift_explanation")
        assert hasattr(response, "_managed_context")

    def test_drift_score_is_numeric(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        response = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "How do loops work?"}],
        )
        assert isinstance(response._drift.score, float)
        assert 0.0 <= response._drift.score <= 100.0

    def test_extracts_user_message_from_messages(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a tutor."},
                {"role": "user", "content": "Tell me about Python"},
            ],
        )
        # Verify turn was recorded
        session = wrapped.tracker.session
        assert len(session.turns) == 2  # user + assistant (1 exchange)
        assert session.turn_count == 1  # 1 Q&A exchange
        assert session.turns[0].content == "Tell me about Python"

    def test_completions_proxy_getattr_fallthrough(self):
        client = _make_openai_client()
        client.chat.completions.some_attr = "hello"
        wrapped = wrap(client, system_prompt="Test.")
        assert wrapped.chat.completions.some_attr == "hello"


class TestAnthropicProxy:
    def test_detects_anthropic_client(self):
        client = _make_anthropic_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        assert hasattr(wrapped, "messages")
        assert isinstance(wrapped.messages, _AnthropicMessagesProxy)

    def test_create_calls_original(self):
        client = _make_anthropic_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        response = wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            system="You are a tutor.",
            messages=[{"role": "user", "content": "How do loops work?"}],
            max_tokens=200,
        )
        client.messages.create.assert_called_once()

    def test_drift_attached_to_response(self):
        client = _make_anthropic_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        response = wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "How do loops work?"}],
            max_tokens=200,
        )
        assert hasattr(response, "_drift")
        assert hasattr(response, "_drift_explanation")
        assert hasattr(response, "_managed_context")

    def test_handles_content_blocks(self):
        """Anthropic messages can have content as list of blocks."""
        client = _make_anthropic_client()
        # Override messages to use list-style content
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Part 1. "), SimpleNamespace(type="text", text="Part 2.")]
        )
        wrapped = wrap(client, system_prompt="You are a tutor.")
        response = wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Tell me about loops"}],
            max_tokens=200,
        )
        # Should concatenate both blocks
        session = wrapped.tracker.session
        assistant_turns = [t for t in session.turns if t.role == "assistant"]
        assert "Part 1" in assistant_turns[0].content
        assert "Part 2" in assistant_turns[0].content

    def test_handles_user_content_blocks(self):
        """Anthropic user messages can be a list of content blocks."""
        client = _make_anthropic_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        wrapped.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }],
            max_tokens=200,
        )
        session = wrapped.tracker.session
        user_turns = [t for t in session.turns if t.role == "user"]
        assert "Hello" in user_turns[0].content
        assert "World" in user_turns[0].content

    def test_messages_proxy_getattr_fallthrough(self):
        client = _make_anthropic_client()
        client.messages.some_attr = "hello"
        wrapped = wrap(client, system_prompt="Test.")
        assert wrapped.messages.some_attr == "hello"


class TestDriftClientWrapper:
    def test_drift_check(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "How do loops work?"}],
        )
        report = wrapped.drift_check()
        assert hasattr(report, "drift")
        assert hasattr(report, "explanation")

    def test_end_session(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "How do loops work?"}],
        )
        report = wrapped.end_session()
        assert report is not None

    def test_get_managed_context(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        ctx = wrapped.get_managed_context()
        assert "You are a tutor" in ctx

    def test_reset(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        wrapped.reset()
        session = wrapped.tracker.session
        assert len(session.turns) == 0

    def test_freeze_unfreeze(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        wrapped.freeze_context()
        assert wrapped.tracker.is_frozen is True
        wrapped.unfreeze_context()
        assert wrapped.tracker.is_frozen is False

    def test_clear_history(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # clear_history clears session summaries from context manager
        # (not session turns — that's reset)
        wrapped.clear_history()
        ctx = wrapped.get_managed_context()
        assert "You are a tutor" in ctx

    def test_tracker_property(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        from context_drift_analyzer.tracker import DriftTracker
        assert isinstance(wrapped.tracker, DriftTracker)

    def test_getattr_falls_through_to_client(self):
        client = _make_openai_client()
        client.some_custom_method = MagicMock(return_value="custom")
        wrapped = wrap(client, system_prompt="Test.")
        assert wrapped.some_custom_method() == "custom"

    def test_multiple_turns_accumulate(self):
        client = _make_openai_client()
        wrapped = wrap(client, system_prompt="You are a tutor.")
        for i in range(3):
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Question {i}"}],
            )
        session = wrapped.tracker.session
        assert len(session.turns) == 6  # 3 user + 3 assistant
        assert session.turn_count == 3  # 3 Q&A exchanges


class TestUnknownClient:
    def test_unknown_client_no_proxy_but_tracker_works(self):
        """Wrapping an unknown client — no proxy, but tracker still usable."""
        client = SimpleNamespace(name="custom")
        wrapped = wrap(client, system_prompt="Test.")
        # No proxy attributes set directly on wrapper
        assert "chat" not in wrapped.__dict__
        assert "messages" not in wrapped.__dict__
        # Direct tracker usage still works
        report = wrapped.drift_check()
        assert report is not None
