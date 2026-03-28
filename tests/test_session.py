"""Tests for session management."""

import pytest
from context_drift_analyzer.core.session import Session, Turn


class TestSession:
    def test_create_session_with_defaults(self):
        s = Session(system_prompt="You are a helper.")
        assert s.system_prompt == "You are a helper."
        assert s.session_id  # auto-generated
        assert len(s.session_id) == 12
        assert s.turn_count == 0
        assert s.turns == []

    def test_create_session_with_custom_id(self):
        s = Session(system_prompt="test", session_id="my-session")
        assert s.session_id == "my-session"

    def test_add_user_message(self):
        s = Session(system_prompt="test")
        turn = s.add_user_message("Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.turn_number == 1
        assert s.turn_count == 1

    def test_add_assistant_message(self):
        s = Session(system_prompt="test")
        s.add_user_message("Hi")
        turn = s.add_assistant_message("Hello! How can I help?")
        assert turn.role == "assistant"
        assert turn.turn_number == 1  # same exchange as user message
        assert s.turn_count == 1  # 1 Q&A exchange

    def test_assistant_turns_filter(self):
        s = Session(system_prompt="test")
        s.add_user_message("Q1")
        s.add_assistant_message("A1")
        s.add_user_message("Q2")
        s.add_assistant_message("A2")
        assert len(s.assistant_turns) == 2
        assert all(t.role == "assistant" for t in s.assistant_turns)

    def test_user_turns_filter(self):
        s = Session(system_prompt="test")
        s.add_user_message("Q1")
        s.add_assistant_message("A1")
        assert len(s.user_turns) == 1

    def test_get_recent_context(self):
        s = Session(system_prompt="test")
        for i in range(10):
            s.add_user_message(f"Q{i}")
            s.add_assistant_message(f"A{i}")
        recent = s.get_recent_context(n=4)
        assert len(recent) == 4
        assert recent[-1].content == "A9"

    def test_get_full_text(self):
        s = Session(system_prompt="test")
        s.add_user_message("Hello")
        s.add_assistant_message("World")
        assert s.get_full_text() == "Q1: Hello\nA1: World"

    def test_reset(self):
        s = Session(system_prompt="Keep this")
        s.add_user_message("Q1")
        s.add_assistant_message("A1")
        s.reset()
        assert s.turn_count == 0
        assert s.turns == []
        assert s.system_prompt == "Keep this"

    def test_turn_counter_increments(self):
        s = Session(system_prompt="test")
        s.add_user_message("1")       # exchange 1
        s.add_user_message("2")       # exchange 2
        s.add_assistant_message("3")  # still exchange 2 (pairs with last user msg)
        assert s.turn_count == 2  # 2 user messages = 2 exchanges
        assert s.turns[-1].turn_number == 2
