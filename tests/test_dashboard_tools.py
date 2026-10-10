from types import SimpleNamespace

import pytest

from dashboard_tools import latest_user_message, parse_navigation_ack


def test_latest_user_message_uses_actual_most_recent_user_turn():
    history = SimpleNamespace(items=[
        SimpleNamespace(type="message", role="user", text_content="What is my balance?"),
        SimpleNamespace(type="message", role="assistant", text_content="I will check."),
        SimpleNamespace(type="function_call_output", role="assistant", text_content="ignored tool output"),
    ])
    assert latest_user_message(history) == "What is my balance?"


def test_latest_user_message_is_empty_when_no_user_turn_exists():
    history = SimpleNamespace(items=[SimpleNamespace(type="message", role="assistant", text_content="Hello")])
    assert latest_user_message(history) == ""


def test_navigation_requires_success_ack_for_the_requested_existing_page():
    assert parse_navigation_ack('{"success":true,"page":"trade-journal"}', "trade-journal")["success"] is True
    with pytest.raises(ValueError):
        parse_navigation_ack('{"success":false,"page":"trade-journal"}', "trade-journal")
    with pytest.raises(ValueError):
        parse_navigation_ack('{"success":true,"page":"settings"}', "trade-journal")
    with pytest.raises(ValueError):
        parse_navigation_ack('{"success":true,"page":"withdrawals"}', "withdrawals")
