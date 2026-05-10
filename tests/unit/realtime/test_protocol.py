"""Unit tests for realtime.protocol."""

from __future__ import annotations

import dataclasses

import pytest

from src.shared.python.realtime.protocol import Subscription, validate_channel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    [
        "pose/canonical",
        "engine/mujoco/state",
        "target/active",
        "session/marker",
        "a/b",
        "scope1/topic_2",
        "x/y/z",
        "scope/topic_with_underscores/sub_topic",
    ],
)
def test_validate_channel_accepts_valid_names(name: str) -> None:
    validate_channel(name)  # should not raise


@pytest.mark.parametrize(
    "name",
    [
        "",
        "noslash",
        "/leading",
        "trailing/",
        "Scope/topic",  # uppercase
        "1scope/topic",  # leading digit
        "scope/Topic",  # uppercase in second segment
        "scope//topic",  # empty segment
        "scope/topic-extra",  # hyphen
        "scope/topic.x",  # dot
        "scope/<wild>",  # angle brackets
    ],
)
def test_validate_channel_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_channel(name)


def test_validate_channel_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        validate_channel(123)  # type: ignore[arg-type]


def test_subscription_is_frozen_and_calls_unsubscribe() -> None:
    calls: list[int] = []

    sub = Subscription(
        channel="pose/canonical",
        callback=lambda _payload: None,
        _unsubscribe=lambda: calls.append(1),
    )

    sub.unsubscribe()
    assert calls == [1]

    with pytest.raises(dataclasses.FrozenInstanceError):
        # Frozen dataclass should prevent attribute assignment.
        sub.channel = "other/topic"  # type: ignore[misc]
