"""Tests for realtime protocol primitives."""

from __future__ import annotations

import pytest

from src.shared.python.realtime.protocol import Subscription, validate_channel


class TestValidateChannel:
    @pytest.mark.parametrize(
        "name",
        [
            "pose/canonical",
            "engine/state",
            "scope/topic/sub",
            "a/b",
            "a1/b2/c3",
            "abc_def/ghi_jkl",
            "z9/aa_bb/cc",
        ],
    )
    def test_valid_channels(self, name: str) -> None:
        validate_channel(name)  # no raise

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "single",
            "/leading",
            "trailing/",
            "Upper/case",
            "1number/start",
            "a/B",
            "a//b",
            "a/b-c",
            "a/b c",
        ],
    )
    def test_invalid_channels_raise_value_error(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_channel(name)

    @pytest.mark.parametrize("bad", [123, None, 1.0, [], {}, object()])
    def test_non_string_raises_type_error(self, bad) -> None:
        with pytest.raises(TypeError):
            validate_channel(bad)  # type: ignore[arg-type]


class TestSubscription:
    def test_unsubscribe_calls_callable(self) -> None:
        calls = []

        def unsub() -> None:
            calls.append(1)

        sub = Subscription(channel="a/b", callback=lambda d: None, _unsubscribe=unsub)
        assert sub.channel == "a/b"
        sub.unsubscribe()
        assert calls == [1]

    def test_subscription_is_frozen(self) -> None:
        sub = Subscription(
            channel="a/b", callback=lambda d: None, _unsubscribe=lambda: None
        )
        with pytest.raises((AttributeError, Exception)):  # noqa: B017
            sub.channel = "x/y"  # type: ignore[misc]
