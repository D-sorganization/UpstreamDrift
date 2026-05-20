"""Tests for realtime channel registry and transport hint resolution."""

from __future__ import annotations

import pytest

from src.shared.python.realtime import channels as ch_mod
from src.shared.python.realtime.channels import (
    get_channel_transport,
    register_channel,
)


class TestRegisterChannel:
    def test_invalid_hint_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency_hint"):
            register_channel("scope/topic", "medium")  # type: ignore[arg-type]

    def test_invalid_literal_name_raises(self) -> None:
        with pytest.raises(ValueError):
            register_channel("BAD-NAME", "low")

    def test_register_literal_low(self) -> None:
        register_channel("tests_chan/low_only", "low")
        assert get_channel_transport("tests_chan/low_only") == "file"

    def test_register_literal_high(self) -> None:
        register_channel("tests_chan/high_only", "high")
        assert get_channel_transport("tests_chan/high_only") == "ws"

    def test_register_wildcard_high(self) -> None:
        register_channel("tests_wild/<name>/state", "high")
        assert get_channel_transport("tests_wild/foo/state") == "ws"
        # Non-matching: missing suffix
        assert get_channel_transport("tests_wild/foo/other") == "file"
        # Non-matching: extra segment in wildcard slot
        assert get_channel_transport("tests_wild/foo/bar/state") == "file"

    def test_register_wildcard_no_suffix(self) -> None:
        register_channel("tests_wild2/<name>", "high")
        assert get_channel_transport("tests_wild2/foo") == "ws"
        # Wildcard captures exactly one segment, not zero or many
        assert get_channel_transport("tests_wild2/foo/bar") == "file"

    def test_multiple_wildcards_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple wildcards"):
            register_channel("a/<x>/b/<y>", "high")


class TestGetChannelTransport:
    def test_unknown_defaults_to_file(self) -> None:
        assert get_channel_transport("zzz_unknown/unknown") == "file"

    def test_builtin_pose_canonical_is_ws(self) -> None:
        assert get_channel_transport("pose/canonical") == "ws"

    def test_builtin_engine_state_wildcard(self) -> None:
        assert get_channel_transport("engine/myname/state") == "ws"

    def test_builtin_target_active_is_file(self) -> None:
        assert get_channel_transport("target/active") == "file"


def test_split_wildcard_helper_returns_none_for_literal() -> None:
    assert ch_mod._split_wildcard("a/b/c") is None


def test_split_wildcard_helper_prefix_suffix() -> None:
    prefix, suffix = ch_mod._split_wildcard("a/<x>/c")
    assert prefix == "a/"
    assert suffix == "/c"
