"""Unit tests for realtime.channels."""

from __future__ import annotations

import pytest

from src.shared.python.realtime import channels


pytestmark = pytest.mark.unit


def test_well_known_channels_have_expected_transports() -> None:
    assert channels.get_channel_transport("pose/canonical") == "ws"
    assert channels.get_channel_transport("target/active") == "file"
    assert channels.get_channel_transport("session/marker") == "file"


def test_engine_state_wildcard_resolves_to_ws() -> None:
    assert channels.get_channel_transport("engine/mujoco/state") == "ws"
    assert channels.get_channel_transport("engine/drake/state") == "ws"
    assert channels.get_channel_transport("engine/pinocchio/state") == "ws"


def test_unknown_channel_defaults_to_file() -> None:
    assert channels.get_channel_transport("custom/unregistered") == "file"
    assert channels.get_channel_transport("foo/bar/baz") == "file"


def test_register_and_lookup_low_frequency() -> None:
    channels.register_channel("test_low/example", "low")
    assert channels.get_channel_transport("test_low/example") == "file"


def test_register_and_lookup_high_frequency() -> None:
    channels.register_channel("test_high/example", "high")
    assert channels.get_channel_transport("test_high/example") == "ws"


def test_register_invalid_frequency_hint_raises() -> None:
    with pytest.raises(ValueError):
        channels.register_channel(
            "test_bad/example",
            "medium",  # type: ignore[arg-type]
        )


def test_register_literal_channel_validates_name() -> None:
    with pytest.raises(ValueError):
        channels.register_channel("BAD/Name", "high")


def test_engine_wildcard_does_not_match_extra_segments() -> None:
    # 'engine/mujoco/state' matches; 'engine/mujoco/state/sub' should not
    # be considered a wildcard match — and is unregistered → file.
    assert channels.get_channel_transport("engine/foo/state/extra") == "file"
