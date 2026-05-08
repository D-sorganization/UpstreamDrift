"""Coverage tests for ``starting_pose_matcher.session_schema``.

Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

import pytest

from src.tools.starting_pose_matcher.session_schema import (
    ALLOWED_SPEEDS,
    DEFAULT_BODY_MARKER_SET,
    DEFAULT_BODY_MARKER_SETS,
    DEFAULT_TIME_ALIGNMENT,
    DEFAULT_TRAIL_FRAMES,
    SESSION_SCHEMA_VERSION,
    AlignOptionsBlock,
    BodySourceBlock,
    ClubSourceBlock,
    DataSourcesBlock,
    PlaybackState,
    default_data_sources,
    parse_data_sources,
    serialize_data_sources,
)


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# PlaybackState                                                              #
# --------------------------------------------------------------------------- #


def test_playback_state_defaults():
    s = PlaybackState()
    assert s.current_frame == 0
    assert s.speed == 1.0
    assert s.loop is True
    assert s.trail_frames == DEFAULT_TRAIL_FRAMES


@pytest.mark.parametrize(
    "kwargs,err",
    [
        ({"current_frame": -1}, "current_frame must be >= 0"),
        ({"speed": 0}, "speed must be > 0"),
        ({"speed": -2.0}, "speed must be > 0"),
        ({"trail_frames": -5}, "trail_frames must be >= 0"),
    ],
)
def test_playback_state_validation_raises(kwargs, err):
    with pytest.raises(ValueError, match=err):
        PlaybackState(**kwargs)


def test_playback_state_from_dict_none_returns_defaults():
    s = PlaybackState.from_dict(None)
    assert s == PlaybackState()
    s = PlaybackState.from_dict({})
    assert s == PlaybackState()


def test_playback_state_from_dict_full():
    s = PlaybackState.from_dict(
        {"current_frame": 7, "speed": 2.0, "loop": False, "trail_frames": 12}
    )
    assert s.current_frame == 7
    assert s.speed == 2.0
    assert s.loop is False
    assert s.trail_frames == 12


def test_playback_state_from_dict_partial_uses_defaults():
    s = PlaybackState.from_dict({"speed": 0.5})
    assert s.speed == 0.5
    assert s.current_frame == 0
    assert s.loop is True


def test_playback_state_from_dict_ignores_unknown_keys():
    s = PlaybackState.from_dict({"speed": 0.25, "future_field": "ignored"})
    assert s.speed == 0.25


def test_playback_state_to_dict_round_trip():
    s = PlaybackState(current_frame=3, speed=2.0, loop=False, trail_frames=10)
    d = s.to_dict()
    assert d == {
        "current_frame": 3,
        "speed": 2.0,
        "loop": False,
        "trail_frames": 10,
    }
    assert PlaybackState.from_dict(d) == s


def test_playback_state_snap_speed_to_allowed_picks_closest():
    s = PlaybackState(speed=0.31)
    assert s.snap_speed_to_allowed() == 0.25
    s = PlaybackState(speed=3.0)
    # closest of 2.0 vs 4.0 — tie-broken to 2.0 (first in iteration with smallest diff)
    snapped = s.snap_speed_to_allowed()
    assert snapped in (2.0, 4.0)
    s = PlaybackState(speed=1.0)
    assert s.snap_speed_to_allowed() == 1.0


def test_allowed_speeds_constants():
    # API guarantee: tuple of positive multipliers including 1.0
    assert 1.0 in ALLOWED_SPEEDS
    assert all(v > 0 for v in ALLOWED_SPEEDS)


# --------------------------------------------------------------------------- #
# DataSourcesBlock                                                           #
# --------------------------------------------------------------------------- #


def test_session_schema_version_at_least_4():
    assert SESSION_SCHEMA_VERSION >= 4


def test_default_body_marker_sets_includes_default():
    assert DEFAULT_BODY_MARKER_SET in DEFAULT_BODY_MARKER_SETS


def test_default_data_sources_returns_empty_block():
    d = default_data_sources()
    assert d == DataSourcesBlock()
    assert d.club.enabled is False
    assert d.body.enabled is False
    assert d.align.time_alignment == DEFAULT_TIME_ALIGNMENT


def test_serialize_data_sources_returns_dict():
    block = DataSourcesBlock(
        club=ClubSourceBlock(enabled=True, file_path="/tmp/x.xlsx", include_ball=True),
        body=BodySourceBlock(
            enabled=True, file_path="/tmp/y.c3d", marker_set="All markers"
        ),
        align=AlignOptionsBlock(
            sample_rate_hz=500.0, simulation_time_s=0.4, time_alignment="address"
        ),
    )
    blob = serialize_data_sources(block)
    assert isinstance(blob, dict)
    assert blob["club"]["enabled"] is True
    assert blob["club"]["include_ball"] is True
    assert blob["body"]["marker_set"] == "All markers"
    assert blob["align"]["time_alignment"] == "address"


def test_parse_data_sources_round_trip():
    src = DataSourcesBlock(
        club=ClubSourceBlock(enabled=True, file_path="/p", include_ball=False),
        body=BodySourceBlock(
            enabled=False, file_path=None, marker_set="Upper body only"
        ),
        align=AlignOptionsBlock(sample_rate_hz=2000.0, simulation_time_s=0.5),
    )
    assert parse_data_sources(serialize_data_sources(src)) == src


def test_parse_data_sources_none_and_empty():
    assert parse_data_sources(None) == default_data_sources()
    assert parse_data_sources({}) == default_data_sources()


def test_parse_data_sources_invalid_time_alignment_falls_back():
    block = parse_data_sources({"align": {"time_alignment": "garbage"}})
    assert block.align.time_alignment == DEFAULT_TIME_ALIGNMENT


def test_parse_data_sources_none_time_alignment_falls_back():
    block = parse_data_sources({"align": {"time_alignment": None}})
    assert block.align.time_alignment == DEFAULT_TIME_ALIGNMENT


def test_parse_data_sources_partial_blocks_use_defaults():
    block = parse_data_sources({"club": {"enabled": True}})
    assert block.club.enabled is True
    assert block.club.file_path is None
    assert block.body == BodySourceBlock()
    assert block.align == AlignOptionsBlock()


def test_parse_data_sources_string_coercion_for_file_path():
    block = parse_data_sources({"body": {"file_path": 12345}})
    # _coerce_str_or_none stringifies non-None
    assert block.body.file_path == "12345"


def test_parse_data_sources_none_subblocks_treated_as_empty():
    block = parse_data_sources({"club": None, "body": None, "align": None})
    assert block == default_data_sources()


def test_parse_data_sources_unknown_keys_are_ignored():
    block = parse_data_sources(
        {
            "club": {"enabled": True, "future_field": "ignored"},
            "extra_top_level": 42,
        }
    )
    assert block.club.enabled is True


def test_dataclass_blocks_are_frozen_and_hashable():
    a = ClubSourceBlock(enabled=True)
    b = ClubSourceBlock(enabled=True)
    assert a == b
    # frozen=True implies hashable by default
    assert hash(a) == hash(b)
