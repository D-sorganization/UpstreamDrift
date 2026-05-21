"""Comprehensive tests for ``starting_pose_matcher.session_schema``."""

from __future__ import annotations

import pytest

from src.tools.starting_pose_matcher import session_schema as ss


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    assert ss.SESSION_SCHEMA_VERSION == 6
    assert 1.0 in ss.ALLOWED_SPEEDS
    assert ss.DEFAULT_TRAIL_FRAMES > 0
    assert ss.DEFAULT_BODY_MARKER_SET in ss.DEFAULT_BODY_MARKER_SETS
    assert ss.DEFAULT_TIME_ALIGNMENT in ("impact", "address")
    assert ss.DEFAULT_BODY_SKELETON_STYLE in ss.BODY_SKELETON_STYLES


# ---------------------------------------------------------------------------
# PlaybackState
# ---------------------------------------------------------------------------


def test_playback_state_defaults():
    p = ss.PlaybackState()
    assert p.current_frame == 0
    assert p.speed == 1.0
    assert p.loop is True
    assert p.trail_frames == ss.DEFAULT_TRAIL_FRAMES


def test_playback_state_validates_current_frame():
    with pytest.raises(ValueError, match="current_frame"):
        ss.PlaybackState(current_frame=-1)


def test_playback_state_validates_speed():
    with pytest.raises(ValueError, match="speed"):
        ss.PlaybackState(speed=0.0)
    with pytest.raises(ValueError, match="speed"):
        ss.PlaybackState(speed=-1.0)


def test_playback_state_validates_trail_frames():
    with pytest.raises(ValueError, match="trail_frames"):
        ss.PlaybackState(trail_frames=-1)


def test_playback_state_from_dict_empty():
    assert ss.PlaybackState.from_dict(None) == ss.PlaybackState()
    assert ss.PlaybackState.from_dict({}) == ss.PlaybackState()


def test_playback_state_from_dict_partial():
    p = ss.PlaybackState.from_dict({"speed": 2.0, "loop": False})
    assert p.speed == 2.0
    assert p.loop is False
    assert p.current_frame == 0  # default


def test_playback_state_from_dict_ignores_unknown_keys():
    p = ss.PlaybackState.from_dict({"speed": 2.0, "junk": 99})
    assert p.speed == 2.0


def test_playback_state_from_dict_coerces_types():
    p = ss.PlaybackState.from_dict(
        {"current_frame": "5", "speed": "0.5", "loop": 0, "trail_frames": "10"}
    )
    assert p.current_frame == 5
    assert p.speed == 0.5
    assert p.loop is False
    assert p.trail_frames == 10


def test_playback_state_to_dict_round_trip():
    p = ss.PlaybackState(current_frame=3, speed=2.0, loop=False, trail_frames=15)
    d = p.to_dict()
    assert d == {
        "current_frame": 3,
        "speed": 2.0,
        "loop": False,
        "trail_frames": 15,
    }
    assert ss.PlaybackState.from_dict(d) == p


def test_playback_state_snap_speed_to_allowed():
    p = ss.PlaybackState(speed=0.12)
    assert p.snap_speed_to_allowed() == 0.1
    p = ss.PlaybackState(speed=0.9)
    assert p.snap_speed_to_allowed() == 1.0


# ---------------------------------------------------------------------------
# DataSources block
# ---------------------------------------------------------------------------


def test_default_data_sources_is_all_disabled():
    d = ss.default_data_sources()
    assert d.club.enabled is False
    assert d.body.enabled is False
    assert d.align.sample_rate_hz == 1000.0


def test_serialize_parse_data_sources_round_trip():
    block = ss.DataSourcesBlock(
        club=ss.ClubSourceBlock(enabled=True, file_path="x.xlsx", include_ball=True),
        body=ss.BodySourceBlock(
            enabled=True, file_path="y.c3d", marker_set="Upper body only"
        ),
        align=ss.AlignOptionsBlock(
            sample_rate_hz=2000.0,
            simulation_time_s=0.5,
            time_alignment="address",
        ),
    )
    serialised = ss.serialize_data_sources(block)
    parsed = ss.parse_data_sources(serialised)
    assert parsed == block


def test_parse_data_sources_none_returns_default():
    assert ss.parse_data_sources(None) == ss.default_data_sources()
    assert ss.parse_data_sources({}) == ss.default_data_sources()


def test_parse_data_sources_ignores_bad_time_alignment():
    parsed = ss.parse_data_sources({"align": {"time_alignment": "bogus"}})
    assert parsed.align.time_alignment == ss.DEFAULT_TIME_ALIGNMENT


def test_parse_data_sources_coerces_none_file_path():
    parsed = ss.parse_data_sources({"club": {"enabled": True, "file_path": None}})
    assert parsed.club.file_path is None


def test_parse_data_sources_str_path():
    parsed = ss.parse_data_sources({"club": {"file_path": 42}})
    assert parsed.club.file_path == "42"


def test_parse_data_sources_partial_block_uses_defaults():
    parsed = ss.parse_data_sources({"club": {"enabled": True}})
    assert parsed.club.enabled is True
    assert parsed.club.file_path is None
    assert parsed.body.enabled is False
    assert parsed.align.sample_rate_hz == 1000.0


# ---------------------------------------------------------------------------
# BodySkeleton block
# ---------------------------------------------------------------------------


def test_default_body_skeleton_uses_default_style():
    assert ss.default_body_skeleton().style == ss.DEFAULT_BODY_SKELETON_STYLE


def test_serialize_parse_body_skeleton_round_trip():
    block = ss.BodySkeletonBlock(style="library_shapes")
    serialised = ss.serialize_body_skeleton(block)
    assert ss.parse_body_skeleton(serialised) == block


def test_parse_body_skeleton_unknown_falls_back_to_default():
    parsed = ss.parse_body_skeleton({"style": "neon"})
    assert parsed.style == ss.DEFAULT_BODY_SKELETON_STYLE


def test_parse_body_skeleton_none_returns_default():
    assert ss.parse_body_skeleton(None) == ss.default_body_skeleton()


# ---------------------------------------------------------------------------
# PlotStyles block
# ---------------------------------------------------------------------------


def test_default_plot_styles_is_all_none():
    p = ss.default_plot_styles()
    assert p.body is None and p.club is None


def test_serialize_plot_styles_writes_null_for_missing():
    serialised = ss.serialize_plot_styles(ss.PlotStylesBlock())
    assert serialised == {"body": None, "club": None}


def test_parse_plot_styles_dict_round_trip():
    block = ss.PlotStylesBlock(body={"color": "red"}, club={"color": "blue"})
    parsed = ss.parse_plot_styles(ss.serialize_plot_styles(block))
    assert parsed.body == {"color": "red"}
    assert parsed.club == {"color": "blue"}


def test_parse_plot_styles_ignores_non_mapping():
    parsed = ss.parse_plot_styles({"body": "not-a-dict", "club": 42})
    assert parsed.body is None
    assert parsed.club is None


def test_parse_plot_styles_empty_returns_default():
    assert ss.parse_plot_styles(None) == ss.default_plot_styles()
    assert ss.parse_plot_styles({}) == ss.default_plot_styles()
