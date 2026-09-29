"""Headless tests for the animated full-trajectory preview (issue #4482).

These tests exercise :class:`PlaybackController` and the per-layer artists
without a real Qt event loop. They use the matplotlib ``Agg`` backend and
set ``QT_QPA_PLATFORM=offscreen`` so they pass on CI machines without a
display.

Acceptance criteria covered:

* Animation runs through 50 frames without exception.
* Toggling layer visibility mid-playback does not tear (artist count
  consistent before/after).
* Session JSON round-trips the playback block.
"""

from __future__ import annotations

import json
import os

# Force a headless GUI / drawing stack BEFORE matplotlib is imported. Both
# QT_QPA_PLATFORM and the matplotlib backend matter on CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.tools.starting_pose_matcher.gui_playback import (
    BallImpactLayer,
    BodyMarkerLayer,
    BodySkeletonLayer,
    ClubTraceLayer,
    ClubfaceTriadLayer,
    PlaybackController,
    TrailLayer,
    precompute_segments_from_pairs,
)
from src.tools.starting_pose_matcher.session_schema import (
    ALLOWED_SPEEDS,
    DEFAULT_TRAIL_FRAMES,
    PlaybackState,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_target():
    """A synthetic 50-frame target with body markers, club trace, ball impact."""
    rng = np.random.default_rng(0)
    T, M = 50, 8
    t = np.linspace(0.0, 1.0, T)

    # Body markers swirl around the origin in metres.
    base = rng.uniform(-0.5, 0.5, size=(M, 3))
    angle = 2 * np.pi * t
    marker_xyz = np.stack(
        [base + 0.05 * np.array([np.cos(a), np.sin(a), 0.0]) for a in angle], axis=0
    )

    butt = np.stack([t, np.sin(angle), np.cos(angle)], axis=-1) * 0.3
    clubhead = butt + np.stack(
        [np.zeros_like(t), np.zeros_like(t), 0.5 * np.ones_like(t)], axis=-1
    )

    # Identity triad for each frame.
    triad = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    segments = precompute_segments_from_pairs(marker_xyz, pairs)

    return {
        "T": T,
        "marker_xyz": marker_xyz,
        "butt": butt,
        "clubhead": clubhead,
        "triad": triad,
        "ball_impact": np.array([0.0, 0.0, 0.0]),
        "segments": segments,
    }


def _build_controller(target):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    layers = [
        BodyMarkerLayer(key="body_markers", marker_xyz=target["marker_xyz"]),
        BodySkeletonLayer(key="body_skeleton", segments=target["segments"]),
        ClubTraceLayer(
            key="club_midhands_trace",
            positions=target["butt"],
            color="tab:orange",
        ),
        ClubTraceLayer(
            key="clubface_trace",
            positions=target["clubhead"],
            color="tab:purple",
        ),
        ClubfaceTriadLayer(
            key="clubface_triad",
            origin=target["clubhead"],
            triad=target["triad"],
        ),
        BallImpactLayer(key="ball_impact", position=target["ball_impact"]),
        TrailLayer(
            key="body_markers_trail",
            marker_xyz=target["marker_xyz"],
            trail_frames=DEFAULT_TRAIL_FRAMES,
        ),
    ]
    return fig, ax, PlaybackController(ax=ax, n_frames=target["T"], layers=layers)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_full_50_frame_playthrough_without_exception(synthetic_target):
    """Animation runs through 50 frames synchronously without raising."""
    _, _, ctrl = _build_controller(synthetic_target)
    # Step forward the full timeline. With loop=True this would wrap; with
    # exact T=50 stepping 49 times brings us to the last frame.
    ctrl.step(49)
    assert ctrl.state.current_frame == synthetic_target["T"] - 1
    # Step once more to wrap (loop default True).
    ctrl.step(1)
    assert ctrl.state.current_frame == 0


def test_layer_visibility_toggle_does_not_tear(synthetic_target):
    """Toggling layer visibility mid-playback keeps artist count constant."""
    _, _, ctrl = _build_controller(synthetic_target)
    before = len(ctrl.all_artists())
    ctrl.step(10)
    ctrl.set_layer_visible("body_markers", False)
    ctrl.set_layer_visible("clubface_trace", False)
    ctrl.step(10)
    ctrl.set_layer_visible("body_markers", True)
    ctrl.step(5)
    after = len(ctrl.all_artists())
    assert before == after, (before, after)
    # Toggled layers report the right visibility state.
    assert ctrl.layer("body_markers").visible is True
    assert ctrl.layer("clubface_trace").visible is False


def test_session_json_round_trips_playback_block(tmp_path):
    """Playback dataclass round-trips through the on-disk session JSON."""
    state = PlaybackState(current_frame=42, speed=2.0, loop=False, trail_frames=15)
    payload = {"playback": state.to_dict(), "schema_version": 3}
    path = tmp_path / "session.json"
    path.write_text(json.dumps(payload))

    loaded = json.loads(path.read_text())
    rebuilt = PlaybackState.from_dict(loaded["playback"])
    assert rebuilt == state

    # Backwards compat: a session with a missing or partial playback block
    # falls back to defaults.
    assert PlaybackState.from_dict(None) == PlaybackState()
    partial = PlaybackState.from_dict({"current_frame": 7})
    assert partial.current_frame == 7
    assert partial.speed == 1.0
    assert partial.loop is True
    assert partial.trail_frames == DEFAULT_TRAIL_FRAMES


def test_allowed_speeds_match_ui_combo():
    """The advertised speed combo values stay in sync with the schema."""
    assert ALLOWED_SPEEDS == (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def test_set_speed_validates_and_updates_state(synthetic_target):
    _, _, ctrl = _build_controller(synthetic_target)
    ctrl.set_speed(0.5)
    assert ctrl.state.speed == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ctrl.set_speed(0.0)


def test_seek_clamps_to_valid_range(synthetic_target):
    _, _, ctrl = _build_controller(synthetic_target)
    ctrl.seek(10**6)
    assert ctrl.state.current_frame == synthetic_target["T"] - 1
    ctrl.seek(-50)
    assert ctrl.state.current_frame == 0


def test_trail_layer_responds_to_trail_frames_setting(synthetic_target):
    _, _, ctrl = _build_controller(synthetic_target)
    ctrl.set_trail_frames(5)
    assert ctrl.state.trail_frames == 5
    # The TrailLayer's internal field updates.
    trail = ctrl.layer("body_markers_trail")
    assert trail.trail_frames == 5
    with pytest.raises(ValueError):
        ctrl.set_trail_frames(-1)


def test_loop_off_stops_at_last_frame(synthetic_target):
    _, _, ctrl = _build_controller(synthetic_target)
    ctrl.set_loop(False)
    ctrl.seek(synthetic_target["T"] - 2)
    ctrl.step(5)
    assert ctrl.state.current_frame == synthetic_target["T"] - 1


def test_precompute_segments_shape(synthetic_target):
    pairs = [(0, 1), (2, 3)]
    segs = precompute_segments_from_pairs(synthetic_target["marker_xyz"], pairs)
    T = synthetic_target["T"]
    assert segs.shape == (T, len(pairs), 2, 3)


def test_pause_resume_safe_without_animation(synthetic_target):
    """pause()/resume() are no-ops when no FuncAnimation has been started."""
    _, _, ctrl = _build_controller(synthetic_target)
    # Should not raise even though .start() was never called.
    ctrl.pause()
    ctrl.resume()
    # Without an underlying FuncAnimation, the running flag stays False.
    assert ctrl.is_running is False
