"""Coverage tests for ``starting_pose_matcher.live_view_controller``.

Edge cases: NaN markers, club-only, ball-only, body-only, layer toggling.
Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

# Headless GUI/drawing stack BEFORE matplotlib is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tools.starting_pose_matcher.live_view_controller import (
    LiveViewController,
    _default_body_segments_safe,
    _finite_rows,
    _maybe_xyz,
)


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def axes_canvas():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax, fig.canvas
    plt.close(fig)


def _body(n_frames=8, n_markers=4, names=None, marker_xyz=None):
    if marker_xyz is None:
        marker_xyz = np.linspace(0.0, 1.0, n_frames * n_markers * 3).reshape(
            n_frames, n_markers, 3
        )
    if names is None:
        names = tuple(f"m{i}" for i in range(marker_xyz.shape[1]))
    return SimpleNamespace(marker_xyz=marker_xyz, marker_names=names)


def _club(n=8):
    return SimpleNamespace(
        mid_hands=np.linspace(0.0, 1.0, n * 3).reshape(n, 3),
        clubhead=np.linspace(0.5, 1.5, n * 3).reshape(n, 3),
        clubface_triad=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
    )


def _ball(pos=(0.0, 0.0, 0.0)):
    return SimpleNamespace(position=np.array(pos, dtype=float))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def test_maybe_xyz_returns_first_present():
    obj = SimpleNamespace(a=None, b=np.array([[1.0, 2.0, 3.0]]))
    out = _maybe_xyz(obj, ("a", "b"))
    np.testing.assert_array_equal(out, [[1.0, 2.0, 3.0]])


def test_maybe_xyz_returns_none_when_all_missing():
    obj = SimpleNamespace()
    assert _maybe_xyz(obj, ("a", "b")) is None


def test_maybe_xyz_skips_empty_arrays():
    obj = SimpleNamespace(a=np.array([]), b=np.array([[0.0, 0.0, 0.0]]))
    out = _maybe_xyz(obj, ("a", "b"))
    np.testing.assert_array_equal(out, [[0.0, 0.0, 0.0]])


def test_finite_rows_filters_nans():
    arr = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.0, 5.0, 6.0]])
    out = _finite_rows(arr)
    assert out.shape == (2, 3)


def test_finite_rows_all_nan_returns_empty():
    arr = np.full((3, 3), np.nan)
    out = _finite_rows(arr)
    assert out.shape == (0, 3)


def test_finite_rows_wrong_shape_reshapes():
    arr = np.zeros(6)  # shape (6,)
    out = _finite_rows(arr)
    assert out.shape == (2, 3)


def test_default_body_segments_safe_returns_pairs():
    # Use names unlikely to be in the shared body segment table — falls
    # back to the consecutive-pairs heuristic.
    pairs = _default_body_segments_safe(("p0", "p1", "p2"))
    assert isinstance(pairs, list)
    assert pairs  # not empty
    # All entries are 2-tuples of ints.
    for pair in pairs:
        assert len(pair) == 2
        assert all(isinstance(p, int) for p in pair)


# --------------------------------------------------------------------------- #
# Controller behaviour                                                        #
# --------------------------------------------------------------------------- #


def test_controller_initial_state(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    assert ctrl.n_frames == 0
    assert ctrl.current_frame == 0
    assert ctrl.layers() == {}
    assert ctrl.has_layer("body_markers") is False


def test_controller_set_target_body_only(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    body = _body(n_frames=10, n_markers=5)
    ctrl.set_target(body=body)
    assert ctrl.n_frames == 10
    assert ctrl.has_layer("body_markers")
    assert ctrl.has_layer("body_skeleton")
    assert ctrl.has_layer("body_trail")
    assert not ctrl.has_layer("clubface_trace")
    assert not ctrl.has_layer("ball_impact")


def test_controller_set_target_club_only(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    club = _club(n=6)
    ctrl.set_target(club=club)
    assert ctrl.n_frames == 6
    assert ctrl.has_layer("club_midhands_trace")
    assert ctrl.has_layer("clubface_trace")
    assert ctrl.has_layer("clubface_triad")
    assert not ctrl.has_layer("body_markers")


def test_controller_set_target_ball_only(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ball = _ball((0.1, 0.2, 0.3))
    ctrl.set_target(ball=ball)
    assert ctrl.has_layer("ball_impact")
    assert ctrl.n_frames == 0  # ball alone doesn't drive frames


def test_controller_set_target_ball_with_bad_shape_skipped(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    bad = SimpleNamespace(position=np.array([1.0, 2.0]))  # not (3,)
    ctrl.set_target(ball=bad)
    assert not ctrl.has_layer("ball_impact")


def test_controller_set_target_ball_position_none_skipped(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_target(ball=SimpleNamespace(position=None))
    assert not ctrl.has_layer("ball_impact")


def test_controller_set_target_full_combo(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_target(body=_body(n_frames=6), club=_club(n=10), ball=_ball())
    # Maximum frame count from the slots.
    assert ctrl.n_frames == 10
    assert ctrl.has_layer("body_markers")
    assert ctrl.has_layer("clubface_triad")
    assert ctrl.has_layer("ball_impact")


def test_controller_body_shape_validation(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    bad = SimpleNamespace(marker_xyz=np.zeros((3, 3)), marker_names=("a", "b", "c"))
    with pytest.raises(ValueError, match="must have shape"):
        ctrl.set_target(body=bad)


def test_controller_body_with_nan_markers_no_fit_crash(axes_canvas):
    """Body with all-NaN markers must not crash auto-fit (covers the
    'finite.any() is False' branch)."""
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    marker_xyz = np.full((4, 3, 3), np.nan)
    body = SimpleNamespace(marker_xyz=marker_xyz, marker_names=("a", "b", "c"))
    ctrl.set_target(body=body)
    assert ctrl.n_frames == 4


def test_controller_set_frame_clamps(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_target(body=_body(n_frames=5))
    ctrl.set_frame(2)
    assert ctrl.current_frame == 2
    ctrl.set_frame(99)
    assert ctrl.current_frame == 4
    ctrl.set_frame(-1)
    assert ctrl.current_frame == 0


def test_controller_set_frame_with_no_target_is_noop(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_frame(5)
    assert ctrl.current_frame == 0


def test_controller_clear_drops_layers(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_target(body=_body(n_frames=4))
    assert ctrl.layers()
    ctrl.clear()
    assert ctrl.layers() == {}
    assert ctrl.n_frames == 0
    assert ctrl.current_frame == 0


def test_controller_set_layer_visible_body_markers_also_toggles_trail(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_target(body=_body(n_frames=4))
    ctrl.set_layer_visible("body_markers", False)
    # Both body_markers and body_trail should now be invisible (we only
    # assert no exception was raised; gui_playback layers handle the
    # visibility internally).
    assert ctrl.has_layer("body_markers")


def test_controller_set_layer_visible_unknown_key_is_noop(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_target(body=_body(n_frames=4))
    # Should not raise.
    ctrl.set_layer_visible("does_not_exist", False)


def test_controller_set_target_replaces_old_layers(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    ctrl.set_target(body=_body(n_frames=4))
    keys_v1 = set(ctrl.layers().keys())
    ctrl.set_target(club=_club(n=3))
    keys_v2 = set(ctrl.layers().keys())
    assert "body_markers" in keys_v1
    assert "body_markers" not in keys_v2
    assert "clubface_trace" in keys_v2


def test_controller_club_without_triad(axes_canvas):
    ax, canvas = axes_canvas
    ctrl = LiveViewController(ax=ax, canvas=canvas)
    club = SimpleNamespace(
        mid_hands=np.zeros((3, 3)),
        clubhead=np.ones((3, 3)),
    )
    ctrl.set_target(club=club)
    assert ctrl.has_layer("clubface_trace")
    assert not ctrl.has_layer("clubface_triad")
