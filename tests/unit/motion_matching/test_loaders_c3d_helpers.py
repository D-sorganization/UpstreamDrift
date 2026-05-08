"""Coverage tests for private helpers in ``loaders.c3d``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.shared.python.motion_matching.loaders.c3d import (
    _fill_rotation_nans,
    _first_clean_frame,
    _marker_xyz,
    _pick_marker,
    _shaft_quaternions,
)


# --- _pick_marker -----------------------------------------------------------


def test_pick_marker_substring_uppercased() -> None:
    """Pin: substring match is case-insensitive."""
    labels = ["clubBUTT_x", "head_y", "shaft"]
    assert _pick_marker(labels, ("BUTT",)) == "clubBUTT_x"


def test_pick_marker_iteration_order() -> None:
    """Pin: candidates are searched in given order."""
    labels = ["foo_HEAD", "bar_BUTT"]
    assert _pick_marker(labels, ("BUTT", "HEAD")) == "bar_BUTT"


def test_pick_marker_returns_none() -> None:
    """Pin: no match -> None."""
    assert _pick_marker(["x", "y"], ("zzz",)) is None


# --- _marker_xyz ------------------------------------------------------------


def test_marker_xyz_returns_sorted_arrays() -> None:
    """Pin: rows for ``marker`` are sorted by ``frame`` and split into time/xyz."""
    df = pd.DataFrame(
        {
            "marker": ["A", "A", "B"],
            "frame": [1, 0, 0],
            "time": [0.1, 0.0, 0.0],
            "x": [1.0, 0.0, 9.0],
            "y": [0.0, 0.0, 9.0],
            "z": [0.0, 0.0, 9.0],
        }
    )
    time, xyz = _marker_xyz(df, "A")
    assert np.allclose(time, [0.0, 0.1])
    assert np.allclose(xyz[:, 0], [0.0, 1.0])


# --- _fill_rotation_nans ----------------------------------------------------


def test_fill_rotation_nans_preserves_finite_rows() -> None:
    """Pin: finite rows pass through; NaN rows are filled from last good."""
    rot = np.zeros((3, 3, 3))
    rot[0] = np.eye(3)
    rot[1] = np.full((3, 3), np.nan)
    rot[2] = np.eye(3) * 2.0
    out = _fill_rotation_nans(rot)
    assert np.allclose(out[0], np.eye(3))
    assert np.allclose(out[1], np.eye(3))  # filled from row 0
    assert np.allclose(out[2], np.eye(3) * 2.0)


def test_fill_rotation_nans_leading_nan_falls_back_to_identity() -> None:
    """Pin: NaN rows before any good row fall back to identity."""
    rot = np.zeros((2, 3, 3))
    rot[0] = np.full((3, 3), np.nan)
    rot[1] = np.eye(3)
    out = _fill_rotation_nans(rot)
    assert np.allclose(out[0], np.eye(3))


# --- _shaft_quaternions -----------------------------------------------------


def test_shaft_quaternions_z_aligned() -> None:
    """Pin: shaft along +z gives identity quaternion."""
    butt = np.zeros((1, 3))
    head = np.array([[0.0, 0.0, 1.0]])
    q = _shaft_quaternions(butt, head)
    assert q.shape == (1, 4)
    assert np.allclose(q[0], [1.0, 0.0, 0.0, 0.0])


def test_shaft_quaternions_z_antiparallel() -> None:
    """Pin: shaft along -z gives the documented (0,1,0,0) sentinel."""
    butt = np.zeros((1, 3))
    head = np.array([[0.0, 0.0, -1.0]])
    q = _shaft_quaternions(butt, head)
    assert np.allclose(q[0], [0.0, 1.0, 0.0, 0.0])


def test_shaft_quaternions_zero_vector() -> None:
    """Pin: zero-length shaft falls back to identity quaternion."""
    butt = np.array([[0.0, 0.0, 0.0]])
    head = np.array([[0.0, 0.0, 0.0]])
    q = _shaft_quaternions(butt, head)
    assert np.allclose(q[0], [1.0, 0.0, 0.0, 0.0])


def test_shaft_quaternions_general() -> None:
    """Pin: a +x shaft gives a quaternion that rotates +z onto +x."""
    butt = np.zeros((1, 3))
    head = np.array([[1.0, 0.0, 0.0]])
    q = _shaft_quaternions(butt, head)
    # Apply rotation to (0,0,1), expect (1,0,0).
    w, x, y, z = q[0]
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = rot @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(out, [1.0, 0.0, 0.0], atol=1e-9)


# --- _first_clean_frame -----------------------------------------------------


def test_first_clean_frame_finds_match() -> None:
    """Pin: the first all-finite cluster frame index is returned."""
    from src.shared.python.motion_matching.loaders._marker_clusters import (
        CLUBHEAD_CLUSTER,
        GRIP_CLUSTER,
    )

    n = 5
    points = {}
    for m in (*CLUBHEAD_CLUSTER, *GRIP_CLUSTER):
        arr = np.zeros((n, 3))
        arr[0] = np.nan  # frame 0 is bad
        points[m] = arr
    assert _first_clean_frame(points) == 1


def test_first_clean_frame_raises_when_no_match() -> None:
    """Pin: with no fully-finite frame, ValueError is raised."""
    from src.shared.python.motion_matching.loaders._marker_clusters import (
        CLUBHEAD_CLUSTER,
        GRIP_CLUSTER,
    )

    n = 3
    points = {m: np.full((n, 3), np.nan) for m in (*CLUBHEAD_CLUSTER, *GRIP_CLUSTER)}
    with pytest.raises(ValueError, match="No frame"):
        _first_clean_frame(points)
