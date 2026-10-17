"""Constant-marker-set preprocessing and branch-continuity checks (#13056)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.marker_continuity import (
    MarkerSetPolicy,
    constant_marker_set,
    largest_coordinate_steps,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit

_TRIO_A = ("a1", "a2", "a3")
_TRIO_B = ("b1", "b2", "b3")
_LOCAL = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.3],
        [0.15, 0.05, 0.0],
        [0.0, 0.1, 0.25],
    ]
)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rigid_capture(frames: int = 40) -> TourCapture:
    """A spinning, translating rigid six-marker cluster plus one smooth marker."""
    time = np.arange(frames) / 100.0
    points = np.zeros((frames, 7, 3))
    for f in range(frames):
        r = _rot_z(3.0 * time[f])
        points[f, :6] = _LOCAL @ r.T + np.array([time[f], 0.5 * time[f], 1.0])
        points[f, 6] = [np.sin(time[f]), np.cos(time[f]), time[f] ** 2]
    labels = (*_TRIO_A, *_TRIO_B, "waist")
    return TourCapture(time, labels, points, np.ones((frames, 7), dtype=bool))


def _with_gaps(cap: TourCapture, gaps: dict[str, range]) -> TourCapture:
    points, valid = cap.points_m.copy(), cap.valid.copy()
    for label, rows in gaps.items():
        j = cap.labels.index(label)
        valid[list(rows), j] = False
        points[list(rows), j] = np.nan
    return TourCapture(cap.time_s, cap.labels, points, valid)


def _policy(**overrides: object) -> MarkerSetPolicy:
    base = {"rigid_groups": (_TRIO_A, _TRIO_B), "max_spline_gap": 6}
    base.update(overrides)
    return MarkerSetPolicy(**base)  # type: ignore[arg-type]


def test_rebuilds_a_missing_trio_rigidly_from_the_other() -> None:
    truth = _rigid_capture()
    gapped = _with_gaps(truth, {m: range(10, 20) for m in _TRIO_A})
    out, report = constant_marker_set(gapped, _policy())
    assert out.valid.all()
    np.testing.assert_allclose(out.points_m, truth.points_m, atol=1e-9)
    assert report["rigid_filled"]["a1"] == list(range(10, 20))
    assert report["spline_filled"] == {}


def test_spline_bridges_a_short_gap_and_reports_it() -> None:
    truth = _rigid_capture()
    out, report = constant_marker_set(
        _with_gaps(truth, {"waist": range(15, 19)}), _policy()
    )
    j = truth.labels.index("waist")
    np.testing.assert_allclose(out.points_m[:, j], truth.points_m[:, j], atol=1e-5)
    assert report["spline_filled"] == {"waist": [15, 16, 17, 18]}


def test_drops_listed_markers_and_trims_an_incomplete_tail() -> None:
    truth = _rigid_capture()
    gapped = _with_gaps(
        truth, {"waist": range(0, 30), **{m: range(37, 40) for m in (*_TRIO_A, *_TRIO_B)}}
    )
    out, report = constant_marker_set(gapped, _policy(dropped=("waist",)))
    assert "waist" not in out.labels
    assert out.frames == 37
    assert report["trimmed_trailing_frames"] == 3
    assert report["dropped_markers"] == ["waist"]


def test_refuses_an_interior_gap_it_cannot_fill() -> None:
    gapped = _with_gaps(_rigid_capture(), {"waist": range(10, 20)})
    with pytest.raises(ValueError, match="interior marker gaps remain"):
        constant_marker_set(gapped, _policy())


def test_policy_rejects_bad_groups() -> None:
    with pytest.raises(ValueError, match="at least three"):
        MarkerSetPolicy(rigid_groups=(("a1", "a2"), _TRIO_B))
    with pytest.raises(ValueError, match="max_spline_gap"):
        MarkerSetPolicy(max_spline_gap=0)


def test_policy_labels_must_exist_in_the_capture() -> None:
    with pytest.raises(ValueError, match="not in the capture"):
        constant_marker_set(_rigid_capture(), _policy(dropped=("nope",)))


def test_largest_steps_ignore_translations_and_rank_by_size() -> None:
    q = np.zeros((5, 3))
    q[3, 0] = 0.5  # rotational jump into frame 3
    q[2, 1] = 0.3  # rotational jump into frame 2
    q[4, 2] = 9.0  # translation: ignored
    steps = largest_coordinate_steps(q, ["rx", "ry", "tx"], [True, True, False], count=2)
    assert steps[0] == {"frame": 3, "coordinate": "rx", "step_rad": pytest.approx(0.5)}
    assert steps[1]["coordinate"] == "ry"
    assert len(steps) == 2


def test_largest_steps_validates_shapes() -> None:
    with pytest.raises(ValueError, match="frames, coordinates"):
        largest_coordinate_steps(np.zeros((4, 2)), ["a"], [True])
