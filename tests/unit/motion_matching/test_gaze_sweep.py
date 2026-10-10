"""Knee-of-the-Pareto-front selection of the default gaze weight (OSV-3b)."""

from __future__ import annotations

import pytest

from src.shared.python.motion_matching.gaze_sweep import (
    SweepPoint,
    feasible,
    pareto_front,
    point_from_row,
    select_default,
    select_knee,
)

pytestmark = pytest.mark.unit


def _p(w: float, marker: float, gaze: float, face: float = 1.0) -> SweepPoint:
    return SweepPoint(w, marker, gaze, face)


BASE = _p(0, 30.0, 20.0)


def test_feasible_enforces_marker_and_face_limits() -> None:
    pts = [BASE, _p(1, 33.0, 2.0), _p(2, 33.1, 1.0), _p(3, 30.0, 1.5, face=6.0)]
    got = feasible(pts, marker_tolerance=0.10, face_cap_deg=5.0)
    assert [p.weight for p in got] == [0, 1]  # 33.1 > 1.1 * 30 ; face 6 > 5


def test_pareto_front_drops_dominated_points() -> None:
    pts = [BASE, _p(1, 31.0, 5.0), _p(2, 32.0, 6.0), _p(3, 33.0, 1.0)]
    assert [p.weight for p in pareto_front(pts)] == [0, 1, 3]


def test_knee_is_the_corner_of_an_l_shaped_front() -> None:
    pts = [BASE, _p(1, 30.5, 2.0), _p(3, 31.0, 1.5), _p(10, 32.5, 1.4)]
    assert select_knee(pts).weight == 1


def test_knee_requires_the_weight_zero_baseline() -> None:
    with pytest.raises(ValueError, match="weight 0"):
        select_knee([_p(1, 31.0, 2.0)])


def test_knee_falls_back_to_baseline_when_nothing_else_is_feasible() -> None:
    pts = [BASE, _p(1, 40.0, 2.0)]
    assert select_knee(pts).weight == 0


def test_default_is_the_smallest_knee_feasible_in_every_capture() -> None:
    driver = [BASE, _p(0.5, 30.5, 2.0), _p(1, 31.0, 1.5), _p(3, 32.5, 1.4)]
    # Weight 1 breaks the iron marker tolerance, so only 0 and 0.5 are common.
    iron = [BASE, _p(0.5, 31.0, 3.0), _p(1, 34.0, 1.0), _p(3, 30.2, 1.6)]
    assert select_default({"driver": driver, "iron": iron}) == 0.5


def test_default_falls_back_to_zero_when_no_common_weight_qualifies() -> None:
    driver = [BASE, _p(1, 31.0, 2.0)]
    iron = [BASE, _p(2, 31.0, 2.0)]
    assert select_default({"driver": driver, "iron": iron}) == 0


def test_default_needs_a_feasible_baseline_in_every_capture() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_default({})
    with pytest.raises(ValueError, match="baseline"):
        select_default({"driver": [_p(0, 30.0, 20.0, face=9.0)]})


def test_point_from_row_reads_the_sweep_row_fields() -> None:
    row = {
        "gaze_weight": 0.5,
        "marker_rms_mm": 33.0,
        "face_fit_deg": {"rms_deg": 0.8},
        "head_gaze": {"address_to_impact": {"theta_gaze_rms_deg": 4.0}},
    }
    assert point_from_row(row) == _p(0.5, 33.0, 4.0, face=0.8)
    with pytest.raises(ValueError, match="missing"):
        point_from_row({"gaze_weight": 1.0})


def test_rejects_nonfinite_and_negative_values() -> None:
    with pytest.raises(ValueError):
        _p(-1, 1.0, 1.0)
    with pytest.raises(ValueError):
        _p(1, float("nan"), 1.0)
