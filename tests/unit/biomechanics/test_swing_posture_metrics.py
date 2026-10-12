"""Tests for neutral swing posture metrics (Issue #11026, #11007 package 3)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.analysis.swing_events import SwingEventFrames
from src.shared.python.biomechanics import swing_posture_metrics as spm
from src.shared.python.biomechanics.swing_posture_metrics import (
    METRIC_NAMES,
    SwingPostureMetrics,
    compute_swing_posture_metrics,
    evaluate_thresholds,
)

TARGET = np.array([1.0, 0.0, 0.0])
BALL = np.array([0.0, 1.0, 0.0])
pytestmark = pytest.mark.unit

EVENTS = SwingEventFrames(address=1, top=3, peak=5, finish=7)
N = 8


def _series(
    toward: tuple[float, float] = (0.10, 0.20),
    lat_back: float = 0.30,
    lat_down: float = -0.05,
    tilt_a: float = 20.0,
    tilt_i: float = 27.0,
):
    """Piecewise-known trajectory. toward=(address, impact) along BALL."""
    pelvis = np.zeros((N, 3))
    pelvis[:, 2] = 1.0
    pelvis[1] = [0.0, toward[0], 1.0]
    pelvis[3] = [lat_back, toward[0], 1.0]
    pelvis[5] = [lat_back + lat_down, toward[1], 1.0]
    trunk = np.tile([0.0, 0.0, 1.0], (N, 1))
    for idx, deg in ((1, tilt_a), (5, tilt_i)):
        r = math.radians(deg)
        trunk[idx] = [0.0, math.sin(r), math.cos(r)]
    return pelvis, trunk


def _compute(pelvis, trunk, events=EVENTS, target=TARGET, ball=BALL):
    return compute_swing_posture_metrics(pelvis, trunk, events, target, ball)


def test_recovers_known_metrics_to_1e_9() -> None:
    pelvis, trunk = _series()
    m = _compute(pelvis, trunk)
    assert m.pelvis_toward_ball_shift_m == pytest.approx(0.10, abs=1e-9)
    assert m.pelvis_lateral_shift_backswing_m == pytest.approx(0.30, abs=1e-9)
    assert m.pelvis_lateral_shift_downswing_m == pytest.approx(-0.05, abs=1e-9)
    assert m.trunk_forward_tilt_change_deg == pytest.approx(7.0, abs=1e-9)


def test_result_is_frozen() -> None:
    pelvis, trunk = _series()
    m = _compute(pelvis, trunk)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.pelvis_toward_ball_shift_m = 1.0  # type: ignore[misc]


def test_invariant_to_translation() -> None:
    pelvis, trunk = _series()
    base = _compute(pelvis, trunk)
    moved = _compute(pelvis + np.array([5.0, -3.0, 0.7]), trunk)
    for name in METRIC_NAMES:
        assert getattr(moved, name) == pytest.approx(getattr(base, name), abs=1e-9)


@pytest.mark.parametrize("angle_deg", [17.0, 90.0, 203.0])
def test_invariant_to_rotation_about_vertical(angle_deg: float) -> None:
    pelvis, trunk = _series()
    base = _compute(pelvis, trunk)
    a = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )
    moved = _compute(
        pelvis @ rot.T, trunk @ rot.T, target=rot @ TARGET, ball=rot @ BALL
    )
    for name in METRIC_NAMES:
        assert getattr(moved, name) == pytest.approx(getattr(base, name), abs=1e-9)


def test_non_orthonormal_axes_refused() -> None:
    pelvis, trunk = _series()
    with pytest.raises(ValueError):
        _compute(
            pelvis, trunk, ball=np.array([0.1, 1.0, 0.0]) / np.linalg.norm([0.1, 1.0])
        )
    with pytest.raises(ValueError):
        _compute(pelvis, trunk, target=np.array([2.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        _compute(pelvis, trunk, target=np.array([1.0, 0.0]))


def test_axes_within_tolerance_accepted() -> None:
    pelvis, trunk = _series()
    tilt = np.array([1.0, 1e-9, 0.0])
    _compute(pelvis, trunk, target=tilt / np.linalg.norm(tilt))


def test_event_frames_out_of_range_refused() -> None:
    pelvis, trunk = _series()
    with pytest.raises(ValueError):
        _compute(pelvis, trunk, events=SwingEventFrames(1, 3, 5, 8))
    with pytest.raises(ValueError):
        _compute(pelvis, trunk[:-1])  # length mismatch


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_refused(bad: float) -> None:
    pelvis, trunk = _series()
    pelvis[2, 0] = bad
    with pytest.raises(ValueError):
        _compute(pelvis, trunk)
    pelvis, trunk = _series()
    trunk[5, 1] = bad
    with pytest.raises(ValueError):
        _compute(pelvis, trunk)


def test_bad_shapes_refused() -> None:
    pelvis, trunk = _series()
    with pytest.raises(ValueError):
        _compute(pelvis[:, :2], trunk)
    with pytest.raises(ValueError):
        _compute(pelvis, np.zeros((N, 3)))  # zero-length trunk axis


def _metrics() -> SwingPostureMetrics:
    return SwingPostureMetrics(0.10, 0.30, -0.05, 7.0)


def test_thresholds_exceeds_boundary_is_strict() -> None:
    m = _metrics()
    out = evaluate_thresholds(
        m,
        {
            "pelvis_toward_ball_shift_m": 0.10,  # equal -> not exceeded
            "pelvis_lateral_shift_backswing_m": 0.2999999,  # just below -> exceeded
            "pelvis_lateral_shift_downswing_m": 0.05,  # |-0.05| equal -> not
            "trunk_forward_tilt_change_deg": 6.9,
        },
        provenance="unit-test",
    )
    assert out == {
        "pelvis_toward_ball_shift_m": False,
        "pelvis_lateral_shift_backswing_m": True,
        "pelvis_lateral_shift_downswing_m": False,
        "trunk_forward_tilt_change_deg": True,
    }


def test_thresholds_subset_and_magnitude() -> None:
    out = evaluate_thresholds(
        _metrics(), {"pelvis_lateral_shift_downswing_m": 0.04}, provenance="p"
    )
    assert out == {"pelvis_lateral_shift_downswing_m": True}


@pytest.mark.parametrize("prov", ["", "   ", None])
def test_thresholds_require_provenance(prov) -> None:
    with pytest.raises(ValueError):
        evaluate_thresholds(_metrics(), {"pelvis_toward_ball_shift_m": 0.1}, prov)


def test_thresholds_reject_unknown_negative_nonfinite_empty() -> None:
    m = _metrics()
    for bad in (
        {"unknown_metric": 1.0},
        {"pelvis_toward_ball_shift_m": -0.1},
        {"pelvis_toward_ball_shift_m": float("nan")},
        {},
    ):
        with pytest.raises(ValueError):
            evaluate_thresholds(m, bad, provenance="p")


def test_module_source_has_no_coaching_vocabulary() -> None:
    text = Path(spm.__file__).read_text(encoding="utf-8").lower()
    for word in ("fault", "early extension", "over the top", "casting", "sway"):
        assert word not in text
    assert "fault" not in " ".join(
        f.name for f in dataclasses.fields(SwingPostureMetrics)
    )
