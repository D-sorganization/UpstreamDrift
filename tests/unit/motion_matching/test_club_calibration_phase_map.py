"""Unit tests for frozen club calibration and slope-bounded phase alignment (#15787)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.club_only.calibration_and_phase import (
    ClubCalibration,
    PhaseAlignmentMap,
    PhaseAnchorInterval,
    calibrate_and_validate_club,
)

pytestmark = pytest.mark.unit


def _synthetic_dataset(
    frames: int = 12, noise_m: float = 2.0e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(15787)
    r_true = Rotation.from_euler("zyx", [35.0, -18.0, 62.0], degrees=True).as_matrix()
    t_true = np.array([0.12, -0.45, 0.88], dtype=np.float64)
    base = np.array(
        [[0, 0, 0], [0.05, 0.08, 0.01], [-0.04, 0.06, -0.02], [0.01, -0.65, 0]]
    )
    club_pts = np.empty((frames, len(base), 3), dtype=np.float64)
    marker_pts = np.empty_like(club_pts)
    for idx in range(frames):
        f_rot = Rotation.from_euler("x", 4.0 * idx, degrees=True).as_matrix()
        club_pts[idx] = base @ f_rot.T + np.array([0.02 * idx, 0.0, 0.01 * idx])
        marker_pts[idx] = (club_pts[idx] - t_true) @ r_true
    noisy = marker_pts + rng.normal(0.0, noise_m, size=marker_pts.shape)
    return noisy, club_pts, r_true, t_true


def _run_cal(
    m_pts: np.ndarray,
    c_pts: np.ndarray,
    cal_i: list[int] | None = None,
    held_i: list[int] | None = None,
    lead: float = 0.04,
    cap: float = 0.10,
    rmse: float = 0.002,
) -> ClubCalibration:
    return calibrate_and_validate_club(
        "driver",
        m_pts,
        c_pts,
        cal_i or [0, 1, 2, 3, 4, 5],
        held_i or [6, 7, 8, 9, 10, 11],
        length_m=1.156,
        lie_angle_deg=56.5,
        lead_hand_offset_m=lead,
        max_hand_offset_m=cap,
        max_held_out_rmse_m=rmse,
    )


def test_phase_anchors_require_strict_monotonic_ordering() -> None:
    PhaseAlignmentMap(0.0, 0.8, 1.1, 1.8, phi_top=0.4, phi_impact=0.75)
    with pytest.raises(ValueError, match="monotone"):
        PhaseAlignmentMap(0.0, 0.8, 0.8, 1.8)
    with pytest.raises(ValueError, match="monotone"):
        PhaseAlignmentMap(0.0, 1.0, 0.9, 1.8)
    with pytest.raises(ValueError, match="phi"):
        PhaseAlignmentMap(0.0, 0.8, 1.1, 1.8, phi_top=0.8, phi_impact=0.75)


def test_phase_map_enforces_slope_bounds_and_evaluates_piecewise() -> None:
    def make(t1: float, t2: float, t3: float) -> PhaseAlignmentMap:
        return PhaseAlignmentMap(0.0, t1, t2, t3, 0.4, 0.75, 0.4, 1.5)

    pmap = make(0.8, 1.05, 1.55)
    assert pmap.evaluate_phase(0.0) == pytest.approx((0.0, 0.5))
    assert pmap.evaluate_phase(0.4) == pytest.approx((0.2, 0.5))
    assert pmap.evaluate_phase(0.8) == pytest.approx((0.4, 0.5))
    assert pmap.evaluate_phase(1.05) == pytest.approx((0.75, 1.4))
    assert pmap.evaluate_phase(1.55)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="slope"):
        make(0.8, 0.9, 1.4)
    with pytest.raises(ValueError, match="slope"):
        make(2.0, 2.3, 2.8)
    with pytest.raises(ValueError, match="outside"):
        pmap.evaluate_phase(1.8)


def test_uncertain_event_intervals_resolve_or_fail_closed() -> None:
    iv = PhaseAnchorInterval
    ok = (iv(0.0, 0.05, 0.0), iv(0.2, 0.55, 0.3), iv(0.8, 1.0, 0.9), iv(1.1, 1.4, 1.25))
    pmap = PhaseAlignmentMap.from_intervals(
        *ok, phi_top=0.4, phi_impact=0.75, slope_min=0.5, slope_max=1.0
    )
    assert 0.0 <= pmap.t_address_s <= 0.05 and 0.2 <= pmap.t_top_s <= 0.55
    assert 0.8 <= pmap.t_impact_s <= 1.0 and 1.1 <= pmap.t_finish_s <= 1.4
    for t_val in (pmap.t_address_s, pmap.t_top_s, pmap.t_impact_s, pmap.t_finish_s):
        assert 0.5 - 1e-9 <= pmap.evaluate_phase(t_val)[1] <= 1.0 + 1e-9
    bad = (iv(0.0, 0.1), iv(0.2, 0.3), iv(0.35, 0.45), iv(0.45, 0.5))
    with pytest.raises(ValueError, match="infeasible"):
        PhaseAlignmentMap.from_intervals(
            *bad, phi_top=0.4, phi_impact=0.75, slope_min=0.5, slope_max=1.0
        )


def test_calibrate_and_validate_club_recovers_transform_and_enforces_gates() -> None:
    m_pts, c_pts, r_true, t_true = _synthetic_dataset()
    cal = _run_cal(m_pts, c_pts)
    assert isinstance(cal, ClubCalibration) and cal.club_id == "driver"
    assert cal.length_m == pytest.approx(1.156)
    assert cal.lie_angle_deg == pytest.approx(56.5)
    assert np.linalg.det(cal.marker_to_club_rotation) == pytest.approx(1.0, abs=1e-6)
    assert cal.marker_to_club_rotation == pytest.approx(r_true, abs=2e-3)
    assert cal.marker_to_club_translation_m == pytest.approx(t_true, abs=2e-3)
    assert cal.held_out_rmse_m < 0.002 and cal.held_out_max_m >= cal.held_out_rmse_m
    assert len(cal.calibration_hash) == 64

    with pytest.raises(FrozenInstanceError):
        cal.length_m = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        cal.marker_to_club_rotation[0, 0] = 99.0
    with pytest.raises(ValueError, match="disjoint"):
        _run_cal(m_pts, c_pts, cal_i=[0, 1, 2, 3], held_i=[3, 4, 5])
    bad = m_pts.copy()
    bad[8, 0] += [0.05, -0.04, 0.03]
    with pytest.raises(ValueError, match="held-out"):
        _run_cal(bad, c_pts)
    with pytest.raises(ValueError, match="hand_offset"):
        _run_cal(m_pts, c_pts, lead=0.18, cap=0.10)


def test_tour_prior_query_preserves_physical_time_vs_phase_separation() -> None:
    g = np.linspace(0.0, 1.0, 5)
    q = np.column_stack([np.sin(np.pi * g), g**2])
    slow = PhaseAlignmentMap(0.0, 1.2, 1.6, 2.4, phi_top=0.5, phi_impact=0.75)
    fast = PhaseAlignmentMap(0.0, 0.6, 0.9, 1.4, phi_top=0.5, phi_impact=0.75)
    assert slow.evaluate_tour_prior(1.2, g, q) == pytest.approx(
        fast.evaluate_tour_prior(0.6, g, q)
    )
    assert slow.eval_tour_prior_at_time(1.6, q) == pytest.approx(
        fast.eval_tour_prior_at_time(0.9, q)
    )
    assert fast.evaluate_phase(0.3)[1] == pytest.approx(
        2.0 * slow.evaluate_phase(0.6)[1]
    )
