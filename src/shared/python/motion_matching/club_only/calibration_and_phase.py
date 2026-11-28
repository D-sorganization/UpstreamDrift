"""Frozen per-club calibration and monotone slope-bounded phase alignment (#15787)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.model_appearance.hand_backends.registration import (
    kabsch_rigid_transform,
)

_SO3_TOL = 1.0e-6
_SLOPE_EPS = 1.0e-12


def _finite(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return val


def _freeze_vec(
    arr: NDArray[np.floating], shape: tuple[int, ...], name: str
) -> NDArray[np.float64]:
    out = np.array(arr, dtype=np.float64, copy=True)
    if out.shape != shape or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    out.setflags(write=False)
    return out


def _check_so3(rot: NDArray[np.float64]) -> None:
    if not np.allclose(rot @ rot.T, np.eye(3), atol=_SO3_TOL):
        raise ValueError("marker_to_club_rotation must be orthogonal")
    det = float(np.linalg.det(rot))
    if abs(det - 1.0) > _SO3_TOL:
        raise ValueError(f"marker_to_club_rotation must have det=+1, got {det:.8f}")


def _check_phase_bounds(
    phi_top: float, phi_impact: float, slope_min: float, slope_max: float
) -> tuple[float, float, float, float]:
    p_top, p_imp = _finite("phi_top", phi_top), _finite("phi_impact", phi_impact)
    s_min, s_max = _finite("slope_min", slope_min), _finite("slope_max", slope_max)
    if not (0.0 < p_top < p_imp < 1.0):
        raise ValueError("phase anchors must satisfy 0 < phi_top < phi_impact < 1")
    if s_min <= 0.0 or s_min > s_max:
        raise ValueError("slope bounds must satisfy 0 < slope_min <= slope_max")
    return p_top, p_imp, s_min, s_max


@dataclass(frozen=True)
class ClubCalibration:
    """Frozen per-club SE(3) marker calibration and bounded hand-offset contract."""

    club_id: str
    length_m: float
    lie_angle_deg: float
    marker_to_club_rotation: NDArray[np.float64]
    marker_to_club_translation_m: NDArray[np.float64]
    lead_hand_offset_m: float
    trail_hand_offset_m: float
    max_hand_offset_m: float
    held_out_rmse_m: float
    held_out_max_m: float
    calibration_hash: str

    def __post_init__(self) -> None:
        if not self.club_id or not self.calibration_hash:
            raise ValueError("club_id and calibration_hash must be non-empty")
        if _finite("length_m", self.length_m) <= 0.0:
            raise ValueError("length_m must be > 0")
        if not (0.0 < _finite("lie_angle_deg", self.lie_angle_deg) < 90.0):
            raise ValueError("lie_angle_deg must be in (0, 90)")
        rot = _freeze_vec(self.marker_to_club_rotation, (3, 3), "rotation")
        _check_so3(rot)
        trans = _freeze_vec(self.marker_to_club_translation_m, (3,), "translation")
        object.__setattr__(self, "marker_to_club_rotation", rot)
        object.__setattr__(self, "marker_to_club_translation_m", trans)
        cap = _finite("max_hand_offset_m", self.max_hand_offset_m)
        lead = _finite("lead_hand_offset_m", self.lead_hand_offset_m)
        trail = _finite("trail_hand_offset_m", self.trail_hand_offset_m)
        if cap <= 0.0 or abs(lead) > cap or abs(trail) > cap or lead == trail:
            raise ValueError(f"hand_offset bounds violated (cap={cap})")
        rmse = _finite("held_out_rmse_m", self.held_out_rmse_m)
        max_e = _finite("held_out_max_m", self.held_out_max_m)
        if rmse < 0.0 or max_e < rmse - _SLOPE_EPS:
            raise ValueError("held-out error metrics must be >= 0 with max >= rmse")


@precondition(
    lambda club_id, *a, **kw: isinstance(club_id, str) and bool(club_id),
    "club_id must be a non-empty string",
)
@postcondition(lambda r: isinstance(r, ClubCalibration), "must return ClubCalibration")
def calibrate_and_validate_club(
    club_id: str,
    marker_points_m: NDArray[np.floating],
    club_points_m: NDArray[np.floating],
    calib_frame_indices: Sequence[int],
    held_out_frame_indices: Sequence[int],
    *,
    length_m: float,
    lie_angle_deg: float,
    lead_hand_offset_m: float = 0.04,
    trail_hand_offset_m: float = -0.04,
    max_hand_offset_m: float = 0.15,
    max_held_out_rmse_m: float = 0.005,
    max_held_out_point_error_m: float | None = None,
) -> ClubCalibration:
    """Fit marker-to-club SE(3) via Kabsch SVD and validate on disjoint held-out frames."""
    m_arr = np.asarray(marker_points_m, dtype=np.float64)
    c_arr = np.asarray(club_points_m, dtype=np.float64)
    if m_arr.ndim != 3 or m_arr.shape[2] != 3 or m_arr.shape[1] < 3:
        raise ValueError("marker_points_m must have shape (F, M, 3) with M >= 3")
    if c_arr.shape == m_arr.shape[1:]:
        c_arr = np.broadcast_to(c_arr, m_arr.shape).copy()
    elif c_arr.shape != m_arr.shape:
        raise ValueError("club_points_m shape must match marker_points_m")

    cal_i = [int(i) for i in calib_frame_indices]
    held_i = [int(i) for i in held_out_frame_indices]
    n_f = m_arr.shape[0]
    if not cal_i or not held_i or any(i < 0 or i >= n_f for i in cal_i + held_i):
        raise ValueError("calibration and held-out indices must be non-empty and valid")
    if not set(cal_i).isdisjoint(set(held_i)):
        raise ValueError("calibration and held-out frame indices must be disjoint")

    cal_m, cal_c = m_arr[cal_i].reshape(-1, 3), c_arr[cal_i].reshape(-1, 3)
    held_m, held_c = m_arr[held_i].reshape(-1, 3), c_arr[held_i].reshape(-1, 3)
    if not all(np.all(np.isfinite(x)) for x in (cal_m, cal_c, held_m, held_c)):
        raise ValueError("marker and club coordinates must be finite")

    tf = kabsch_rigid_transform(cal_m, cal_c)
    rot = np.array(tf[:3, :3], dtype=np.float64, copy=True)
    trans = np.array(tf[:3, 3], dtype=np.float64, copy=True)
    _check_so3(rot)

    errs = np.linalg.norm((held_m @ rot.T + trans) - held_c, axis=1)
    rmse_m, max_err_m = float(np.sqrt(np.mean(errs**2))), float(np.max(errs))
    rmse_ceil = _finite("max_held_out_rmse_m", max_held_out_rmse_m)
    if rmse_ceil <= 0.0 or rmse_m > rmse_ceil:
        raise ValueError(f"held-out RMSE {rmse_m:.5f} m exceeds {rmse_ceil:.5f} m")
    if max_held_out_point_error_m is not None:
        max_ceil = _finite("max_held_out_point_error_m", max_held_out_point_error_m)
        if max_err_m > max_ceil:
            raise ValueError(f"held-out max {max_err_m:.5f} m exceeds {max_ceil:.5f} m")

    tail = (
        float(lead_hand_offset_m),
        float(trail_hand_offset_m),
        float(max_hand_offset_m),
        rmse_m,
        max_err_m,
    )
    l_m, lie_d = float(length_m), float(lie_angle_deg)
    raw = json.dumps(
        {
            "id": club_id,
            "r": np.round(rot, 9).tolist(),
            "t": np.round(trans, 9).tolist(),
            "s": [round(x, 9) for x in (l_m, lie_d, *tail)],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return ClubCalibration(club_id, l_m, lie_d, rot, trans, *tail, digest)


@dataclass(frozen=True)
class PhaseAnchorInterval:
    """Bounded uncertainty interval ``[t_min_s, t_max_s]`` for one swing event."""

    t_min_s: float
    t_max_s: float
    nominal_s: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        lo, hi = _finite("t_min_s", self.t_min_s), _finite("t_max_s", self.t_max_s)
        if lo > hi:
            raise ValueError(f"t_min_s={lo} > t_max_s={hi}")
        if self.nominal_s is not None and not (
            lo <= _finite("nominal_s", self.nominal_s) <= hi
        ):
            raise ValueError(f"nominal_s={self.nominal_s} outside [{lo}, {hi}]")

    @property
    def target_s(self) -> float:
        """Preferred timestamp inside the interval when feasible."""
        if self.nominal_s is not None:
            return float(self.nominal_s)
        return 0.5 * (self.t_min_s + self.t_max_s)


def _solve_interval_anchors(
    intervals: tuple[PhaseAnchorInterval, ...],
    phi_top: float,
    phi_impact: float,
    slope_min: float,
    slope_max: float,
) -> tuple[float, float, float, float]:
    p_top, p_imp, s_min, s_max = _check_phase_bounds(
        phi_top, phi_impact, slope_min, slope_max
    )
    phases = (0.0, p_top, p_imp, 1.0)
    d_min = [(phases[i + 1] - phases[i]) / s_max for i in range(3)]
    d_max = [(phases[i + 1] - phases[i]) / s_min for i in range(3)]
    lo, hi = [iv.t_min_s for iv in intervals], [iv.t_max_s for iv in intervals]
    for i in range(3):
        lo[i + 1] = max(lo[i + 1], lo[i] + d_min[i])
        hi[i + 1] = min(hi[i + 1], hi[i] + d_max[i])
    for i in (2, 1, 0):
        lo[i] = max(lo[i], lo[i + 1] - d_max[i])
        hi[i] = min(hi[i], hi[i + 1] - d_min[i])
    if any(a > b + _SLOPE_EPS for a, b in zip(lo, hi, strict=True)):
        raise ValueError("infeasible anchor intervals: no valid alignment exists")
    chosen = [float(np.clip(intervals[0].target_s, lo[0], hi[0]))]
    for i in range(3):
        s_lo = max(lo[i + 1], chosen[i] + d_min[i])
        s_hi = max(s_lo, min(hi[i + 1], chosen[i] + d_max[i]))
        chosen.append(float(np.clip(intervals[i + 1].target_s, s_lo, s_hi)))
    return (chosen[0], chosen[1], chosen[2], chosen[3])


@dataclass(frozen=True)
class PhaseAlignmentMap:
    """Strictly monotone, slope-bounded piecewise phase map ``phi(t) in [0, 1]``."""

    t_address_s: float
    t_top_s: float
    t_impact_s: float
    t_finish_s: float
    phi_top: float = 0.4
    phi_impact: float = 0.75
    slope_min: float = 0.1
    slope_max: float = 5.0

    def __post_init__(self) -> None:
        raw = (self.t_address_s, self.t_top_s, self.t_impact_s, self.t_finish_s)
        times = tuple(_finite(f"t_{i}", t) for i, t in enumerate(raw))
        p_top, p_imp, s_min, s_max = _check_phase_bounds(
            self.phi_top, self.phi_impact, self.slope_min, self.slope_max
        )
        phases = (0.0, p_top, p_imp, 1.0)
        for i in range(3):
            if times[i] >= times[i + 1]:
                raise ValueError("event timestamps must be strictly monotone")
            slope = (phases[i + 1] - phases[i]) / (times[i + 1] - times[i])
            if slope < s_min - _SLOPE_EPS or slope > s_max + _SLOPE_EPS:
                raise ValueError(f"segment {i} slope {slope:.4f} violates bounds")

    @classmethod
    @postcondition(lambda r: isinstance(r, PhaseAlignmentMap), "must return map")
    def from_intervals(
        cls,
        address: PhaseAnchorInterval,
        top: PhaseAnchorInterval,
        impact: PhaseAnchorInterval,
        finish: PhaseAnchorInterval,
        *,
        phi_top: float = 0.4,
        phi_impact: float = 0.75,
        slope_min: float = 0.1,
        slope_max: float = 5.0,
    ) -> PhaseAlignmentMap:
        """Resolve uncertain anchor intervals into a valid monotone slope-bounded map."""
        t0, t1, t2, t3 = _solve_interval_anchors(
            (address, top, impact, finish), phi_top, phi_impact, slope_min, slope_max
        )
        return cls(t0, t1, t2, t3, phi_top, phi_impact, slope_min, slope_max)

    @precondition(lambda self, t_s: math.isfinite(float(t_s)), "t_s must be finite")
    @postcondition(lambda r: 0.0 <= r[0] <= 1.0 and r[1] > 0.0, "phi in [0,1]")
    def evaluate_phase(self, t_s: float) -> tuple[float, float]:
        """Return ``(phi, dphi_dt)`` at physical time ``t_s``."""
        ta, tt, ti, tf = (
            self.t_address_s,
            self.t_top_s,
            self.t_impact_s,
            self.t_finish_s,
        )
        t = float(t_s)
        if t < ta - _SLOPE_EPS or t > tf + _SLOPE_EPS:
            raise ValueError(f"t_s={t} outside [{ta}, {tf}]")
        t = min(max(t, ta), tf)
        if t <= tt:
            t0, t1, p0, p1 = ta, tt, 0.0, self.phi_top
        elif t <= ti:
            t0, t1, p0, p1 = tt, ti, self.phi_top, self.phi_impact
        else:
            t0, t1, p0, p1 = ti, tf, self.phi_impact, 1.0
        slope = (p1 - p0) / (t1 - t0)
        return float(np.clip(p0 + slope * (t - t0), 0.0, 1.0)), float(slope)

    def evaluate_tour_prior(
        self,
        t_s: float,
        phase_grid: NDArray[np.floating],
        q_tour_grid: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Interpolate ``q_tour(phi(t_s))``, preserving physical time vs. phase."""
        phi, _ = self.evaluate_phase(t_s)
        g, q = (
            np.asarray(phase_grid, dtype=np.float64),
            np.asarray(q_tour_grid, dtype=np.float64),
        )
        if (
            g.ndim != 1
            or g.size < 2
            or not np.all(np.isfinite(g))
            or np.any(np.diff(g) <= 0.0)
        ):
            raise ValueError("phase_grid must be strictly increasing finite 1D (>= 2)")
        if phi < g[0] - _SLOPE_EPS or phi > g[-1] + _SLOPE_EPS:
            raise ValueError("phi(t_s) outside phase_grid span")
        if q.shape[0] != g.size or q.ndim not in (1, 2) or not np.all(np.isfinite(q)):
            raise ValueError("q_tour_grid must be finite (M,) or (M, D)")
        idx = int(np.clip(np.searchsorted(g, phi, side="right") - 1, 0, g.size - 2))
        alpha = float(np.clip((phi - g[idx]) / (g[idx + 1] - g[idx]), 0.0, 1.0))
        return np.asarray((1.0 - alpha) * q[idx] + alpha * q[idx + 1], dtype=np.float64)

    def eval_tour_prior_at_time(
        self,
        t_s: float,
        q_tour_knots: NDArray[np.floating],
        phase_grid: NDArray[np.floating] | None = None,
    ) -> NDArray[np.float64]:
        """Query tour prior knots at ``phi(t_s)`` (uniform ``[0, 1]`` grid if omitted)."""
        knots = np.asarray(q_tour_knots, dtype=np.float64)
        grid = (
            np.linspace(0.0, 1.0, knots.shape[0], dtype=np.float64)
            if phase_grid is None
            else phase_grid
        )
        return self.evaluate_tour_prior(t_s, grid, knots)
