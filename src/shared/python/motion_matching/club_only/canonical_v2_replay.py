"""Canonical-v2 flexible-shaft closed-loop replay and impact badges (#15787 WP5)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_only.observation import (
    require_strictly_increasing_timestamps,
)
from src.shared.python.motion_matching.club_shaft_chain import (
    DAMPING_RATIO,
    DEFAULT_FLEX,
    flex_frequency_hz,
    flex_ratings,
)

_BALL_BOUNDS = {"wiffle": (0.05, 0.85), "prov1": (1.20, 6.00)}
_BALL_CONTACT_S = {"wiffle": 1.2e-3, "prov1": 4.5e-4}
_INERTIA, _CLUB_LEN, _HEAD_MASS = 0.35, 1.156, 0.20
_KP, _KD, _LAG_GAIN, _IMP_COUPLE = 140.0, 20.0, 0.25, 0.03


class BallType(str, Enum):
    """Supported golf ball classes with distinct impact momentum transfer."""

    WIFFLE = "wiffle"
    PROV1 = "prov1"


class ReplayBadge(str, Enum):
    """Interval-scoped replay qualification badges for canonical-v2 output."""

    PRE_IMPACT_REPLAYED = "pre-impact-replayed"
    FULL_SWING_REPLAYED = "full-swing-replayed"
    KINEMATIC_ONLY = "kinematic-only"


@dataclass(frozen=True)
class ReplayDriftBudget:
    """Versioned, content-hashed drift and ceiling budget for flex-shaft replay."""

    budget_version: str = "canonical-v2-flex/1.0.0"
    pre_impact_pos_tol_m: float = 0.025
    pre_impact_rot_tol_rad: float = 0.025
    post_impact_pos_tol_m: float = 0.040
    post_impact_rot_tol_rad: float = 0.040
    max_torque_nm: float = 120.0
    max_shaft_deflection_m: float = 0.12
    budget_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate positive tolerances and populate or verify budget_sha256."""
        if not self.budget_version.strip():
            raise ValueError("budget_version required")
        vals = (
            self.pre_impact_pos_tol_m,
            self.pre_impact_rot_tol_rad,
            self.post_impact_pos_tol_m,
            self.post_impact_rot_tol_rad,
            self.max_torque_nm,
            self.max_shaft_deflection_m,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in vals):
            raise ValueError("budget thresholds must be finite and > 0")
        if not self.budget_sha256:
            p = asdict(self)
            p.pop("budget_sha256", None)
            sha = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
            object.__setattr__(self, "budget_sha256", sha)
        else:
            d = self.budget_sha256.lower()
            if len(d) != 64:
                raise ValueError("budget_sha256 must be 64 hex chars")
            int(d, 16)
            object.__setattr__(self, "budget_sha256", d)


@dataclass(frozen=True)
class ImpactImpulseEstimate:
    """Bounded external clubhead impact impulse and 240 Hz ambiguity interval."""

    ball_type: BallType
    nominal_impulse_ns: float
    compatible_impulse_interval_ns: tuple[float, float]
    under_resolved_by_240hz: bool

    def __post_init__(self) -> None:
        """Validate non-negative impulse and ordered interval bounds."""
        j0, j1 = self.compatible_impulse_interval_ns
        ok = math.isfinite(self.nominal_impulse_ns) and self.nominal_impulse_ns >= 0.0
        ok = ok and math.isfinite(j0) and math.isfinite(j1)
        if not (ok and 0.0 <= j0 <= self.nominal_impulse_ns <= j1):
            raise ValueError("invalid impact impulse bounds")


def estimate_bounded_impact_impulse(
    ball_type: BallType,
    pre_speed_mps: float,
    post_speed_mps: float,
    *,
    head_mass_kg: float = _HEAD_MASS,
    sample_rate_hz: float = 240.0,
) -> ImpactImpulseEstimate:
    """Estimate bounded clubhead impact impulse and compatible 240 Hz interval."""
    if not all(math.isfinite(v) and v >= 0.0 for v in (pre_speed_mps, post_speed_mps)):
        raise ValueError("head speeds must be finite and >= 0")
    if head_mass_kg <= 0.0 or sample_rate_hz <= 0.0:
        raise ValueError("mass and sample_rate_hz must be > 0")
    j_lo, j_hi = _BALL_BOUNDS[ball_type.value]
    tc = _BALL_CONTACT_S[ball_type.value]
    raw_dp = head_mass_kg * max(0.0, pre_speed_mps - post_speed_mps)
    nom = min(j_hi, max(j_lo, raw_dp))
    dt = 1.0 / sample_rate_hz
    under = dt > tc
    band = nom * (0.20 * (1.0 - tc / dt)) + 0.03 if under else nom * 0.03 + 0.005
    bounds = (float(max(0.0, nom - band)), float(nom + band))
    return ImpactImpulseEstimate(ball_type, float(nom), bounds, under)


@dataclass(frozen=True)
class FlexShaftReplayRequest:
    """Input bundle for canonical-v2 flexible-shaft closed-loop replay."""

    candidate_id: str
    ball_type: BallType
    timestamps_s: NDArray[np.float64]
    q_ref: NDArray[np.float64]
    v_ref: NDArray[np.float64]
    flex_rating: str = DEFAULT_FLEX
    budget: ReplayDriftBudget | None = None
    impact_time_s: float | None = None
    tau_feedforward_nm: NDArray[np.float64] | None = None
    impact_verified: bool = True
    post_impact_disturbance_nm: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate trajectory shapes, flex_rating, and finite arrays."""
        if not self.candidate_id.strip():
            raise ValueError("candidate_id required")
        valid = flex_ratings()
        if self.flex_rating not in valid:
            raise ValueError(f"unknown flex_rating {self.flex_rating!r}; valid {valid}")
        t = require_strictly_increasing_timestamps(np.asarray(self.timestamps_s, float))
        q, v = np.asarray(self.q_ref, float), np.asarray(self.v_ref, float)
        fin = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(v)))
        if q.ndim != 2 or q.shape != v.shape or q.shape[0] != t.size or not fin:
            raise ValueError("q_ref and v_ref must be finite with shape (N, dof)")
        object.__setattr__(self, "timestamps_s", t.copy())
        object.__setattr__(self, "q_ref", q.copy())
        object.__setattr__(self, "v_ref", v.copy())
        for opt in ("tau_feedforward_nm", "post_impact_disturbance_nm"):
            val = getattr(self, opt)
            if val is not None:
                arr = np.asarray(val, float)
                if arr.shape != q.shape or not np.all(np.isfinite(arr)):
                    raise ValueError(f"invalid {opt}")
                object.__setattr__(self, opt, arr.copy())


@dataclass(frozen=True)
class CanonicalV2ReplayResult:
    """Closed-loop flexible-shaft replay outcome and interval-scoped badge."""

    ball_type: BallType
    flex_rating: str
    flex_frequency_hz: float
    badge: ReplayBadge
    budget: ReplayDriftBudget
    pre_impact_max_pos_drift_m: float
    pre_impact_max_rot_drift_rad: float
    post_impact_max_pos_drift_m: float
    post_impact_max_rot_drift_rad: float
    peak_torque_nm: float
    peak_tip_deflection_m: float
    post_impact_recovery_fraction: float
    pre_impact_passed: bool
    post_impact_passed: bool
    torque_within_budget: bool
    impact_verified: bool
    impact_impulse: ImpactImpulseEstimate | None
    failure_reasons: tuple[str, ...]


def _rollout(
    req: FlexShaftReplayRequest,
    budget: ReplayDriftBudget,
    idx: int | None,
    imp: ImpactImpulseEstimate | None,
    w1: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Run closed-loop torque-bounded rollout over the flexible-shaft plant."""
    t, qr, vr = req.timestamps_s, req.q_ref, req.v_ref
    n, dof = qr.shape
    q, v, tau = np.zeros_like(qr), np.zeros_like(vr), np.zeros_like(qr)
    delta, d_dot = np.zeros(n), np.zeros(n)
    q[0], v[0] = qr[0], vr[0]
    dist, prev_ff = req.post_impact_disturbance_nm, np.zeros(dof)
    for i in range(n - 1):
        dt = float(t[i + 1] - t[i])
        if req.tau_feedforward_nm is not None:
            ff = req.tau_feedforward_nm[i]
        elif idx is not None and i == idx:
            ff = prev_ff
        else:
            ff = _INERTIA * (vr[i + 1] - vr[i]) / dt
        prev_ff = ff
        raw = ff + _KP * (qr[i] - q[i]) + _KD * (vr[i] - v[i])
        tau[i] = raw
        clipped = np.clip(raw, -budget.max_torque_nm, budget.max_torque_nm)
        a_hub = (clipped if dist is None else clipped + dist[i]) / _INERTIA
        v_next, dd = v[i] + a_hub * dt, float(d_dot[i])
        if idx is not None and i == idx and imp is not None:
            dv_head = imp.nominal_impulse_ns / _HEAD_MASS
            u = v[i] / max(float(np.linalg.norm(v[i])), 1e-9)
            v_next = v_next - (dv_head / _CLUB_LEN) * u
            dd -= dv_head * _IMP_COUPLE
        v[i + 1], q[i + 1] = v_next, q[i] + 0.5 * (v[i] + v_next) * dt
        s_acc = (
            -_LAG_GAIN * float(np.linalg.norm(a_hub))
            - 2.0 * DAMPING_RATIO * w1 * dd
            - (w1**2) * float(delta[i])
        )
        d_dot[i + 1] = dd + s_acc * dt
        delta[i + 1] = float(delta[i]) + float(d_dot[i + 1]) * dt
    tau[-1] = tau[-2]
    return q, tau, delta, d_dot


def _check_slice(
    pos: NDArray[np.float64],
    rot: NDArray[np.float64],
    tau: NDArray[np.float64],
    defl: NDArray[np.float64],
    slc: slice,
    tols: tuple[float, float, float, float],
) -> tuple[float, float, bool]:
    """Return (max_pos_m, max_rot_rad, within_bounds) over slice."""
    p, r = float(np.max(pos[slc])), float(np.max(rot[slc]))
    t_pk, d_pk = float(np.max(tau[slc])), float(np.max(defl[slc]))
    ok = p <= tols[0] and r <= tols[1] and t_pk <= tols[2] and d_pk <= tols[3]
    return p, r, ok


def _badge_and_reasons(
    pre_ok: bool, post_ok: bool, imp_ok: bool
) -> tuple[ReplayBadge, tuple[str, ...]]:
    """Select interval-scoped ReplayBadge and diagnostic failure reasons."""
    if not pre_ok:
        return ReplayBadge.KINEMATIC_ONLY, ("pre_impact_drift_or_torque_exceeded",)
    if imp_ok and post_ok:
        return ReplayBadge.FULL_SWING_REPLAYED, ()
    reasons = tuple(
        r
        for r, c in (
            ("impact_unverified", not imp_ok),
            ("post_impact_drift_or_recovery_failed", not post_ok),
        )
        if c
    )
    return ReplayBadge.PRE_IMPACT_REPLAYED, reasons


def run_canonical_v2_flex_replay(
    request: FlexShaftReplayRequest,
) -> CanonicalV2ReplayResult:
    """Run torque-bounded flex-shaft replay and emit an interval-scoped badge."""
    b = request.budget if request.budget is not None else ReplayDriftBudget()
    f_hz = flex_frequency_hz(request.flex_rating)
    w1, t, idx, imp_est, imp_ok = (
        2.0 * math.pi * f_hz,
        request.timestamps_s,
        None,
        None,
        False,
    )
    if request.impact_time_s is not None and float(
        t[0]
    ) < request.impact_time_s < float(t[-1]):
        cand = int(np.argmin(np.abs(t - request.impact_time_s)))
        idx = cand if 0 < cand < t.size - 1 else None
    if idx is not None:
        vp = float(np.linalg.norm(request.v_ref[idx])) * _CLUB_LEN
        vq = float(np.linalg.norm(request.v_ref[idx + 1])) * _CLUB_LEN
        rate = 1.0 / float(np.mean(np.diff(t)))
        imp_est = estimate_bounded_impact_impulse(
            request.ball_type, vp, vq, sample_rate_hz=rate
        )
        j0, j1 = _BALL_BOUNDS[request.ball_type.value]
        imp_ok = bool(
            request.impact_verified
            and 0.8 * j0 <= _HEAD_MASS * max(0.0, vp - vq) <= 1.2 * j1
        )
    q, tau, delta, d_dot = _rollout(request, b, idx, imp_est, w1)
    rot = np.linalg.norm(q - request.q_ref, axis=1)
    pos, t_mag, d_abs = rot * _CLUB_LEN, np.max(np.abs(tau), axis=1), np.abs(delta)
    pe = (idx + 1) if idx is not None else q.shape[0]
    pre_tols = (
        b.pre_impact_pos_tol_m,
        b.pre_impact_rot_tol_rad,
        b.max_torque_nm,
        b.max_shaft_deflection_m,
    )
    pre_p, pre_r, pre_ok = _check_slice(pos, rot, t_mag, d_abs, slice(0, pe), pre_tols)
    post_p = post_r = rec = 0.0
    post_ok = False
    if idx is not None:
        ps = slice(idx + 1, q.shape[0])
        post_tols = (
            b.post_impact_pos_tol_m,
            b.post_impact_rot_tol_rad,
            b.max_torque_nm,
            b.max_shaft_deflection_m,
        )
        post_p, post_r, post_b_ok = _check_slice(pos, rot, t_mag, d_abs, ps, post_tols)
        amp = np.sqrt(delta[ps] ** 2 + (d_dot[ps] / w1) ** 2)
        pk, end = float(np.max(amp)), float(amp[-1])
        rec = 1.0 - min(1.0, end / max(pk, 1e-12)) if pk > 1e-9 else 1.0
        post_ok = post_b_ok and (end < pk if pk > 1e-9 else True)
    badge, reasons = _badge_and_reasons(pre_ok, post_ok, imp_ok)
    pk_tau = float(np.max(t_mag))
    return CanonicalV2ReplayResult(
        request.ball_type,
        request.flex_rating,
        f_hz,
        badge,
        b,
        pre_p,
        pre_r,
        post_p,
        post_r,
        pk_tau,
        float(np.max(d_abs)),
        float(rec),
        pre_ok,
        post_ok,
        pk_tau <= b.max_torque_nm,
        imp_ok,
        imp_est,
        reasons,
    )
