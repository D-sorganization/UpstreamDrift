"""Canonical-v2 closed-chain constrained IK solver and audit gate (Refs #15787)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

import src.shared.python.motion_matching._geodesic as geo
import src.shared.python.motion_matching.club_only.hand_geometry as hg
from src.shared.python.motion_matching.same_input.closure import ClosurePlant

F64 = NDArray[np.float64]
_St = tuple[bool, float, int]
_FMet = tuple[float, float, float, float, int]
_Prob = tuple[
    "ClosedChainIkConfig",
    "ClubTargetPoses",
    "ClosedChainKinematics",
    ClosurePlant | None,
]


def _ro(arr: F64) -> F64:
    """Copy array to float64 and mark read-only."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class GripHeadTolerances:
    """Separate translation/rotation tolerances for grip, head, and closure."""

    grip_translation_tol_m: float = 0.005
    grip_rotation_tol_rad: float = 0.05
    head_translation_tol_m: float = 0.015
    head_rotation_tol_rad: float = 0.05
    bilateral_closure_pos_tol_m: float = 0.005
    bilateral_closure_rot_tol_rad: float = 0.05

    def __post_init__(self) -> None:
        """Validate all tolerances are finite and strictly positive."""
        vals = [float(getattr(self, f.name)) for f in fields(self)]
        if any(not math.isfinite(v) or v <= 0.0 for v in vals):
            raise ValueError("tolerances must be finite > 0")


@dataclass(frozen=True)
class JointKinematicLimits:
    """Per-joint position, velocity, and acceleration bounds."""

    q_min: F64
    q_max: F64
    qd_max: F64
    qdd_max: F64

    def __post_init__(self) -> None:
        """Validate shapes, finiteness, ordering, and freeze limit arrays."""
        vs = [_ro(getattr(self, f.name)) for f in fields(self)]
        lo, hi, qd, qdd = vs
        bad_s = lo.ndim != 1 or lo.size == 0 or any(a.shape != lo.shape for a in vs)
        bad_v = not all(np.all(np.isfinite(a)) for a in vs) or np.any(lo > hi)
        if bad_s or bad_v or np.any(qd <= 0.0) or np.any(qdd <= 0.0):
            raise ValueError("invalid joint kinematic limits")
        for f, v in zip(fields(self), vs, strict=True):
            object.__setattr__(self, f.name, v)


@dataclass(frozen=True)
class FootContactPhaseSpec:
    """Phase-windowed foot contact slip and vertical heel-rise envelope."""

    site_name: str
    start_frame: int
    end_frame: int
    anchor_xyz: F64
    max_horizontal_slip_m: float = 0.01
    max_vertical_lift_m: float = 0.005
    max_penetration_m: float = 0.005
    ground_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        """Validate phase window, anchor, and ground normal."""
        anc = _ro(self.anchor_xyz)
        ok_w = bool(self.site_name) and (0 <= self.start_frame < self.end_frame)
        if not ok_w or anc.shape != (3,) or not np.all(np.isfinite(anc)):
            raise ValueError("invalid foot contact phase spec")
        object.__setattr__(self, "anchor_xyz", anc)


@dataclass(frozen=True)
class ClubTargetPoses:
    """Target SE(3) trajectories for club grip and clubhead."""

    grip_xyz: F64
    grip_quat: F64
    head_xyz: F64
    head_quat: F64


@dataclass(frozen=True)
class ClosedChainFrameState:
    """Forward-kinematics snapshot for one posture ``q``."""

    grip_xyz: F64
    grip_quat: F64
    head_xyz: F64
    head_quat: F64
    lead_hand_xyz: F64
    lead_hand_quat: F64
    trail_hand_xyz: F64
    trail_hand_quat: F64
    foot_sites: Mapping[str, F64]


class ClosedChainKinematics(Protocol):
    """Protocol for closed-chain forward kinematics evaluation."""

    def forward_kinematics(self, q: F64) -> ClosedChainFrameState:
        """Evaluate club, hand grip sites, and foot contact sites at ``q``."""


@dataclass(frozen=True)
class ClosedChainIkConfig:
    """Configuration for canonical-v2 closed-chain IK optimization and audit."""

    tolerances: GripHeadTolerances
    joint_limits: JointKinematicLimits
    hand_offsets: hg.ModelHandFrameOffsets
    foot_phases: tuple[FootContactPhaseSpec, ...] = ()
    grip_cov_inv_sqrt: F64 | None = None
    head_cov_inv_sqrt: F64 | None = None
    closure_weight: float = 50.0
    foot_contact_weight: float = 50.0
    tour_prior_q: F64 | None = None
    tour_prior_weight: float = 0.05
    smoothness_weight: float = 0.05
    max_nfev: int = 150
    tol: float = 1e-8

    def __post_init__(self) -> None:
        """Validate solver parameters and freeze array fields."""
        if not math.isfinite(self.tol) or self.tol <= 0.0 or self.max_nfev < 1:
            raise ValueError("invalid tol or max_nfev")
        for nm in ("grip_cov_inv_sqrt", "head_cov_inv_sqrt", "tour_prior_q"):
            if (raw := getattr(self, nm)) is not None:
                arr = _ro(raw)
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{nm} non-finite")
                object.__setattr__(self, nm, arr)


@dataclass(frozen=True)
class CanonicalV2IkResult:
    """Closed-chain IK solution with per-frame residuals and fail-closed audit."""

    q_trajectory: F64
    qd_trajectory: F64
    qdd_trajectory: F64
    per_frame_residuals: Mapping[str, F64]
    violation_counts: Mapping[str, int]
    zero_kinematic_limit_violations: bool
    zero_tolerance_violations: bool
    passed: bool
    failure_reasons: tuple[str, ...]
    solver_success: bool
    solver_cost: float
    solver_nfev: int
    kinematic_preview_only: bool = True

    def __post_init__(self) -> None:
        """Enforce kinematic_preview_only=True, freeze arrays/maps, check status."""
        if not self.kinematic_preview_only or bool(self.failure_reasons) == self.passed:
            raise ValueError("invalid kinematic_preview_only or pass/failure status")
        for k in ("q_trajectory", "qd_trajectory", "qdd_trajectory"):
            object.__setattr__(self, k, _ro(getattr(self, k)))
        rm = {k: _ro(v) for k, v in self.per_frame_residuals.items()}
        object.__setattr__(self, "per_frame_residuals", MappingProxyType(rm))
        vc = MappingProxyType(dict(self.violation_counts))
        object.__setattr__(self, "violation_counts", vc)


def _so3_log(q_ref: F64, q_cur: F64) -> F64:
    """Signed 3-D rotation-vector error in ``so(3)`` from ``q_ref`` to ``q_cur``."""
    a = geo.normalise_quaternions(q_ref.reshape(1, 4))[0]
    b = geo.normalise_quaternions(q_cur.reshape(1, 4))[0]
    w = float(np.dot(a, b))
    if w < 0.0:
        b, w = -b, -w
    v = a[0] * b[1:] - b[0] * a[1:] - np.cross(a[1:], b[1:])
    vn = float(np.linalg.norm(v))
    return 2.0 * v if vn < 1e-12 else (2.0 * math.atan2(vn, w) / vn) * v


def _eval_frame(q_t: F64, t: int, p: _Prob) -> tuple[F64, ClosedChainFrameState, _FMet]:
    """Evaluate frame residual vector, kinematics state, and scalar metrics."""
    cfg, tgt, kin, plant = p
    st, off, tl = kin.forward_kinematics(q_t), cfg.hand_offsets, cfg.tolerances
    ge = np.r_[st.grip_xyz - tgt.grip_xyz[t], _so3_log(tgt.grip_quat[t], st.grip_quat)]
    he = np.r_[st.head_xyz - tgt.head_xyz[t], _so3_log(tgt.head_quat[t], st.head_quat)]
    gz = geo.quaternion_to_rotmat_series(st.grip_quat.reshape(1, 4))[0][:, 2]
    lp = st.lead_hand_xyz - (st.grip_xyz + off.lead_hand_offset_m * gz)
    tp = st.trail_hand_xyz - (st.grip_xyz + off.trail_hand_offset_m * gz)
    lr = _so3_log(st.grip_quat, st.lead_hand_quat)
    tr = _so3_log(st.grip_quat, st.trail_hand_quat)
    cv = np.r_[lp, lr, tp, tr]
    cp = max(float(np.linalg.norm(lp)), float(np.linalg.norm(tp)))
    cr = max(float(np.linalg.norm(lr)), float(np.linalg.norm(tr)))
    if plant is not None:
        pr = np.asarray(plant.closure_pose_residual(q_t), dtype=np.float64)
        cv = np.r_[cv, pr]
        cp = max(cp, float(np.linalg.norm(pr[:3] if pr.size >= 6 else pr)))
        cr = max(cr, float(np.linalg.norm(pr[3:6]))) if pr.size >= 6 else cr
    fvs, f_sl, f_vt, f_bad = [], 0.0, 0.0, 0
    for ph in cfg.foot_phases:
        if ph.start_frame <= t < ph.end_frame:
            site = st.foot_sites.get(ph.site_name)
            if site is None:
                raise ValueError(f"foot contact site {ph.site_name!r} missing")
            d = np.asarray(site, float) - ph.anchor_xyz
            n = np.asarray(ph.ground_normal, float)
            n = n / float(np.linalg.norm(n))
            vt = float(np.dot(d, n))
            hv = d - vt * n
            sl = float(np.linalg.norm(hv))
            hx = max(0.0, sl - ph.max_horizontal_slip_m)
            ux = max(0.0, vt - ph.max_vertical_lift_m)
            dx = max(0.0, -vt - ph.max_penetration_m)
            fvs.append(np.r_[(hv / sl) * hx if sl > 1e-12 else np.zeros(3), ux, dx])
            f_sl, f_vt = max(f_sl, sl), max(f_vt, ux, dx)
            f_bad += int(hx > 0.0 or ux > 0.0 or dx > 0.0)
    gs = np.repeat([1 / tl.grip_translation_tol_m, 1 / tl.grip_rotation_tol_rad], 3)
    hs = np.repeat([1 / tl.head_translation_tol_m, 1 / tl.head_rotation_tol_rad], 3)
    gc, hc = cfg.grip_cov_inv_sqrt, cfg.head_cov_inv_sqrt
    wg = gs * ge if gc is None else (gc * ge if gc.ndim == 1 else gc @ ge)
    wh = hs * he if hc is None else (hc * he if hc.ndim == 1 else hc @ he)
    fa = np.concatenate(fvs) if fvs else np.empty(0)
    rv = np.r_[wg, wh, cfg.closure_weight * cv, cfg.foot_contact_weight * fa]
    return rv, st, (cp, cr, f_sl, f_vt, f_bad)


def _audit(q: F64, dt_s: float, p: _Prob, stats: _St) -> CanonicalV2IkResult:
    """Audit kinematic bounds, tolerances, closure, and foot contacts."""
    cfg, tgt, _, _ = p
    n, lim, tl = q.shape[0], cfg.joint_limits, cfg.tolerances
    qd = np.empty_like(q)
    qd[1:], qd[0] = np.diff(q, axis=0) / dt_s, (q[1] - q[0]) / dt_s
    qdd = np.zeros_like(q)
    if n >= 3:
        qdd[1:-1] = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt_s * dt_s)
        qdd[0], qdd[-1] = qdd[1], qdd[-2]
    gp, hp, gq, hq = np.empty(n), np.empty(n), np.empty((n, 4)), np.empty((n, 4))
    cp, cr, fs, fv, fvc = np.empty(n), np.empty(n), np.zeros(n), np.zeros(n), 0
    for t in range(n):
        _, st, m = _eval_frame(q[t], t, p)
        gp[t] = float(np.linalg.norm(st.grip_xyz - tgt.grip_xyz[t]))
        hp[t] = float(np.linalg.norm(st.head_xyz - tgt.head_xyz[t]))
        gq[t], hq[t] = st.grip_quat, st.head_quat
        cp[t], cr[t], fs[t], fv[t], fvc = m[0], m[1], m[2], m[3], fvc + m[4]
    gr = geo.quaternion_geodesic_angles(tgt.grip_quat, gq)
    hr = geo.quaternion_geodesic_angles(tgt.head_quat, hq)
    pv = int(np.sum((q < lim.q_min) | (q > lim.q_max)))
    vv = int(np.sum(np.abs(qd) > lim.qd_max))
    av = int(np.sum(np.abs(qdd) > lim.qdd_max))
    gpv = int(np.sum(gp > tl.grip_translation_tol_m))
    grv = int(np.sum(gr > tl.grip_rotation_tol_rad))
    hpv = int(np.sum(hp > tl.head_translation_tol_m))
    hrv = int(np.sum(hr > tl.head_rotation_tol_rad))
    cp_t, cr_t = tl.bilateral_closure_pos_tol_m, tl.bilateral_closure_rot_tol_rad
    cv = int(np.sum((cp > cp_t) | (cr > cr_t)))
    cnt = {
        "position": pv,
        "velocity": vv,
        "acceleration": av,
        "grip_translation": gpv,
        "grip_rotation": grv,
        "head_translation": hpv,
        "head_rotation": hrv,
        "bilateral_closure": cv,
        "foot_contact": fvc,
    }
    rm = {
        "grip_translation_m": gp,
        "grip_rotation_rad": gr,
        "head_translation_m": hp,
        "head_rotation_rad": hr,
        "bilateral_closure_pos_m": cp,
        "bilateral_closure_rot_rad": cr,
        "foot_slip_m": fs,
        "foot_vertical_m": fv,
    }
    reasons = tuple(f"{k}_exceeded ({v})" for k, v in cnt.items() if v > 0)
    zk, zt = not (pv or vv or av), not (gpv or grv or hpv or hrv or cv or fvc)
    return CanonicalV2IkResult(q, qd, qdd, rm, cnt, zk, zt, zk and zt, reasons, *stats)


def solve_canonical_v2_closed_chain_ik(
    config: ClosedChainIkConfig,
    club_target_poses: ClubTargetPoses,
    initial_q: F64,
    dt_s: float,
    kinematics: ClosedChainKinematics,
    *,
    closure_plant: ClosurePlant | None = None,
) -> CanonicalV2IkResult:
    """Solve canonical-v2 closed-chain constrained IK and audit all gates."""
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("dt_s must be finite > 0")
    tgt, lim = club_target_poses, config.joint_limits
    gx = np.asarray(tgt.grip_xyz, dtype=float)
    n, nq = gx.shape[0], lim.q_min.size
    ok_t = n >= 2 and gx.shape == (n, 3) and np.asarray(tgt.head_xyz).shape == (n, 3)
    if not ok_t or any(ph.end_frame > n for ph in config.foot_phases):
        raise ValueError("invalid club target shapes or foot phase end_frame")
    q0 = np.asarray(initial_q, dtype=np.float64)
    if not np.all(np.isfinite(q0)) or q0.shape not in ((nq,), (n, nq)):
        raise ValueError("initial_q invalid")
    base = np.tile(q0, (n, 1)) if q0.shape == (nq,) else q0.copy()
    locked = lim.q_min == lim.q_max
    base[:, locked] = lim.q_min[locked]
    free = np.tile(~locked, (n, 1)).ravel()
    prob: _Prob = (config, tgt, kinematics, closure_plant)

    def _scatter(xf: F64) -> F64:
        """Scatter free optimization variables into full (N, nq) trajectory."""
        full = base.ravel().copy()
        full[free] = xf
        return full.reshape(base.shape)

    def _res(xf: F64) -> F64:
        """Assemble full least-squares residual vector across all frames."""
        q = _scatter(xf)
        ps = [_eval_frame(q[t], t, prob)[0] for t in range(n)]
        pr, pw = config.tour_prior_q, config.tour_prior_weight
        d_q = pw * (q - pr) if pr is not None and pw > 0 else 1e-5 * (q - base)
        ps.append(d_q.ravel())
        if (sw := config.smoothness_weight) > 0.0 and n >= 3:
            ps.append(sw * (q[2:] - 2.0 * q[1:-1] + q[:-2]).ravel())
        return np.concatenate(ps)

    if np.any(free):
        t, nf, x0 = config.tol, config.max_nfev, base.ravel()[free]
        sol = least_squares(_res, x0, ftol=t, xtol=t, gtol=t, max_nfev=nf)
        q_sol, st = _scatter(sol.x), (bool(sol.success), float(sol.cost), int(sol.nfev))
    else:
        rv = _res(np.empty(0, dtype=np.float64))
        q_sol, st = base, (True, float(0.5 * np.dot(rv, rv)), 1)
    return _audit(q_sol, float(dt_s), prob, st)
