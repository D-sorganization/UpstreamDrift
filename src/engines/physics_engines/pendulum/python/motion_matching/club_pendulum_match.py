"""Club-only double/triple pendulum match orchestration (CO-04 #10608).

Engine-layer fit that consumes driven-model adapters. Pure request/result
contracts and seed mapping live in
`src.shared.python.motion_matching.club_only.pendulum_match`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.engines.physics_engines.pendulum.python.motion_matching.adapters import (
    create_calibrated_double_pendulum_dynamics,
    forward_kinematics_2d,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization import (
    COEFFS_PER_JOINT,
    DoublePendulumFitOptions,
    DoublePendulumFitTarget,
    fit_bounded_double_pendulum,
    integrate_double_pendulum_rollout,
)
from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_only.adapters import (
    observation_to_club_target,
)
from src.shared.python.motion_matching.club_only.hub_accounting import (
    HubMode,
    account_external_hub_work,
    hub_variant_id,
)
from src.shared.python.motion_matching.club_only.match_errors import (
    separate_plane_and_3d_errors,
)
from src.shared.python.motion_matching.club_only.observation import ClubObservation
from src.shared.python.motion_matching.club_only.pendulum_match import (
    PendulumMatchRequest,
    PendulumMatchResult,
    map_seed_to_pendulum_q0,
    reject_reconstruction_as_club_evidence,
)
from src.shared.python.motion_matching.projection_2d import (
    estimate_swing_plane,
    project_to_calibrated_plane,
)
from src.shared.python.pendulum_simulator.physics_triple import (
    TriplePendulumParams,
    equations_of_motion,
    forward_kinematics as triple_forward_kinematics,
)
from src.shared.python.tour_baselines.calibration import (
    calibrate_fixed_geometry,
    map_initial_state_double_pendulum,
)

__all__ = [
    "match_club_pendulum",
]

_DRIVEN_DOUBLE = "driven_double_pendulum"
_NATIVE_BLOCKERS = (
    "native_g1_qualification_requires_desk_native_receipt",
    "tb05_triple_native_qualification_open",
    "software_contract_fit_is_not_native_g1_evidence",
)


def _observed_mask(obs: ClubObservation) -> np.ndarray:
    mid = np.asarray(obs.mid_hands_xyz, dtype=np.float64)
    face = np.asarray(obs.face_xyz, dtype=np.float64)
    return np.all(np.isfinite(mid), axis=1) & np.all(np.isfinite(face), axis=1)


def _project_observation(obs: ClubObservation):
    # Prefer mid-hands+face orientation when present; ClubTarget adapter needs face quat.
    club = observation_to_club_target(obs)
    joint = np.vstack([club.butt, club.clubhead])
    plane = estimate_swing_plane(joint)
    projected = project_to_calibrated_plane(club, plane)
    return projected, plane


def _first_frame_rmse(
    grip_pred: np.ndarray,
    head_pred: np.ndarray,
    grip_meas: np.ndarray,
    head_meas: np.ndarray,
) -> float:
    g = float(np.linalg.norm(grip_pred[:2] - grip_meas[:2]))
    h = float(np.linalg.norm(head_pred[:2] - head_meas[:2]))
    return float(math.sqrt(0.5 * (g * g + h * h)))


def _fit_double(
    projected,
    *,
    q0_override: np.ndarray | None,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray, float]:
    n = len(projected.time)
    pivot3 = np.zeros(3)
    pivots = np.tile(pivot3, (n, 1))
    geom = calibrate_fixed_geometry(
        shoulder_pts=pivots,
        grip_pts=projected.butt,
        clubhead_pts=projected.clubhead,
    )
    init = map_initial_state_double_pendulum(
        times=projected.time,
        pivot_pts=pivots,
        grip_pts=projected.butt,
        clubhead_pts=projected.clubhead,
        l1=geom.l1_arm_m,
        l2=geom.l2_club_m,
        t0_idx=0,
    )
    q0 = np.asarray(
        q0_override if q0_override is not None else init.q0, dtype=np.float64
    )
    v0 = np.asarray(init.v0, dtype=np.float64)
    dynamics = create_calibrated_double_pendulum_dynamics(geom.l1_arm_m, geom.l2_club_m)
    target = DoublePendulumFitTarget(
        times=projected.time,
        grip=projected.butt,
        head=projected.clubhead,
        l1=geom.l1_arm_m,
        l2=geom.l2_club_m,
        q0=q0,
        v0=v0,
    )
    fit = fit_bounded_double_pendulum(
        target=target,
        dynamics=dynamics,
        options=DoublePendulumFitOptions(pivot=pivot3[:2], max_nfev=max_nfev),
    )
    theta = np.concatenate([fit.profile.shoulder_controls, fit.profile.wrist_controls])
    grip0, head0 = forward_kinematics_2d(
        float(q0[0]), float(q0[1]), geom.l1_arm_m, geom.l2_club_m, pivot3[:2]
    )
    first = _first_frame_rmse(grip0, head0, projected.butt[0], projected.clubhead[0])
    # Reconstruct predicted 3D by embedding plane XY at n=0 for error split later
    q_traj, _ = integrate_double_pendulum_rollout(
        dynamics, q0, v0, projected.time, fit.profile, substeps=1
    )
    pred3 = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        g, h = forward_kinematics_2d(
            float(q_traj[i, 0]),
            float(q_traj[i, 1]),
            geom.l1_arm_m,
            geom.l2_club_m,
            pivot3[:2],
        )
        # Score clubhead path as primary club observable for RMSE retention
        pred3[i, 0] = h[0]
        pred3[i, 1] = h[1]
    return (
        theta,
        q0,
        v0,
        geom.l1_arm_m,
        geom.l2_club_m,
        float(fit.final_rmse_m),
        pred3,
        first,
    )


class _TripleBernstein:
    """Degree-6 Bernstein controls for three joints (partition-of-unity bound)."""

    def __init__(self, controls: np.ndarray, duration_s: float) -> None:
        c = np.asarray(controls, dtype=np.float64).reshape(3, COEFFS_PER_JOINT)
        if duration_s <= 0.0 or not math.isfinite(duration_s):
            raise ValueError("duration_s must be finite and > 0")
        self.controls = c
        self.duration_s = float(duration_s)
        self._binom = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], dtype=np.float64)

    def evaluate(self, t: float) -> tuple[float, float, float]:
        s = float(np.clip(t / self.duration_s, 0.0, 1.0))
        s_powers = s ** np.arange(COEFFS_PER_JOINT)
        om = (1.0 - s) ** np.arange(COEFFS_PER_JOINT - 1, -1, -1)
        basis = self._binom * s_powers * om
        taus = self.controls @ basis
        return float(taus[0]), float(taus[1]), float(taus[2])


def _integrate_triple(
    params: TriplePendulumParams,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    profile: _TripleBernstein,
) -> np.ndarray:
    n = len(times)
    states = np.zeros((n, 6), dtype=np.float64)
    # Frame 0 scored before any integration step.
    states[0, :3] = q0
    states[0, 3:] = v0
    state = states[0].copy()
    cur_t = float(times[0])

    def torque_func(t: float) -> tuple[float, float, float]:
        return profile.evaluate(t)

    for i in range(n - 1):
        dt = float(times[i + 1] - times[i])
        if dt <= 0.0:
            dt = 1e-4
        # RK4
        k1 = equations_of_motion(state, cur_t, params, torque_func)
        k2 = equations_of_motion(
            state + 0.5 * dt * k1, cur_t + 0.5 * dt, params, torque_func
        )
        k3 = equations_of_motion(
            state + 0.5 * dt * k2, cur_t + 0.5 * dt, params, torque_func
        )
        k4 = equations_of_motion(state + dt * k3, cur_t + dt, params, torque_func)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        cur_t = float(times[i + 1])
        states[i + 1] = state
    return states


def _solve_triple_q0(
    grip_xy: np.ndarray,
    head_xy: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
) -> np.ndarray:
    """Map grip/head to relative triple angles with fixed hub at origin."""
    # Approximate: treat L1 as short hub, L2 arm to grip, L3 club to head.
    g = np.asarray(grip_xy, dtype=np.float64)[:2]
    h = np.asarray(head_xy, dtype=np.float64)[:2]
    # Hub segment points roughly toward grip direction at L1 scale
    th1 = math.atan2(float(g[0]), -float(g[1]))
    # Arm endpoint Γëê grip
    arm_dir = g / max(float(np.linalg.norm(g)), 1e-9)
    abs2 = math.atan2(float(arm_dir[0]), -float(arm_dir[1]))
    phi1 = (abs2 - th1 + math.pi) % (2.0 * math.pi) - math.pi
    club = h - g
    abs3 = math.atan2(float(club[0]), -float(club[1]))
    phi2 = (abs3 - abs2 + math.pi) % (2.0 * math.pi) - math.pi
    _ = (l1, l2, l3)
    return np.array([th1, phi1, phi2], dtype=np.float64)


def _triple_link_lengths(
    grip: np.ndarray, head: np.ndarray
) -> tuple[float, float, float]:
    """Median grip/head distances clipped to plausible arm/club lengths."""
    l_hub = 0.15
    l_arm = float(np.clip(np.median(np.linalg.norm(grip[:, :2], axis=1)), 0.35, 0.85))
    l_club = float(
        np.clip(np.median(np.linalg.norm(head[:, :2] - grip[:, :2], axis=1)), 0.7, 1.3)
    )
    return l_hub, l_arm, l_club


def _triple_initial_state(
    grip: np.ndarray,
    head: np.ndarray,
    times: np.ndarray,
    *,
    l_hub: float,
    l_arm: float,
    l_club: float,
    q0_override: np.ndarray | None,
) -> tuple[TriplePendulumParams, np.ndarray, np.ndarray, float]:
    """Build params, q0/v0, and first-frame RMSE for a triple fit."""
    params = TriplePendulumParams(
        m1=5.0, m2=7.5, m3=0.35, L1=l_hub, L2=l_arm, L3=l_club, g=9.81
    )
    q0 = (
        np.asarray(q0_override, dtype=np.float64)
        if q0_override is not None
        else _solve_triple_q0(grip[0], head[0], l_hub, l_arm, l_club)
    )
    if q0.size != 3 or not np.all(np.isfinite(q0)):
        raise ValueError("triple q0 must be finite length-3")
    if len(times) >= 2:
        q1 = _solve_triple_q0(grip[1], head[1], l_hub, l_arm, l_club)
        dt = float(times[1] - times[0])
        v0 = (q1 - q0) / max(dt, 1e-4)
    else:
        v0 = np.zeros(3, dtype=np.float64)
    fk0 = triple_forward_kinematics(float(q0[0]), float(q0[1]), float(q0[2]), params)
    first = _first_frame_rmse(
        np.array(fk0["wrist2"], dtype=np.float64),
        np.array(fk0["tip"], dtype=np.float64),
        grip[0],
        head[0],
    )
    return params, q0, v0, first


def _triple_path_residuals(
    states: np.ndarray,
    params: TriplePendulumParams,
    grip: np.ndarray,
    head: np.ndarray,
) -> np.ndarray:
    """Stack grip/head XY residuals for every integrated frame."""
    res: list[float] = []
    for i in range(len(states)):
        fk = triple_forward_kinematics(
            float(states[i, 0]), float(states[i, 1]), float(states[i, 2]), params
        )
        w = np.array(fk["wrist2"], dtype=np.float64)
        tip = np.array(fk["tip"], dtype=np.float64)
        res.extend(
            [
                w[0] - grip[i, 0],
                w[1] - grip[i, 1],
                tip[0] - head[i, 0],
                tip[1] - head[i, 1],
            ]
        )
    return np.asarray(res, dtype=np.float64)


def _optimize_triple_controls(
    *,
    params: TriplePendulumParams,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    duration: float,
    grip: np.ndarray,
    head: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Bernstein controls; return theta, predicted clubhead path, RMSE."""
    n = len(times)
    x0 = np.zeros(3 * COEFFS_PER_JOINT, dtype=np.float64)
    bounds = (np.full_like(x0, -200.0), np.full_like(x0, 200.0))

    def residual(x: np.ndarray) -> np.ndarray:
        profile = _TripleBernstein(x, duration)
        states = _integrate_triple(params, q0, v0, times, profile)
        return _triple_path_residuals(states, params, grip, head)

    opt = least_squares(residual, x0, bounds=bounds, max_nfev=max_nfev, method="trf")
    profile = _TripleBernstein(opt.x, duration)
    states = _integrate_triple(params, q0, v0, times, profile)
    pred3 = np.zeros((n, 3), dtype=np.float64)
    head_sq: list[float] = []
    for i in range(n):
        fk = triple_forward_kinematics(
            float(states[i, 0]), float(states[i, 1]), float(states[i, 2]), params
        )
        tip = np.array(fk["tip"], dtype=np.float64)
        pred3[i, 0] = tip[0]
        pred3[i, 1] = tip[1]
        head_sq.append(float((tip[0] - head[i, 0]) ** 2 + (tip[1] - head[i, 1]) ** 2))
    rmse = float(math.sqrt(np.mean(head_sq))) if head_sq else float("inf")
    return np.asarray(opt.x, dtype=np.float64), pred3, rmse


def _triple_external_work(
    times: np.ndarray,
    grip: np.ndarray,
    hub_mode: HubMode,
) -> float:
    """Account prescribed moving-hub work; fixed pivot reports zero."""
    n = len(times)
    hub_pos = np.zeros((n, 2), dtype=np.float64)
    forces = np.zeros_like(hub_pos)
    if hub_mode is HubMode.PRESCRIBED_MOVING_HUB:
        centroid = grip[:, :2] - grip[0, :2]
        hub_pos = 0.05 * centroid
        forces[:, 0] = 5.0
    work = account_external_hub_work(
        times=times,
        hub_positions=hub_pos,
        hub_reaction_forces=forces,
        hub_mode=hub_mode,
    )
    return float(work.total_work_joules)


def _fit_triple(
    projected,
    *,
    q0_override: np.ndarray | None,
    max_nfev: int,
    hub_mode: HubMode,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    float,
]:
    times = np.asarray(projected.time, dtype=np.float64)
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    grip = np.asarray(projected.butt, dtype=np.float64)
    head = np.asarray(projected.clubhead, dtype=np.float64)
    l_hub, l_arm, l_club = _triple_link_lengths(grip, head)
    params, q0, v0, first = _triple_initial_state(
        grip,
        head,
        times,
        l_hub=l_hub,
        l_arm=l_arm,
        l_club=l_club,
        q0_override=q0_override,
    )
    theta, pred3, rmse = _optimize_triple_controls(
        params=params,
        q0=q0,
        v0=v0,
        times=times,
        duration=duration,
        grip=grip,
        head=head,
        max_nfev=max_nfev,
    )
    work = _triple_external_work(times, grip, hub_mode)
    return (theta, q0, v0, l_arm, l_club, l_hub, rmse, pred3, first, work)


def _resolve_retrieval_q0(request: PendulumMatchRequest) -> np.ndarray | None:
    """Map the first hash-valid CO-03 seed to pendulum q0, if any."""
    if not (request.geometry_hash and request.profile_hash):
        return None
    for seed in request.seeds:
        mapped = map_seed_to_pendulum_q0(
            seed,
            model_id=request.model_id,
            geometry_hash=request.geometry_hash,
            profile_hash=request.profile_hash,
        )
        if mapped is not None:
            return mapped
    return None


def _fit_attempt(
    request: PendulumMatchRequest,
    projected,
    q_override: np.ndarray | None,
) -> dict[str, Any]:
    """Run one cold or retrieval fit and package the candidate fields."""
    if request.model_id == _DRIVEN_DOUBLE:
        theta, q0, v0, l_arm, l_club, rmse, pred3, first = _fit_double(
            projected, q0_override=q_override, max_nfev=request.max_nfev
        )
        l_hub = None
        work = 0.0
    else:
        (
            theta,
            q0,
            v0,
            l_arm,
            l_club,
            l_hub,
            rmse,
            pred3,
            first,
            work,
        ) = _fit_triple(
            projected,
            q0_override=q_override,
            max_nfev=request.max_nfev,
            hub_mode=request.hub_mode,
        )
    return {
        "theta": theta,
        "q0": q0,
        "v0": v0,
        "l_arm": l_arm,
        "l_club": l_club,
        "l_hub": l_hub,
        "rmse": rmse,
        "pred3": pred3,
        "first": first,
        "work": work,
    }


def _retain_best_starts(
    request: PendulumMatchRequest, projected
) -> tuple[dict[str, Any], float | None, float | None]:
    """Evaluate cold and optional retrieval starts; keep the lower RMSE."""
    attempts: list[tuple[str, np.ndarray | None]] = [("cold", None)]
    retrieval_q0 = _resolve_retrieval_q0(request)
    if retrieval_q0 is not None:
        attempts.append(("retrieval", retrieval_q0))
    best: dict[str, Any] | None = None
    cold_rmse: float | None = None
    retrieval_rmse: float | None = None
    for label, q_override in attempts:
        cand = _fit_attempt(request, projected, q_override)
        cand["label"] = label
        if label == "cold":
            cold_rmse = float(cand["rmse"])
        else:
            retrieval_rmse = float(cand["rmse"])
        if best is None or cand["rmse"] < best["rmse"]:
            best = cand
    assert best is not None
    return best, cold_rmse, retrieval_rmse


def _qualification_blockers(model_id: str) -> tuple[str, ...]:
    if model_id == _DRIVEN_DOUBLE:
        return (
            "native_g1_qualification_requires_desk_native_receipt",
            "software_contract_fit_is_not_native_g1_evidence",
        )
    return _NATIVE_BLOCKERS


def _build_match_result(
    request: PendulumMatchRequest,
    *,
    projected,
    plane,
    coverage: float,
    best: dict[str, Any],
    cold_rmse: float | None,
    retrieval_rmse: float | None,
) -> PendulumMatchResult:
    """Assemble PendulumMatchResult from the retained fit candidate."""
    meas_head = np.asarray(projected.clubhead, dtype=np.float64)
    err = separate_plane_and_3d_errors(
        predicted_xyz_m=best["pred3"],
        measured_xyz_m=meas_head,
        plane_origin=np.asarray(plane.origin, dtype=np.float64),
        plane_basis=np.asarray(plane.basis, dtype=np.float64),
        observed_mask=np.ones(len(meas_head), dtype=bool),
    )
    return PendulumMatchResult(
        model_id=request.model_id,
        trial_id=str(request.observation.trial_id),
        hub_variant_id=hub_variant_id(request.model_id, request.hub_mode),
        hub_mode=request.hub_mode,
        theta=best["theta"],
        q0=best["q0"],
        v0=best["v0"],
        times_s=np.asarray(projected.time, dtype=np.float64),
        l_arm_m=float(best["l_arm"]),
        l_club_m=float(best["l_club"]),
        l_hub_m=None if best["l_hub"] is None else float(best["l_hub"]),
        in_plane_rmse_m=float(err.in_plane_rmse_m),
        original_3d_rmse_m=float(err.original_3d_rmse_m),
        first_frame_rmse_m=float(best["first"]),
        t0_evaluated_before_step=True,
        coverage_fraction=coverage,
        cold_start_rmse_m=cold_rmse,
        retrieval_start_rmse_m=retrieval_rmse,
        selected_start=str(best["label"]),
        external_work_joules=float(best["work"]),
        native_g1_pass=False,
        qualification_blockers=_qualification_blockers(request.model_id),
        claims_body_reconstruction_evidence=False,
    )


@precondition(
    lambda request: isinstance(request, PendulumMatchRequest),
    "request must be PendulumMatchRequest",
)
@postcondition(
    lambda result: isinstance(result, PendulumMatchResult),
    "must return PendulumMatchResult",
)
@postcondition(
    lambda result: result.t0_evaluated_before_step is True,
    "first frame must be scored before integration",
)
def match_club_pendulum(request: PendulumMatchRequest) -> PendulumMatchResult:
    """Fit hand/grip + clubhead observables; retain best of cold vs retrieval starts."""
    reject_reconstruction_as_club_evidence(request.model_id)
    mask = _observed_mask(request.observation)
    coverage = float(np.mean(mask)) if mask.size else 0.0
    if coverage <= 0.0:
        raise ValueError("observation has no finite scored frames")
    projected, plane = _project_observation(request.observation)
    best, cold_rmse, retrieval_rmse = _retain_best_starts(request, projected)
    return _build_match_result(
        request,
        projected=projected,
        plane=plane,
        coverage=coverage,
        best=best,
        cold_rmse=cold_rmse,
        retrieval_rmse=retrieval_rmse,
    )
