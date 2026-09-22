"""Match club-only observations with driven double and triple pendulums (CO-04 #10608).

Orchestrates TB-04 Bernstein torque fitting against club-only profiles, scores
in-plane versus original 3D residuals separately, warm-starts from CO-03 seeds
when mapping is valid, and retains the best of cold versus retrieval starts.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.engines.physics_engines.pendulum.python.motion_matching.adapters import (
    create_calibrated_double_pendulum_dynamics,
    forward_kinematics_2d,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization import (
    DoublePendulumFitOptions,
    DoublePendulumFitTarget,
    fit_bounded_double_pendulum,
    integrate_double_pendulum_rollout,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization_triple import (
    TriplePendulumFitOptions,
    TriplePendulumFitTarget,
    create_calibrated_triple_pendulum_dynamics,
    fit_bounded_triple_pendulum,
    forward_kinematics_triple_2d,
    integrate_triple_pendulum_rollout,
)
from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_only.acceptance import (
    ClubOnlyResidualReport,
    evaluate_club_only_acceptance,
)
from src.shared.python.motion_matching.club_only.ambiguity import CandidateScore
from src.shared.python.motion_matching.club_only.adapters import (
    observation_to_club_target,
)
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import (
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.seeds import CandidateSeed
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)
from src.shared.python.motion_matching.projection_2d import (
    estimate_swing_plane,
    project_to_calibrated_plane,
)
from src.shared.python.tour_baselines.calibration import (
    calibrate_fixed_geometry,
    compute_moving_hub_power,
    map_initial_state_double_pendulum,
)

MATCH_SCHEMA = "club-pendulum-match/1.0.0"
MODEL_ID_DOUBLE = "driven_double_pendulum"
MODEL_ID_TRIPLE = "driven_triple_pendulum"
_SUPPORTED_MODELS = frozenset({MODEL_ID_DOUBLE, MODEL_ID_TRIPLE})
_GOVERNING_ISSUE = 10608
_HUB_SEGMENT_M = 0.20

__all__ = [
    "MATCH_SCHEMA",
    "MODEL_ID_DOUBLE",
    "MODEL_ID_TRIPLE",
    "PendulumMatchMatrix",
    "PendulumMatchOutcome",
    "build_pendulum_match_matrix",
    "evidence_payload",
    "match_club_only_pendulum",
]


@dataclass(frozen=True)
class PendulumMatchOutcome:
    """One model/trial fit with separated residuals and replay inputs."""

    model_id: str
    trial_id: str
    start_mode: str
    warm_start_applied: bool
    is_moving_hub: bool
    t0_evaluated_before_step: bool
    first_frame_grip_error_m: float
    first_frame_face_error_m: float
    in_plane_grip_rmse_m: float
    in_plane_face_rmse_m: float
    spatial_3d_face_rmse_m: float
    out_of_plane_rmse_m: float
    native_coverage_fraction: float
    external_work_joules: float
    solver_converged: bool
    measured_accepted: bool
    overall_accepted: bool
    limitations: tuple[str, ...]
    replay_package: Mapping[str, Any]
    cold_in_plane_face_rmse_m: float | None = None
    retrieval_in_plane_face_rmse_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trial_id": self.trial_id,
            "start_mode": self.start_mode,
            "warm_start_applied": self.warm_start_applied,
            "is_moving_hub": self.is_moving_hub,
            "t0_evaluated_before_step": self.t0_evaluated_before_step,
            "first_frame_grip_error_m": self.first_frame_grip_error_m,
            "first_frame_face_error_m": self.first_frame_face_error_m,
            "in_plane_grip_rmse_m": self.in_plane_grip_rmse_m,
            "in_plane_face_rmse_m": self.in_plane_face_rmse_m,
            "spatial_3d_face_rmse_m": self.spatial_3d_face_rmse_m,
            "out_of_plane_rmse_m": self.out_of_plane_rmse_m,
            "native_coverage_fraction": self.native_coverage_fraction,
            "external_work_joules": self.external_work_joules,
            "solver_converged": self.solver_converged,
            "measured_accepted": self.measured_accepted,
            "overall_accepted": self.overall_accepted,
            "limitations": list(self.limitations),
            "cold_in_plane_face_rmse_m": self.cold_in_plane_face_rmse_m,
            "retrieval_in_plane_face_rmse_m": self.retrieval_in_plane_face_rmse_m,
            "replay_package_keys": sorted(self.replay_package.keys()),
        }


@dataclass(frozen=True)
class PendulumMatchMatrix:
    """Eight-cell double/triple × four-trial match matrix."""

    schema: str
    outcomes: tuple[PendulumMatchOutcome, ...]
    qualification_blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "governing_issue": _GOVERNING_ISSUE,
            "cell_count": len(self.outcomes),
            "outcomes": [o.as_dict() for o in self.outcomes],
            "qualification_blockers": list(self.qualification_blockers),
        }


def _seed_q_compatible(seed: CandidateSeed | None, model_id: str) -> bool:
    if seed is None:
        return False
    need = 2 if model_id == MODEL_ID_DOUBLE else 3
    q = np.asarray(seed.q, dtype=np.float64)
    # Exact DOF match required — truncating a longer q silently is not a mapping.
    return q.ndim == 1 and q.size == need and bool(np.all(np.isfinite(q)))


def _project_observation(
    obs: ClubObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return plane times, grip_2d, head_2d, and out-of-plane RMSE."""
    club = observation_to_club_target(obs)
    plane = estimate_swing_plane(np.vstack([club.butt, club.clubhead]))
    projected = project_to_calibrated_plane(club, plane)
    grip_uvn = plane.project_points_to_plane(club.butt)
    head_uvn = plane.project_points_to_plane(club.clubhead)
    out_vals = np.concatenate([np.abs(grip_uvn[:, 2]), np.abs(head_uvn[:, 2])])
    out_rmse = float(np.sqrt(np.mean(out_vals**2))) if out_vals.size else 0.0
    return (
        np.asarray(projected.time, dtype=np.float64),
        np.asarray(projected.butt, dtype=np.float64),
        np.asarray(projected.clubhead, dtype=np.float64),
        out_rmse,
    )


def _calibrate_double(
    times: np.ndarray, grip: np.ndarray, head: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    pivot = np.zeros(3)
    pivots = np.tile(pivot, (len(times), 1))
    try:
        geom = calibrate_fixed_geometry(pivots, grip, head)
        l1, l2 = geom.l1_arm_m, geom.l2_club_m
    except (ValueError, RuntimeError, ZeroDivisionError):
        l1, l2 = 0.65, 1.05
    init = map_initial_state_double_pendulum(times, pivots, grip, head, l1, l2)
    return l1, l2, init.q0, init.v0


def _calibrate_triple(
    times: np.ndarray, grip: np.ndarray, head: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    l1 = _HUB_SEGMENT_M
    # Shoulder tip at rest hangs at [0, -l1]; solve remaining two links like double.
    shoulder = np.tile(np.array([0.0, -l1, 0.0]), (len(times), 1))
    try:
        geom = calibrate_fixed_geometry(shoulder, grip, head)
        l2, l3 = geom.l1_arm_m, geom.l2_club_m
    except (ValueError, RuntimeError, ZeroDivisionError):
        l2, l3 = 0.55, 1.05
    # theta1 ≈ 0 at address; map arm/club relative to hanging hub tip.
    init = map_initial_state_double_pendulum(times, shoulder, grip, head, l2, l3)
    q0 = np.array([0.0, float(init.q0[0]), float(init.q0[1])], dtype=np.float64)
    v0 = np.array([0.0, float(init.v0[0]), float(init.v0[1])], dtype=np.float64)
    return l1, l2, l3, q0, v0


def _coverage(grip: np.ndarray, head: np.ndarray) -> float:
    valid = np.isfinite(grip).all(axis=1) & np.isfinite(head).all(axis=1)
    return float(np.mean(valid)) if len(valid) else 0.0


def _rmse_series(pred: np.ndarray, meas: np.ndarray) -> float:
    mask = np.isfinite(pred).all(axis=1) & np.isfinite(meas).all(axis=1)
    if not np.any(mask):
        return float("inf")
    err = np.linalg.norm(pred[mask, :2] - meas[mask, :2], axis=1)
    return float(np.sqrt(np.mean(err**2)))


def _first_frame_errors(
    pred_grip: np.ndarray, pred_head: np.ndarray, grip: np.ndarray, head: np.ndarray
) -> tuple[float, float]:
    g = float(np.linalg.norm(pred_grip[:2] - grip[0, :2]))
    h = float(np.linalg.norm(pred_head[:2] - head[0, :2]))
    return g, h


def _fit_double(
    times: np.ndarray,
    grip: np.ndarray,
    head: np.ndarray,
    *,
    q0: np.ndarray,
    v0: np.ndarray,
    l1: float,
    l2: float,
    max_nfev: int,
    initial_controls: np.ndarray | None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dynamics = create_calibrated_double_pendulum_dynamics(l1, l2)
    target = DoublePendulumFitTarget(
        times=times, grip=grip, head=head, l1=l1, l2=l2, q0=q0, v0=v0
    )
    fit = fit_bounded_double_pendulum(
        target,
        dynamics,
        DoublePendulumFitOptions(max_nfev=max_nfev, initial_controls=initial_controls),
    )
    # Independent tighter-step replay (4x).
    integrate_double_pendulum_rollout(dynamics, q0, v0, times, fit.profile, substeps=4)
    pred_grip = np.zeros((len(times), 2))
    pred_head = np.zeros((len(times), 2))
    for i, q in enumerate(fit.q_traj):
        g, h = forward_kinematics_2d(float(q[0]), float(q[1]), l1, l2)
        pred_grip[i] = g
        pred_head[i] = h
    return fit, pred_grip, pred_head, fit.q_traj, fit.v_traj


def _fit_triple(
    times: np.ndarray,
    grip: np.ndarray,
    head: np.ndarray,
    *,
    q0: np.ndarray,
    v0: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
    max_nfev: int,
    initial_controls: np.ndarray | None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dynamics = create_calibrated_triple_pendulum_dynamics(l1, l2, l3)
    target = TriplePendulumFitTarget(
        times=times,
        grip=grip,
        head=head,
        l1=l1,
        l2=l2,
        l3=l3,
        q0=q0,
        v0=v0,
    )
    fit = fit_bounded_triple_pendulum(
        target,
        dynamics,
        TriplePendulumFitOptions(max_nfev=max_nfev, initial_controls=initial_controls),
    )
    integrate_triple_pendulum_rollout(dynamics, q0, v0, times, fit.profile, substeps=4)
    pred_grip = np.zeros((len(times), 2))
    pred_head = np.zeros((len(times), 2))
    for i, q in enumerate(fit.q_traj):
        _, g, h = forward_kinematics_triple_2d(
            float(q[0]), float(q[1]), float(q[2]), l1, l2, l3
        )
        pred_grip[i] = g
        pred_head[i] = h
    return fit, pred_grip, pred_head, fit.q_traj, fit.v_traj, fit.hub_path


def _build_outcome(
    *,
    model_id: str,
    trial_id: str,
    obs: ClubObservation,
    start_mode: str,
    warm_start_applied: bool,
    fit: Any,
    pred_grip: np.ndarray,
    pred_head: np.ndarray,
    grip: np.ndarray,
    head: np.ndarray,
    out_of_plane_rmse: float,
    times: np.ndarray,
    q0: np.ndarray,
    v0: np.ndarray,
    lengths: tuple[float, ...],
    hub_path: np.ndarray | None,
    cold_rmse: float | None,
    retrieval_rmse: float | None,
) -> PendulumMatchOutcome:
    is_moving = model_id == MODEL_ID_TRIPLE
    ff_g, ff_h = _first_frame_errors(pred_grip[0], pred_head[0], grip, head)
    in_grip = _rmse_series(pred_grip, grip)
    in_face = _rmse_series(pred_head, head)
    # Spatial 3D face RMSE: embed in-plane head error with out-of-plane residual.
    spatial = float(math.sqrt(in_face**2 + out_of_plane_rmse**2))
    coverage = _coverage(grip, head)
    external_work = 0.0
    if is_moving and hub_path is not None and len(hub_path) >= 2:
        forces = np.zeros_like(hub_path)
        hub_motion = compute_moving_hub_power(times, hub_path, forces)
        external_work = float(hub_motion.total_work_joules)

    profile = get_club_only_profile(model_id)
    residuals = ClubOnlyResidualReport(
        grip_position_rmse_m=in_grip,
        face_position_rmse_m=in_face,
        grip_orientation_rmse_rad=None,
        face_orientation_rmse_rad=None,
        native_coverage_fraction=coverage,
        speed_error_m_s=None,
        phase_error_s=None,
        unweighted_physical={"closure_residual_m": ff_g, "penetration_m": 0.0},
    )
    candidates = (
        CandidateScore(
            candidate_id=f"{model_id}:{trial_id}:{start_mode}",
            measured_residual_m=in_face,
            prior_score=0.5,
            body_configuration_hash=hashlib.sha256(
                np.asarray(q0, dtype=np.float64).tobytes()
            ).hexdigest()[:16],
            closure_residual_m=ff_g,
            contact_feasible=True,
            claims_force_measurement=False,
        ),
    )
    verdict = evaluate_club_only_acceptance(
        residuals,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=True,
        torque_replay_validated=False,
        kinematic_preview_ok=True,
    )
    torque_controls = (
        np.concatenate([fit.profile.shoulder_controls, fit.profile.wrist_controls])
        if model_id == MODEL_ID_DOUBLE
        else fit.profile.as_controls()
    )
    content = hashlib.sha256()
    content.update(np.asarray(torque_controls, dtype=np.float64).tobytes())
    content.update(np.asarray(q0, dtype=np.float64).tobytes())
    replay = {
        "model_id": model_id,
        "trial_id": trial_id,
        "schema": MATCH_SCHEMA,
        "q0": np.asarray(q0, dtype=np.float64).tolist(),
        "v0": np.asarray(v0, dtype=np.float64).tolist(),
        "lengths_m": list(lengths),
        "torque_controls": np.asarray(torque_controls, dtype=np.float64).tolist(),
        "times_s": np.asarray(times, dtype=np.float64).tolist(),
        "t0_evaluated_before_step": True,
        "package_hash": content.hexdigest()[:16],
        "is_moving_hub": is_moving,
        "external_work_joules": external_work,
    }
    limitations = list(verdict.limitations) + list(profile.limitations)
    if not verdict.measured_accepted:
        limitations.append("measured_club_gates_unmet_on_synthetic_fixture")
    return PendulumMatchOutcome(
        model_id=model_id,
        trial_id=trial_id,
        start_mode=start_mode,
        warm_start_applied=warm_start_applied,
        is_moving_hub=is_moving,
        t0_evaluated_before_step=True,
        first_frame_grip_error_m=ff_g,
        first_frame_face_error_m=ff_h,
        in_plane_grip_rmse_m=in_grip,
        in_plane_face_rmse_m=in_face,
        spatial_3d_face_rmse_m=spatial,
        out_of_plane_rmse_m=out_of_plane_rmse,
        native_coverage_fraction=coverage,
        external_work_joules=external_work,
        solver_converged=bool(fit.converged),
        measured_accepted=bool(verdict.measured_accepted),
        overall_accepted=bool(verdict.overall_accepted),
        limitations=tuple(dict.fromkeys(limitations)),
        replay_package=replay,
        cold_in_plane_face_rmse_m=cold_rmse,
        retrieval_in_plane_face_rmse_m=retrieval_rmse,
    )


@precondition(
    lambda observation, model_id, **_: isinstance(observation, ClubObservation),
    "observation must be ClubObservation",
)
@postcondition(
    lambda result: isinstance(result, PendulumMatchOutcome),
    "must return PendulumMatchOutcome",
)
def match_club_only_pendulum(
    *,
    observation: ClubObservation,
    model_id: str,
    seed: CandidateSeed | None = None,
    max_nfev: int = 40,
    compare_cold_and_retrieval: bool = False,
) -> PendulumMatchOutcome:
    """Fit one driven pendulum model to a club-only observation."""
    if model_id not in _SUPPORTED_MODELS:
        raise ValueError(f"unsupported model_id for CO-04 pendulum match: {model_id!r}")
    if max_nfev < 1:
        raise ValueError("max_nfev must be >= 1")

    times, grip, head, out_rmse = _project_observation(observation)
    trial_id = observation.trial_id
    warm_ok = _seed_q_compatible(seed, model_id)

    if model_id == MODEL_ID_DOUBLE:
        l1, l2, q0_cold, v0_cold = _calibrate_double(times, grip, head)
        lengths: tuple[float, ...] = (l1, l2)
        q0_warm = (
            np.asarray(seed.q[:2], dtype=np.float64) if warm_ok and seed else q0_cold
        )
        v0_warm = v0_cold
    else:
        l1, l2, l3, q0_cold, v0_cold = _calibrate_triple(times, grip, head)
        lengths = (l1, l2, l3)
        q0_warm = (
            np.asarray(seed.q[:3], dtype=np.float64) if warm_ok and seed else q0_cold
        )
        v0_warm = v0_cold

    def _run(
        q0: np.ndarray, v0: np.ndarray
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray | None]:
        if model_id == MODEL_ID_DOUBLE:
            fit, pg, ph, _, _ = _fit_double(
                times,
                grip,
                head,
                q0=q0,
                v0=v0,
                l1=lengths[0],
                l2=lengths[1],
                max_nfev=max_nfev,
                initial_controls=None,
            )
            return fit, pg, ph, None
        fit, pg, ph, _, _, hub = _fit_triple(
            times,
            grip,
            head,
            q0=q0,
            v0=v0,
            l1=lengths[0],
            l2=lengths[1],
            l3=lengths[2],
            max_nfev=max_nfev,
            initial_controls=None,
        )
        return fit, pg, ph, hub

    cold_fit, cold_g, cold_h, cold_hub = _run(q0_cold, v0_cold)
    cold_rmse = _rmse_series(cold_h, head)

    if compare_cold_and_retrieval and warm_ok:
        ret_fit, ret_g, ret_h, ret_hub = _run(q0_warm, v0_warm)
        ret_rmse = _rmse_series(ret_h, head)
        if ret_rmse <= cold_rmse:
            chosen = (
                ret_fit,
                ret_g,
                ret_h,
                ret_hub,
                q0_warm,
                v0_warm,
                "retrieval",
                True,
            )
        else:
            chosen = (
                cold_fit,
                cold_g,
                cold_h,
                cold_hub,
                q0_cold,
                v0_cold,
                "cold",
                False,
            )
        fit, pg, ph, hub, q0, v0, mode, warm_applied = chosen
        return _build_outcome(
            model_id=model_id,
            trial_id=trial_id,
            obs=observation,
            start_mode=mode,
            warm_start_applied=warm_applied,
            fit=fit,
            pred_grip=pg,
            pred_head=ph,
            grip=grip,
            head=head,
            out_of_plane_rmse=out_rmse,
            times=times,
            q0=q0,
            v0=v0,
            lengths=lengths,
            hub_path=hub,
            cold_rmse=cold_rmse,
            retrieval_rmse=ret_rmse,
        )

    if warm_ok and seed is not None and not compare_cold_and_retrieval:
        fit, pg, ph, hub = _run(q0_warm, v0_warm)
        return _build_outcome(
            model_id=model_id,
            trial_id=trial_id,
            obs=observation,
            start_mode="retrieval",
            warm_start_applied=True,
            fit=fit,
            pred_grip=pg,
            pred_head=ph,
            grip=grip,
            head=head,
            out_of_plane_rmse=out_rmse,
            times=times,
            q0=q0_warm,
            v0=v0_warm,
            lengths=lengths,
            hub_path=hub,
            cold_rmse=None,
            retrieval_rmse=None,
        )

    return _build_outcome(
        model_id=model_id,
        trial_id=trial_id,
        obs=observation,
        start_mode="cold",
        warm_start_applied=False,
        fit=cold_fit,
        pred_grip=cold_g,
        pred_head=cold_h,
        grip=grip,
        head=head,
        out_of_plane_rmse=out_rmse,
        times=times,
        q0=q0_cold,
        v0=v0_cold,
        lengths=lengths,
        hub_path=cold_hub,
        cold_rmse=None,
        retrieval_rmse=None,
    )


def build_pendulum_match_matrix(*, max_nfev: int = 20) -> PendulumMatchMatrix:
    """Fit driven double and triple pendulums across all four canonical trials."""
    outcomes: list[PendulumMatchOutcome] = []
    for model_id in (MODEL_ID_DOUBLE, MODEL_ID_TRIPLE):
        for trial_id in CANONICAL_TRIAL_SHEETS:
            obs = build_calibrated_observation_fixture(trial_id)
            outcomes.append(
                match_club_only_pendulum(
                    observation=obs, model_id=model_id, max_nfev=max_nfev
                )
            )
    blockers: list[str] = []
    for outcome in outcomes:
        if not outcome.measured_accepted:
            blockers.append(
                f"{outcome.model_id}/{outcome.trial_id}: measured gates unmet"
            )
        if not outcome.overall_accepted:
            blockers.append(
                f"{outcome.model_id}/{outcome.trial_id}: overall acceptance unmet"
            )
    blockers.append(
        "synthetic_fixtures_only: native Club_Data.xlsx qualification deferred to CO-08"
    )
    # Deduplicate while preserving order.
    blockers = list(dict.fromkeys(blockers))
    return PendulumMatchMatrix(
        schema=MATCH_SCHEMA,
        outcomes=tuple(outcomes),
        qualification_blockers=tuple(blockers),
    )


def evidence_payload(matrix: PendulumMatchMatrix) -> dict[str, Any]:
    """Versioned evidence document for the eight-cell pendulum match matrix."""
    payload = matrix.as_dict()
    payload["note"] = (
        "Software-contract matrix on calibrated synthetic fixtures; "
        "not native physical qualification evidence."
    )
    return payload
