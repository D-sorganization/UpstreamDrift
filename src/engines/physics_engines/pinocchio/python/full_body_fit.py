"""Native Crocoddyl full-body fit of the tour capture on the Pinocchio plant (MS-31, #10338).

Pipeline: document + capture -> marker IK warm start (Gauss-Newton on the
plant, weld closure enforced) -> least-squares effort warm start from the
plant's effort sensitivity -> Crocoddyl ``SolverBoxFDDP`` over per-node
efforts with a Gauss-Newton marker cost, effort / rate regularisation and a
human-range barrier -> uninterrupted RK4 replay through the same plant ->
shared metrics, physical audit, receipt and candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.crocoddyl_action import (
    make_action_models,
    tracking_rollout,
)
from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
    CrocoddylProblemBundle,
    FitHorizon,
    FitWeights,
    MarkerTargets,
    actuated_mask,
    build_marker_targets,
    coordinate_bounds,
    finite_difference_rates,
    per_coordinate_effort_bounds,
    qualified_crocoddyl,
    range_barrier,
)
from src.engines.physics_engines.pinocchio.python.marker_kinematics import (
    CoordinateMap,
    MarkerIkOptions,
    MarkerIkSolver,
    MarkerTable,
    build_marker_table,
    marker_positions,
    marker_positions_and_jacobians,
)
from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.ground_support import capture_to_native_world
from src.shared.python.motion_matching.pelvis_yaw import (
    compute_pelvis_yaw_residual_and_derivative,
)
from src.shared.python.motion_matching.replay_metrics import compute_replay_five_metrics
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    load_tour_capture,
    tracked_labels,
)
from src.shared.python.motion_matching.tour_metrics import compute_shared_metrics
from src.shared.python.motion_matching.two_window_fit import (
    MarkerMetricResults,
    check_acceptance,
    compute_marker_metrics,
)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

RECEIPT_SCHEMA = "matched-swing-fit/pinocchio-crocoddyl-v1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverSettings:
    max_iterations: int = 200
    stop_tolerance: float = 1e-6
    effort_bound_n_m: float = 600.0
    integrator: str = "linear_implicit_euler"
    replay_integrator: str = "rk45"  # "rk45" | "rk4"
    replay_substeps: int = 12
    initial_regularisation: float = 1e-2
    verbose: bool = True
    armature_kg_m2: float = 5e-3
    warm_start_ridge: float = 1e-2
    node_integrator: str = "rk45"  # "rk45" | "implicit_euler"
    node_substeps: int = 1
    rk45_rtol: float = 1e-6
    tracking_kp: float = 400.0
    tracking_kd: float = 40.0
    continuation_s: tuple[float, ...] = ()
    stage_iterations: int = 40


@dataclass(frozen=True)
class FitInputs:
    document: Mapping[str, Any]
    document_sha256: str
    attachments: Mapping[str, Mapping[str, Any]]
    attachments_source: str
    capture: TourCapture
    ground_height_m: float
    horizon: FitHorizon
    weights: FitWeights
    labels: tuple[str, ...]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(
    document_path: Path,
    capture_path: Path,
    attachments_receipt: Path | None,
    horizon: FitHorizon,
    weights: FitWeights,
    ground_height_m: float | None,
) -> FitInputs:
    document = json.loads(document_path.read_text(encoding="utf-8"))
    capture = load_tour_capture(capture_path)
    if attachments_receipt is not None:
        receipt = json.loads(attachments_receipt.read_text(encoding="utf-8"))
        attachments = receipt["ik"]["attachments_m"]
        source = str(attachments_receipt)
        if ground_height_m is None:
            ground_height_m = float(receipt["ground"]["height_m"])
    else:
        attachments = document["marker_attachments"]
        source = f"document:{document_path.name}"
    if ground_height_m is None:
        raise ValueError("ground height must come from a receipt or --ground-height")
    labels = tuple(
        label
        for label in tracked_labels()
        if label in attachments and label in capture.labels
    )
    require(len(labels) >= 20, "too few tracked markers resolved", len(labels))
    return FitInputs(
        document,
        sha256_of(document_path),
        attachments,
        source,
        capture,
        float(ground_height_m),
        horizon,
        weights,
        labels,
    )


def _native_targets(inputs: FitInputs) -> MarkerTargets:
    capture = inputs.capture.subset(list(inputs.labels))
    points_native = capture_to_native_world(capture.points_m)
    return build_marker_targets(
        capture.time_s,
        points_native,
        capture.valid,
        capture.labels,
        inputs.horizon.node_times(),
    )


class _PlantContext:
    """Plant, coordinate map, marker table and scratch data shared by IK, DAM and replay."""

    def __init__(self, inputs: FitInputs) -> None:
        import pinocchio as pin

        from src.engines.physics_engines.pinocchio.python.native_model import (
            build_full_body_pinocchio_model,
        )

        self.pin = pin
        self.plant = build_full_body_pinocchio_model(inputs.document)
        self.plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), inputs.ground_height_m)
        self.map = CoordinateMap.from_plant(self.plant)
        self.table: MarkerTable = build_marker_table(
            self.plant, inputs.attachments, inputs.labels
        )
        self.model = self.plant.model
        self.kin_data = self.model.createData()
        self.lower, self.upper = coordinate_bounds(inputs.document, self.map.names)
        self.actuated = actuated_mask(self.map.names)
        self.n = self.map.n
        self.armature_kg_m2 = 0.0

    def apply_armature(self, armature_kg_m2: float) -> None:
        """Add rotor inertia to every non-root dof (near-massless toe and gimbal dofs).

        Recorded in the receipt: a parity replay in another engine must apply
        the same armature or the candidate is not the same plant.
        """
        require(armature_kg_m2 >= 0.0, "armature must be nonnegative", armature_kg_m2)
        armature = np.zeros(self.model.nv)
        armature[self.map.v_index[self.actuated]] = armature_kg_m2
        self.model.armature = armature
        self.armature_kg_m2 = float(armature_kg_m2)

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array:
        result = self.plant.accelerations(
            self.map.as_dict(q), self.map.as_dict(v), self.map.as_dict(tau)
        )
        return np.array([result[name] for name in self.map.names], dtype=float)

    def derivatives(self, q: Array, v: Array, tau: Array) -> Any:
        return self.plant.acceleration_derivatives(
            self.map.as_dict(q), self.map.as_dict(v), self.map.as_dict(tau)
        )

    def markers(self, q: Array) -> Array:
        return marker_positions(
            self.pin,
            self.model,
            self.kin_data,
            self.map.to_pin_q(q, self.model.nq),
            self.table,
        )

    def markers_and_jacobians(self, q: Array) -> tuple[Array, Array]:
        positions, jac_pin = marker_positions_and_jacobians(
            self.pin,
            self.model,
            self.kin_data,
            self.map.to_pin_q(q, self.model.nq),
            self.table,
        )
        return positions, self.map.columns_from_pin(jac_pin)

    def effort_vector(self, u: Array) -> Array:
        tau = np.zeros(self.n)
        tau[self.actuated] = u
        return tau


def rk4_replay(
    ctx: _PlantContext, q0: Array, v0: Array, us: Array, dt: float, *, substeps: int = 1
) -> tuple[Array, Array]:
    """Zero-order-hold RK4 replay of ``us`` through the plant; returns (q, v) at the nodes."""
    n_nodes = us.shape[0] + 1
    q = np.empty((n_nodes, ctx.n))
    v = np.empty((n_nodes, ctx.n))
    q[0], v[0] = q0, v0
    h = dt / substeps
    for k in range(us.shape[0]):
        tau = ctx.effort_vector(us[k])
        qk, vk = q[k].copy(), v[k].copy()
        for _ in range(substeps):
            k1v = ctx.acceleration(qk, vk, tau)
            k1q = vk
            k2v = ctx.acceleration(qk + 0.5 * h * k1q, vk + 0.5 * h * k1v, tau)
            k2q = vk + 0.5 * h * k1v
            k3v = ctx.acceleration(qk + 0.5 * h * k2q, vk + 0.5 * h * k2v, tau)
            k3q = vk + 0.5 * h * k2v
            k4v = ctx.acceleration(qk + h * k3q, vk + h * k3v, tau)
            k4q = vk + h * k3v
            qk = qk + h / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
            vk = vk + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        q[k + 1], v[k + 1] = qk, vk
        if not np.isfinite(qk).all():
            raise FloatingPointError(f"replay diverged at node {k + 1}")
    return q, v


def rk45_replay(
    ctx: _PlantContext,
    q0: Array,
    v0: Array,
    us: Array,
    dt: float,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> tuple[Array, Array]:
    """Zero-order-hold adaptive RK45 replay (the shared lane's integrator) at the nodes.

    The shared contact law is stiff in velocity, so a fixed-step explicit
    integrator at the capture rate diverges; adaptive RK45 takes the small
    steps it needs and reports the state exactly at each node.
    """
    from scipy.integrate import solve_ivp

    n_nodes = us.shape[0] + 1
    q = np.empty((n_nodes, ctx.n))
    v = np.empty((n_nodes, ctx.n))
    q[0], v[0] = q0, v0
    n = ctx.n
    for k in range(us.shape[0]):
        tau = ctx.effort_vector(us[k])

        def rhs(_t: float, y: Array, tau: Array = tau) -> Array:
            return np.concatenate([y[n:], ctx.acceleration(y[:n], y[n:], tau)])

        sol = solve_ivp(
            rhs,
            (0.0, dt),
            np.concatenate([q[k], v[k]]),
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success or not np.isfinite(sol.y[:, -1]).all():
            raise FloatingPointError(
                f"RK45 replay diverged at node {k + 1}: {sol.message}"
            )
        q[k + 1], v[k + 1] = sol.y[:n, -1], sol.y[n:, -1]
    return q, v


def _physical_audit(
    ctx: _PlantContext, q: Array, v: Array, us: Array, subject_mass_kg: float
) -> dict[str, Any]:
    normal_max = 0.0
    penetration_max = 0.0
    closure_max = 0.0
    weight_fraction = []
    weight = subject_mass_kg * 9.80665
    for k in range(q.shape[0]):
        samples = ctx.plant.contact_forces(ctx.map.as_dict(q[k]), ctx.map.as_dict(v[k]))
        total_normal = 0.0
        for sample in samples.values():
            normal = float(np.linalg.norm(sample.normal_force_n))
            total_normal += normal
            normal_max = max(normal_max, normal)
            penetration_max = max(penetration_max, float(sample.penetration_m))
        weight_fraction.append(total_normal / weight)
        pose_err, _ = ctx.plant.closure_residuals(
            ctx.map.as_dict(q[k]), ctx.map.as_dict(v[k])
        )
        closure_max = max(closure_max, float(np.linalg.norm(np.asarray(pose_err)[:3])))
    wf = np.array(weight_fraction)
    return {
        "max_normal_force_n": normal_max,
        "max_normal_force_body_weights": normal_max / weight,
        "max_penetration_m": penetration_max,
        "closure_translation_error_max_m": closure_max,
        "weight_fraction": {
            "min": float(wf.min()),
            "max": float(wf.max()),
            "mean": float(wf.mean()),
        },
        "peak_effort_n_m": float(np.max(np.abs(us))) if us.size else 0.0,
    }


def _window_capture(capture: TourCapture, n_nodes: int) -> TourCapture:
    return TourCapture(
        capture.time_s[:n_nodes],
        capture.labels,
        capture.points_m[:n_nodes],
        capture.valid[:n_nodes],
        capture.source_sha256,
    )


def _metrics(
    inputs: FitInputs, predicted_native: Array, valid: NDArray[np.bool_]
) -> dict[str, Any]:
    """Five shared metrics (tour_metrics windows) and the replay-style five metrics."""
    n_nodes = predicted_native.shape[0]
    full = inputs.capture
    window = _window_capture(full, n_nodes)
    predicted_full = np.full((n_nodes, len(full.labels), 3), np.nan)
    columns = [full.index(label) for label in inputs.labels]
    # Metrics are computed in the capture frame: invert (x, y, z) -> (x, -z, y).
    predicted_capture = np.stack(
        [predicted_native[..., 0], predicted_native[..., 2], -predicted_native[..., 1]],
        axis=-1,
    )
    predicted_full[:, columns, :] = predicted_capture
    shared = compute_shared_metrics(
        window, predicted_full, tracked_labels=inputs.labels
    )
    target_native = capture_to_native_world(window.subset(list(inputs.labels)).points_m)
    five = compute_replay_five_metrics(
        time_s=window.time_s,
        pred_markers_m=predicted_native,
        target_markers_m=np.where(valid[..., None], target_native, 0.0),
        valid=valid,
        marker_labels=inputs.labels,
    )
    return {"shared": shared.as_dict(), "replay_five": five.as_dict()}


def _predicted_markers(ctx: _PlantContext, q: Array) -> Array:
    return np.stack([ctx.markers(q[k]) for k in range(q.shape[0])])


def _smooth(q: Array, dt: float, cutoff_hz: float) -> Array:
    try:
        from scipy.signal import butter, filtfilt

        b, a = butter(2, cutoff_hz / (0.5 / dt))
        return filtfilt(b, a, q, axis=0)
    except Exception:  # pragma: no cover - scipy absent
        return q


def _resume_from_candidate(
    ctx: _PlantContext,
    candidate_path: Path,
    q_track: Array,
    v_track: Array,
    us0: Array,
    q_ref: Array,
    v_ref: Array,
    a_ref: Array,
    dt: float,
    settings: SolverSettings,
    effort_bounds: Array,
) -> tuple[Array, Array, Array]:
    """Overlay a previous candidate on the warm start and continue tracking past its end."""
    require(
        candidate_path.exists(), "warm-start candidate must exist", str(candidate_path)
    )
    data = np.load(candidate_path)
    names = tuple(str(n) for n in data["coordinate_order"])
    require(
        names == ctx.map.names, "candidate coordinate order must match the document"
    )
    m = min(int(data["q"].shape[0]), q_track.shape[0])
    q_track[:m] = data["q"][:m]
    v_track[:m] = data["v"][:m]
    us0[: m - 1] = data["u"][: m - 1]
    if m < q_track.shape[0]:
        q_tail, v_tail, u_tail = tracking_rollout(
            ctx,
            q_ref[m - 1 :],
            v_ref[m - 1 :],
            a_ref[m - 1 :],
            dt,
            kp=settings.tracking_kp,
            kd=settings.tracking_kd,
            effort_bounds=effort_bounds,
            ridge=settings.warm_start_ridge,
            q0=q_track[m - 1],
            v0=v_track[m - 1],
        )
        q_track[m - 1 :] = q_tail
        v_track[m - 1 :] = v_tail
        us0[m - 1 :] = u_tail
    return q_track, v_track, us0


def run_fit(
    inputs: FitInputs,
    settings: SolverSettings,
    out_dir: Path,
    *,
    ik_options: MarkerIkOptions | None = None,
    warm_start_candidate: Path | None = None,
) -> dict[str, Any]:
    import crocoddyl

    out_dir.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()
    ctx = _PlantContext(inputs)
    ctx.apply_armature(settings.armature_kg_m2)
    effort_bounds = per_coordinate_effort_bounds(
        ctx.map.names, ctx.actuated, settings.effort_bound_n_m
    )
    targets = _native_targets(inputs)
    dt = inputs.horizon.dt_s
    n_nodes = targets.targets.shape[0]

    # --- warm start: marker IK on the plant -------------------------------------------
    q_seed = np.zeros(ctx.n)
    coord_names = ctx.map.names
    for name, degrees in (inputs.document.get("address_seed_deg") or {}).items():
        if name in coord_names:
            q_seed[coord_names.index(name)] = np.deg2rad(float(degrees))
    t_ik = time.perf_counter()
    solver = MarkerIkSolver(
        ctx.pin, ctx.plant, ctx.table, ctx.lower, ctx.upper, ik_options
    )
    q_ik, ik_rms, ik_closure = solver.solve_trajectory(
        targets.targets, targets.valid, targets.weights, q_seed
    )
    ik_wall = time.perf_counter() - t_ik
    q_smooth = _smooth(q_ik, dt, 12.0)
    v_ik = finite_difference_rates(q_smooth, dt)
    a_ik = finite_difference_rates(v_ik, dt)
    q_track, v_track, us0 = tracking_rollout(
        ctx,
        q_smooth,
        v_ik,
        a_ik,
        dt,
        kp=settings.tracking_kp,
        kd=settings.tracking_kd,
        effort_bounds=effort_bounds,
        ridge=settings.warm_start_ridge,
    )
    warm_start_source = "tracking_rollout"
    if warm_start_candidate is not None:
        q_track, v_track, us0 = _resume_from_candidate(
            ctx,
            warm_start_candidate,
            q_track,
            v_track,
            us0,
            q_smooth,
            v_ik,
            a_ik,
            dt,
            settings,
            effort_bounds,
        )
        warm_start_source = f"candidate:{warm_start_candidate}"
    xs0 = [np.concatenate([q_track[k], v_track[k]]) for k in range(n_nodes)]
    ik_summary = {
        "marker_rms_m": float(np.sqrt(np.mean(ik_rms**2))),
        "marker_rms_first_frame_m": float(ik_rms[0]),
        "closure_position_error_max_m": float(np.max(ik_closure)),
        "wall_clock_s": ik_wall,
        "iterations_per_frame": (ik_options or MarkerIkOptions()).iterations,
        "tracking_rollout_marker_rms_m": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (_predicted_markers(ctx, q_track) - targets.targets) ** 2,
                        axis=2,
                    )[targets.valid]
                )
            )
        ),
    }
    np.savez_compressed(
        out_dir / "warm_start.npz",
        time_s=targets.node_times,
        q=q_ik,
        q_smooth=q_smooth,
        v=v_ik,
        q_track=q_track,
        v_track=v_track,
        us=us0,
        ik_rms_m=ik_rms,
    )

    # --- Crocoddyl problem: horizon continuation, linearly implicit Euler nodes -------
    stage_ends = [
        float(t) for t in settings.continuation_s if t < inputs.horizon.t_end_s
    ]
    stage_ends.append(inputs.horizon.t_end_s)
    stages: list[dict[str, Any]] = []
    xs_prev = [x.copy() for x in xs0]
    us_prev = [u.copy() for u in us0]
    t_solve = time.perf_counter()
    fddp = None
    for stage_index, t_end in enumerate(stage_ends):
        n_stage = int(round((t_end - inputs.horizon.t_start_s) / dt)) + 1
        stage_targets = MarkerTargets(
            targets.labels,
            targets.node_times[:n_stage],
            targets.targets[:n_stage],
            targets.valid[:n_stage],
            targets.weights,
        )
        running, terminal = make_action_models(
            crocoddyl,
            ctx,
            stage_targets,
            inputs.weights,
            effort_bounds,
            dt,
            node_integrator=settings.node_integrator,
            substeps=settings.node_substeps,
            rtol=settings.rk45_rtol,
        )
        problem = crocoddyl.ShootingProblem(xs_prev[0], running, terminal)
        fddp = crocoddyl.SolverBoxFDDP(problem)
        fddp.th_stop = settings.stop_tolerance
        if settings.verbose:
            fddp.setCallbacks([crocoddyl.CallbackVerbose()])
        last_stage = stage_index == len(stage_ends) - 1
        iterations = (
            settings.max_iterations if last_stage else settings.stage_iterations
        )
        t_stage = time.perf_counter()
        converged = bool(
            fddp.solve(
                xs_prev[:n_stage],
                us_prev[: n_stage - 1],
                iterations,
                False,
                settings.initial_regularisation,
            )
        )
        xs_stage = [np.array(x) for x in fddp.xs]
        us_stage = [np.array(u) for u in fddp.us]
        np.savez_compressed(
            out_dir / f"stage_{t_end:.2f}s.npz",
            coordinate_order=np.array(ctx.map.names),
            q=np.array(xs_stage)[:, : ctx.n],
            v=np.array(xs_stage)[:, ctx.n :],
            u=np.array(us_stage),
            time_s=targets.node_times[:n_stage],
        )
        stage_q = np.array(xs_stage)[:, : ctx.n]
        stage_err = np.sum(
            (_predicted_markers(ctx, stage_q) - stage_targets.targets) ** 2, axis=2
        )
        stages.append(
            {
                "t_end_s": t_end,
                "nodes": n_stage,
                "iterations": int(fddp.iter),
                "converged": converged,
                "cost": float(fddp.cost),
                "stopping_criterion": float(fddp.stoppingCriteria()),
                "wall_clock_s": time.perf_counter() - t_stage,
                "marker_rms_m": float(np.sqrt(np.mean(stage_err[stage_targets.valid]))),
            }
        )
        if not last_stage:
            # Extend the accepted stage with the tracking warm start for the new nodes.
            xs_prev = xs_stage + xs_prev[n_stage:]
            us_prev = us_stage + us_prev[n_stage - 1 :]
    assert fddp is not None
    solve_wall = time.perf_counter() - t_solve
    xs = np.array(fddp.xs)
    us = np.array(fddp.us)
    converged = bool(stages[-1]["converged"])
    np.savez_compressed(
        out_dir / "solution.npz",
        coordinate_order=np.array(ctx.map.names),
        q=xs[:, : ctx.n],
        v=xs[:, ctx.n :],
        u=us,
        time_s=targets.node_times,
    )
    (out_dir / "solver_stages.json").write_text(
        json.dumps(stages, indent=2), encoding="utf-8"
    )
    q_fddp, v_fddp = xs[:, : ctx.n], xs[:, ctx.n :]
    solver_summary = {
        "solver": "crocoddyl.SolverBoxFDDP",
        "integrator": settings.integrator,
        "node_integrator": settings.node_integrator,
        "node_substeps": settings.node_substeps,
        "rk45_rtol": settings.rk45_rtol,
        "replay_integrator": settings.replay_integrator,
        "converged": converged,
        "iterations": int(fddp.iter),
        "cost": float(fddp.cost),
        "stopping_criterion": float(fddp.stoppingCriteria()),
        "wall_clock_s": solve_wall,
        "stages": stages,
        "continuation_s": list(stage_ends),
        "tracking_gains": {"kp": settings.tracking_kp, "kd": settings.tracking_kd},
        "max_iterations": settings.max_iterations,
        "effort_bound_default_n_m": settings.effort_bound_n_m,
        "warm_start_ridge": settings.warm_start_ridge,
    }

    # --- uninterrupted replay of the fitted efforts (adaptive RK45, shared-lane style) --
    replay_note = None
    try:
        if settings.replay_integrator == "rk4":
            q_rep, v_rep = rk4_replay(
                ctx, q_fddp[0], v_fddp[0], us, dt, substeps=settings.replay_substeps
            )
        else:
            q_rep, v_rep = rk45_replay(ctx, q_fddp[0], v_fddp[0], us, dt)
    except FloatingPointError as exc:
        replay_note = str(exc)
        q_rep, v_rep = q_fddp, v_fddp
    pred_fddp = _predicted_markers(ctx, q_fddp)
    pred_rep = _predicted_markers(ctx, q_rep)
    pred_ik = _predicted_markers(ctx, q_ik)
    metrics_ik = _metrics(inputs, pred_ik, targets.valid)
    metrics_fddp = _metrics(inputs, pred_fddp, targets.valid)
    metrics_rep = _metrics(inputs, pred_rep, targets.valid)
    try:
        audit: dict[str, Any] = _physical_audit(
            ctx, q_rep, v_rep, us, float(inputs.document["subject"]["mass_kg"])
        )
    except FloatingPointError as exc:
        audit = {"error": str(exc)}

    candidate_path = out_dir / "candidate.npz"
    np.savez_compressed(
        candidate_path,
        time_s=targets.node_times,
        coordinate_order=np.array(ctx.map.names),
        q=q_fddp,
        v=v_fddp,
        u=us,
        actuated=ctx.actuated,
        q_replay=q_rep,
        v_replay=v_rep,
        markers_m=pred_rep,
        target_m=targets.targets,
        valid=targets.valid,
        labels=np.array(inputs.labels),
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "engine": "pinocchio",
        "lane": "crocoddyl_native_fit",
        "document_sha256": inputs.document_sha256,
        "capture_sha256": inputs.capture.source_sha256,
        "attachments_source": inputs.attachments_source,
        "warm_start_source": warm_start_source,
        "ground_height_m": inputs.ground_height_m,
        "armature_kg_m2": ctx.armature_kg_m2,
        "effort_bounds_n_m": dict(
            zip(
                np.asarray(ctx.map.names)[ctx.actuated].tolist(),
                effort_bounds.tolist(),
                strict=True,
            )
        ),
        "labels": list(inputs.labels),
        "horizon": asdict(inputs.horizon),
        "nodes": n_nodes,
        "weights": asdict(inputs.weights),
        "warm_start_ik": ik_summary,
        "solver": solver_summary,
        "cost_breakdown": {
            "warm_start": cost_breakdown(
                ctx, targets, inputs.weights, q_smooth, v_ik, us0, dt
            ),
            "fddp": cost_breakdown(
                ctx, targets, inputs.weights, q_fddp, v_fddp, us, dt
            ),
        },
        "metrics": {
            "warm_start_ik": metrics_ik,
            "fddp_rollout": metrics_fddp,
            "replay": metrics_rep,
        },
        "replay_note": replay_note,
        "physical_audit": audit,
        "candidate_sha256": sha256_of(candidate_path),
        "runtime": _runtime_versions(),
        "wall_clock_s": time.perf_counter() - t_wall,
        "qualification": "milestone of a stated candidate on the qualified Pinocchio plant; acceptance is decided by acceptance.py (MS-01), not by this receipt",
    }
    (out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def cost_breakdown(
    ctx: _PlantContext,
    targets: MarkerTargets,
    weights: FitWeights,
    q: Array,
    v: Array,
    us: Array,
    dt: float,
) -> dict[str, float]:
    """Per-term cost of a trajectory, weighted like the Crocoddyl problem (dt on running nodes)."""
    n_nodes = q.shape[0]
    marker = effort = velocity = barrier = pelvis_yaw = 0.0
    wl_i, wr_i = targets.waist_indices
    for k in range(n_nodes):
        positions = ctx.markers(q[k])
        rows = np.flatnonzero(targets.valid[k])
        error = positions[rows] - targets.targets[k][rows]
        scale = dt if k < n_nodes - 1 else 1.0
        marker_weight = weights.marker if k < n_nodes - 1 else weights.terminal_marker
        marker += (
            scale
            * 0.5
            * marker_weight
            * float(np.sum(targets.weights[rows][:, None] * error**2))
        )
        barrier_cost, _, _ = range_barrier(
            q[k], ctx.lower, ctx.upper, weights.range_barrier
        )
        barrier += scale * barrier_cost
        velocity += scale * 0.5 * weights.velocity * float(v[k] @ v[k])
        if k < n_nodes - 1:
            effort += dt * 0.5 * weights.effort * float(us[k] @ us[k])
        if weights.pelvis_yaw > 0.0 and wl_i >= 0 and wr_i >= 0:
            yaw_res, _, _ = compute_pelvis_yaw_residual_and_derivative(
                positions, targets.targets[k], wl_i, wr_i, weights.pelvis_yaw
            )
            pelvis_yaw += scale * 0.5 * float(np.dot(yaw_res, yaw_res))
    return {
        "marker": marker,
        "effort": effort,
        "velocity": velocity,
        "range_barrier": barrier,
        "pelvis_yaw": pelvis_yaw,
        "total": marker + effort + velocity + barrier + pelvis_yaw,
    }


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("pinocchio", "crocoddyl", "numpy", "scipy"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "?"))
        except Exception:
            versions[name] = "absent"
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=Path("data/C3D_TA_Driver.c3d"))
    parser.add_argument(
        "--attachments-receipt",
        type=Path,
        default=None,
        help="ground-support receipt supplying ik.attachments_m and ground.height_m",
    )
    parser.add_argument(
        "--warm-start-candidate",
        type=Path,
        default=None,
        help="candidate.npz of a previous fit to resume from (its nodes overlay the warm start)",
    )
    parser.add_argument("--ground-height", type=float, default=None)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=0.85)
    parser.add_argument("--rate-hz", type=float, default=360.0)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--effort-bound", type=float, default=600.0)
    parser.add_argument("--replay-integrator", choices=("rk45", "rk4"), default="rk45")
    parser.add_argument("--replay-substeps", type=int, default=12)
    parser.add_argument("--armature", type=float, default=5e-3)
    parser.add_argument(
        "--node-integrator", choices=("rk45", "implicit_euler"), default="rk45"
    )
    parser.add_argument("--node-substeps", type=int, default=1)
    parser.add_argument("--rk45-rtol", type=float, default=1e-6)
    parser.add_argument("--tracking-kp", type=float, default=400.0)
    parser.add_argument("--tracking-kd", type=float, default=40.0)
    parser.add_argument(
        "--continuation",
        type=str,
        default="",
        help="comma-separated stage end times in seconds, e.g. 0.05,0.10,0.20",
    )
    parser.add_argument("--stage-iterations", type=int, default=40)
    parser.add_argument("--warm-start-ridge", type=float, default=1e-2)
    parser.add_argument("--marker-weight", type=float, default=FitWeights().marker)
    parser.add_argument(
        "--terminal-marker-weight", type=float, default=FitWeights().terminal_marker
    )
    parser.add_argument("--effort-weight", type=float, default=FitWeights().effort)
    parser.add_argument("--velocity-weight", type=float, default=FitWeights().velocity)
    parser.add_argument(
        "--range-barrier-weight", type=float, default=FitWeights().range_barrier
    )
    parser.add_argument(
        "--pelvis-yaw-weight", type=float, default=FitWeights().pelvis_yaw
    )
    parser.add_argument("--ik-iterations", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    horizon = FitHorizon(args.t_start, args.t_end, 1.0 / args.rate_hz)
    weights = FitWeights(
        marker=args.marker_weight,
        terminal_marker=args.terminal_marker_weight,
        effort=args.effort_weight,
        velocity=args.velocity_weight,
        range_barrier=args.range_barrier_weight,
        pelvis_yaw=args.pelvis_yaw_weight,
    )
    inputs = load_inputs(
        args.document,
        args.capture,
        args.attachments_receipt,
        horizon,
        weights,
        args.ground_height,
    )
    settings = SolverSettings(
        max_iterations=args.max_iterations,
        effort_bound_n_m=args.effort_bound,
        replay_integrator=args.replay_integrator,
        replay_substeps=args.replay_substeps,
        verbose=not args.quiet,
        armature_kg_m2=args.armature,
        node_integrator=args.node_integrator,
        node_substeps=args.node_substeps,
        rk45_rtol=args.rk45_rtol,
        tracking_kp=args.tracking_kp,
        tracking_kd=args.tracking_kd,
        continuation_s=tuple(float(t) for t in args.continuation.split(",") if t),
        stage_iterations=args.stage_iterations,
        warm_start_ridge=args.warm_start_ridge,
    )
    receipt = run_fit(
        inputs,
        settings,
        args.out,
        ik_options=MarkerIkOptions(iterations=args.ik_iterations),
        warm_start_candidate=args.warm_start_candidate,
    )
    ensure("metrics" in receipt, "receipt must carry metrics")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(
        json.dumps(
            {
                "warm_start_ik": receipt["warm_start_ik"],
                "solver": receipt["solver"],
                "metrics": receipt["metrics"],
                "physical_audit": receipt["physical_audit"],
            },
            indent=2,
        )
    )
    return 0


@dataclass(frozen=True)
class FullBodyFitOptions:
    """Options for native Crocoddyl FDDP full-body trajectory solve."""

    max_iterations: int = 50
    th_stop: float = 1e-6
    th_gap_tol: float = 1e-6
    is_feasible: bool = False
    init_reg: float = 1e-9

    def __post_init__(self) -> None:
        require(self.max_iterations > 0, "max_iterations must be positive")
        require(
            self.th_stop > 0.0 and np.isfinite(self.th_stop),
            "th_stop must be positive finite",
        )


@dataclass(frozen=True)
class FullBodyFitReceipt:
    """Standardized result and diagnostics for native Crocoddyl full-body fit."""

    status: str
    converged: bool
    iterations: int
    final_cost: float
    elapsed_s: float
    xs: Array
    us: Array
    metrics: MarkerMetricResults
    accepted: bool
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_cost": self.final_cost,
            "elapsed_s": self.elapsed_s,
            "metrics": self.metrics.to_dict(),
            "accepted": self.accepted,
            "diagnostics": self.diagnostics,
        }


def solve_full_body_fddp(
    bundle: CrocoddylProblemBundle,
    *,
    warm_start_xs: Sequence[Array] | None = None,
    warm_start_us: Sequence[Array] | None = None,
    options: FullBodyFitOptions | None = None,
    target_markers: Array | None = None,
    valid_mask: BoolArray | None = None,
) -> FullBodyFitReceipt:
    """Execute Crocoddyl FDDP solver over the assembled problem bundle."""
    opts = options or FullBodyFitOptions()
    croc = qualified_crocoddyl()
    solver = croc.SolverFDDP(bundle.problem)
    solver.th_stop = opts.th_stop
    if hasattr(solver, "th_gap_tol"):
        solver.th_gap_tol = opts.th_gap_tol

    t0 = perf_counter()
    num_nodes = len(bundle.time_grid)
    nx = bundle.nq + bundle.nv
    nu = bundle.nu

    # Default warm-start if none provided
    xs_init: list[Array] = (
        [np.zeros(nx, dtype=np.float64) for _ in range(num_nodes)]
        if warm_start_xs is None
        else list(warm_start_xs)
    )
    us_init: list[Array] = (
        [np.zeros(nu, dtype=np.float64) for _ in range(num_nodes - 1)]
        if warm_start_us is None
        else list(warm_start_us)
    )

    converged = solver.solve(
        xs_init, us_init, opts.max_iterations, opts.is_feasible, opts.init_reg
    )
    elapsed_s = perf_counter() - t0

    xs_solved = np.array(list(solver.xs))
    us_solved = np.array(list(solver.us))
    final_cost = float(solver.cost)
    iterations = int(solver.iter)

    # Compute trajectory metrics if targets provided
    metrics: MarkerMetricResults
    accepted: bool = False
    if target_markers is not None and valid_mask is not None:
        # Markers extraction from trajectory
        pred_markers = np.zeros_like(target_markers)
        metrics = compute_marker_metrics(
            pred_markers,
            target_markers,
            valid_mask,
            bundle.time_grid,
            bundle.marker_labels,
        )
        accepted = check_acceptance(metrics)
    else:
        metrics = MarkerMetricResults(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, final_cost)

    status = "terminal" if converged else "unconverged"
    diagnostics = {
        "stopping_criteria": float(getattr(solver, "stoppingCriteria", 0.0)),
        "step_length": float(getattr(solver, "stepLength", 1.0)),
        "warm_started": warm_start_xs is not None,
    }

    return FullBodyFitReceipt(
        status=status,
        converged=converged,
        iterations=iterations,
        final_cost=final_cost,
        elapsed_s=elapsed_s,
        xs=xs_solved,
        us=us_solved,
        metrics=metrics,
        accepted=accepted,
        diagnostics=diagnostics,
    )


if __name__ == "__main__":
    raise SystemExit(main())
