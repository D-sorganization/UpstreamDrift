"""Native Crocoddyl full-body fit of the tour capture on the Pinocchio plant (MS-31, #10338).

Pipeline: document + capture -> marker IK warm start (Gauss-Newton on the
plant, weld closure enforced) -> least-squares effort warm start from the
plant's effort sensitivity -> Crocoddyl ``SolverBoxFDDP`` over per-node
efforts with a Gauss-Newton marker cost, effort / rate regularisation and a
human-range barrier -> uninterrupted RK4 replay through the same plant ->
shared metrics, physical audit, receipt and candidate.

The physics (compliant Hunt-Crossley/Coulomb foot contact, six-dimensional
weld closure) is the qualified ``FullBodyPinocchioModel`` used by every other
lane, so a candidate from this fit replays in MuJoCo and Drake without a
model change. ``pinocchio`` and ``crocoddyl`` are imported only here and in
``marker_kinematics``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
    FitHorizon,
    FitWeights,
    MarkerTargets,
    actuated_mask,
    build_marker_targets,
    coordinate_bounds,
    finite_difference_rates,
    least_squares_controls,
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
from src.shared.python.motion_matching.replay_metrics import compute_replay_five_metrics
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    load_tour_capture,
    tracked_labels,
)
from src.shared.python.motion_matching.tour_metrics import compute_shared_metrics

Array = NDArray[np.float64]

RECEIPT_SCHEMA = "matched-swing-fit/pinocchio-crocoddyl-v1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverSettings:
    max_iterations: int = 200
    stop_tolerance: float = 1e-6
    effort_bound_n_m: float = 600.0
    integrator: str = "euler"  # "euler" | "rk4"
    initial_regularisation: float = 1e-2
    verbose: bool = True


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
    require(
        ground_height_m is not None,
        "ground height must come from a receipt or --ground-height",
    )
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
        self.kin_data = self.plant.model.createData()
        self.lower, self.upper = coordinate_bounds(inputs.document, self.map.names)
        self.actuated = actuated_mask(self.map.names)
        self.n = self.map.n

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
            self.plant.model,
            self.kin_data,
            self.map.to_pin_q(q, self.plant.model.nq),
            self.table,
        )

    def markers_and_jacobians(self, q: Array) -> tuple[Array, Array]:
        positions, jac_pin = marker_positions_and_jacobians(
            self.pin,
            self.plant.model,
            self.kin_data,
            self.map.to_pin_q(q, self.plant.model.nq),
            self.table,
        )
        return positions, self.map.columns_from_pin(jac_pin)

    def effort_vector(self, u: Array) -> Array:
        tau = np.zeros(self.n)
        tau[self.actuated] = u
        return tau


def _make_dam_class(crocoddyl: Any) -> type:
    class MarkerTrackingDAM(crocoddyl.DifferentialActionModelAbstract):  # type: ignore[misc]
        """Differential action model: plant dynamics + Gauss-Newton marker cost."""

        def __init__(
            self,
            ctx: _PlantContext,
            target: Array,
            valid: NDArray[np.bool_],
            marker_weights: Array,
            weights: FitWeights,
            *,
            terminal: bool,
            effort_bound: float,
        ) -> None:
            self.ctx = ctx
            self.n = ctx.n
            nu = 0 if terminal else int(ctx.actuated.sum())
            crocoddyl.DifferentialActionModelAbstract.__init__(
                self, crocoddyl.StateVector(2 * ctx.n), nu, 1
            )
            self.terminal = terminal
            self.target = target
            self.rows = np.flatnonzero(valid)
            self.marker_w = (
                weights.terminal_marker if terminal else weights.marker
            ) * marker_weights[self.rows]
            self.weights = weights
            if nu:
                self.u_lb = -effort_bound * np.ones(nu)
                self.u_ub = effort_bound * np.ones(nu)

        def _cost_terms(
            self, q: Array, v: Array, u: Array | None, with_derivatives: bool
        ) -> tuple[float, dict[str, Array]]:
            ctx = self.ctx
            if with_derivatives:
                positions, jac = ctx.markers_and_jacobians(q)
            else:
                positions, jac = ctx.markers(q), None
            error = positions[self.rows] - self.target[self.rows]
            cost = 0.5 * float(np.sum(self.marker_w[:, None] * error**2))
            barrier_cost, barrier_grad, barrier_hess = range_barrier(
                q, ctx.lower, ctx.upper, self.weights.range_barrier
            )
            cost += barrier_cost + 0.5 * self.weights.velocity * float(v @ v)
            if u is not None and u.size:
                cost += 0.5 * self.weights.effort * float(u @ u)
            terms: dict[str, Array] = {}
            if with_derivatives and jac is not None:
                weighted = self.marker_w[:, None] * error  # (m, 3)
                jac_rows = jac[self.rows]  # (m, 3, n)
                terms["Lq"] = np.einsum("mij,mi->j", jac_rows, weighted) + barrier_grad
                terms["Lqq"] = np.einsum(
                    "mij,mik->jk", jac_rows * self.marker_w[:, None, None], jac_rows
                ) + np.diag(barrier_hess)
            return cost, terms

        def calc(self, data: Any, x: Array, u: Array | None = None) -> None:
            q, v = x[: self.n], x[self.n :]
            u_arr = None if (u is None or self.terminal) else np.asarray(u, dtype=float)
            tau = (
                self.ctx.effort_vector(u_arr) if u_arr is not None else np.zeros(self.n)
            )
            data.xout[:] = self.ctx.acceleration(q, v, tau)
            data.cost, _ = self._cost_terms(q, v, u_arr, with_derivatives=False)

        def calcDiff(self, data: Any, x: Array, u: Array | None = None) -> None:
            q, v = x[: self.n], x[self.n :]
            u_arr = None if (u is None or self.terminal) else np.asarray(u, dtype=float)
            tau = (
                self.ctx.effort_vector(u_arr) if u_arr is not None else np.zeros(self.n)
            )
            der = self.ctx.derivatives(q, v, tau)
            data.Fx[:, : self.n] = der.dq
            data.Fx[:, self.n :] = der.dv
            if u_arr is not None:
                data.Fu[:, :] = der.deffort[:, self.ctx.actuated]
            _, terms = self._cost_terms(q, v, u_arr, with_derivatives=True)
            data.Lx[: self.n] = terms["Lq"]
            data.Lx[self.n :] = self.weights.velocity * v
            data.Lxx[:, :] = 0.0
            data.Lxx[: self.n, : self.n] = terms["Lqq"]
            data.Lxx[self.n :, self.n :] = self.weights.velocity * np.eye(self.n)
            if u_arr is not None:
                data.Lu[:] = self.weights.effort * u_arr
                data.Luu[:, :] = self.weights.effort * np.eye(u_arr.size)
                data.Lxu[:, :] = 0.0

        def createData(self) -> Any:
            return crocoddyl.DifferentialActionDataAbstract(self)

    return MarkerTrackingDAM


def _integrate(crocoddyl: Any, dam: Any, dt: float, integrator: str) -> Any:
    if integrator == "rk4":
        return crocoddyl.IntegratedActionModelRK(dam, crocoddyl.RKType.four, dt)
    return crocoddyl.IntegratedActionModelEuler(dam, dt)


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


def run_fit(
    inputs: FitInputs,
    settings: SolverSettings,
    out_dir: Path,
    *,
    ik_options: MarkerIkOptions | None = None,
) -> dict[str, Any]:
    import crocoddyl

    out_dir.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()
    ctx = _PlantContext(inputs)
    targets = _native_targets(inputs)
    dt = inputs.horizon.dt_s
    n_nodes = targets.targets.shape[0]

    # --- warm start: marker IK on the plant -------------------------------------------
    q_seed = np.zeros(ctx.n)
    for name, degrees in (inputs.document.get("address_seed_deg") or {}).items():
        if name in ctx.map.names:
            q_seed[ctx.map.names.index(name)] = np.deg2rad(float(degrees))
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
    us0 = np.empty((n_nodes - 1, int(ctx.actuated.sum())))
    for k in range(n_nodes - 1):
        der = ctx.derivatives(q_smooth[k], v_ik[k], np.zeros(ctx.n))
        a_zero = ctx.acceleration(q_smooth[k], v_ik[k], np.zeros(ctx.n))
        us0[k] = np.clip(
            least_squares_controls(
                a_ik[k], a_zero, np.asarray(der.deffort), ctx.actuated
            ),
            -settings.effort_bound_n_m,
            settings.effort_bound_n_m,
        )
    xs0 = [np.concatenate([q_smooth[k], v_ik[k]]) for k in range(n_nodes)]
    ik_summary = {
        "marker_rms_m": float(np.sqrt(np.mean(ik_rms**2))),
        "marker_rms_first_frame_m": float(ik_rms[0]),
        "closure_position_error_max_m": float(np.max(ik_closure)),
        "wall_clock_s": ik_wall,
        "iterations_per_frame": (ik_options or MarkerIkOptions()).iterations,
    }
    np.savez_compressed(
        out_dir / "warm_start.npz",
        time_s=targets.node_times,
        q=q_ik,
        q_smooth=q_smooth,
        v=v_ik,
        us=us0,
        ik_rms_m=ik_rms,
    )

    # --- Crocoddyl problem --------------------------------------------------------------
    DAM = _make_dam_class(crocoddyl)
    running = []
    for k in range(n_nodes - 1):
        dam = DAM(
            ctx,
            targets.targets[k],
            targets.valid[k],
            targets.weights,
            inputs.weights,
            terminal=False,
            effort_bound=settings.effort_bound_n_m,
        )
        running.append(_integrate(crocoddyl, dam, dt, settings.integrator))
    terminal_dam = DAM(
        ctx,
        targets.targets[-1],
        targets.valid[-1],
        targets.weights,
        inputs.weights,
        terminal=True,
        effort_bound=settings.effort_bound_n_m,
    )
    terminal = _integrate(crocoddyl, terminal_dam, 0.0, settings.integrator)
    problem = crocoddyl.ShootingProblem(xs0[0], running, terminal)
    fddp = crocoddyl.SolverBoxFDDP(problem)
    fddp.th_stop = settings.stop_tolerance
    if settings.verbose:
        fddp.setCallbacks([crocoddyl.CallbackVerbose()])
    t_solve = time.perf_counter()
    converged = bool(
        fddp.solve(
            xs0,
            [u.copy() for u in us0],
            settings.max_iterations,
            False,
            settings.initial_regularisation,
        )
    )
    solve_wall = time.perf_counter() - t_solve
    xs = np.array(fddp.xs)
    us = np.array(fddp.us)
    q_fddp, v_fddp = xs[:, : ctx.n], xs[:, ctx.n :]
    solver_summary = {
        "solver": "crocoddyl.SolverBoxFDDP",
        "integrator": settings.integrator,
        "converged": converged,
        "iterations": int(fddp.iter),
        "cost": float(fddp.cost),
        "stopping_criterion": float(fddp.stoppingCriteria()),
        "wall_clock_s": solve_wall,
        "max_iterations": settings.max_iterations,
        "effort_bound_n_m": settings.effort_bound_n_m,
    }

    # --- uninterrupted RK4 replay of the fitted efforts ---------------------------------
    replay_note = None
    try:
        q_rep, v_rep = rk4_replay(ctx, q_fddp[0], v_fddp[0], us, dt, substeps=2)
    except FloatingPointError as exc:
        replay_note = str(exc)
        q_rep, v_rep = q_fddp, v_fddp
    pred_fddp = _predicted_markers(ctx, q_fddp)
    pred_rep = _predicted_markers(ctx, q_rep)
    pred_ik = _predicted_markers(ctx, q_ik)
    metrics_ik = _metrics(inputs, pred_ik, targets.valid)
    metrics_fddp = _metrics(inputs, pred_fddp, targets.valid)
    metrics_rep = _metrics(inputs, pred_rep, targets.valid)
    audit = _physical_audit(
        ctx, q_rep, v_rep, us, float(inputs.document["subject"]["mass_kg"])
    )

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
        "ground_height_m": inputs.ground_height_m,
        "labels": list(inputs.labels),
        "horizon": asdict(inputs.horizon),
        "nodes": n_nodes,
        "weights": asdict(inputs.weights),
        "warm_start_ik": ik_summary,
        "solver": solver_summary,
        "metrics": {
            "warm_start_ik": metrics_ik,
            "fddp_rollout": metrics_fddp,
            "rk4_replay": metrics_rep,
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
    parser.add_argument("--ground-height", type=float, default=None)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=0.85)
    parser.add_argument("--rate-hz", type=float, default=360.0)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--effort-bound", type=float, default=600.0)
    parser.add_argument("--integrator", choices=("euler", "rk4"), default="euler")
    parser.add_argument("--marker-weight", type=float, default=FitWeights().marker)
    parser.add_argument(
        "--terminal-marker-weight", type=float, default=FitWeights().terminal_marker
    )
    parser.add_argument("--effort-weight", type=float, default=FitWeights().effort)
    parser.add_argument("--velocity-weight", type=float, default=FitWeights().velocity)
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
        integrator=args.integrator,
        verbose=not args.quiet,
    )
    receipt = run_fit(
        inputs,
        settings,
        args.out,
        ik_options=MarkerIkOptions(iterations=args.ik_iterations),
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


if __name__ == "__main__":
    raise SystemExit(main())
