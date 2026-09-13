"""Bounded first-window Pinocchio collocation on retracted native weld nodes."""

from __future__ import annotations

import argparse
import json
import hashlib
from time import perf_counter
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import Bounds, minimize, least_squares

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.motion_matching.node_retraction import (
    retract_node,
    scaled_tangent_basis,
)
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.constrained_trajectory import (
    compose_chart_residual_jacobian,
    spline_chart_derivative_jacobians,
)


def validate_chart_bounds(values: np.ndarray, bound: float) -> float:
    """Reject infeasible returned charts; return their maximum absolute value."""
    array = np.asarray(values, dtype=float)
    if (
        array.size == 0
        or not np.isfinite(array).all()
        or not np.isfinite(bound)
        or bound <= 0
        or np.max(abs(array)) > bound + 1e-12
    ):
        raise ValueError("Returned state violates finite chart bounds")
    return float(np.max(abs(array)))


def make_path_spline(
    times: np.ndarray, coordinates: np.ndarray, initial_rate: np.ndarray | None
) -> CubicSpline:
    """Interpolate nodes with prescribed initial rate and natural final boundary.

    For sensitivity maps pass identity node values and a zero initial rate.
    Omitted rate retains the historical not-a-knot spline.
    """
    if initial_rate is None:
        return CubicSpline(times, coordinates, axis=0)
    velocity = np.asarray(initial_rate, dtype=float)
    if velocity.shape != coordinates.shape[1:] or not np.isfinite(velocity).all():
        raise ValueError("Initial rate must match finite coordinate dimensions")
    return CubicSpline(
        times,
        coordinates,
        axis=0,
        bc_type=((1, velocity), (2, np.zeros_like(velocity))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--chart-check-step", type=float, default=1e-6)
    parser.add_argument("--use-constraint-jacobian", action="store_true")
    parser.add_argument(
        "--solver", choices=("trust-constr", "least-squares"), default="trust-constr"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--marker-scale-m", type=float, default=0.001)
    args = parser.parse_args()
    tracking = args.candidate is not None
    if (
        tracking != (args.payload is not None)
        or not np.isfinite(args.marker_scale_m)
        or args.marker_scale_m <= 0
    ):
        raise ValueError(
            "Provide candidate and payload together and a positive marker scale"
        )
    if tracking and args.solver != "least-squares":
        raise ValueError("Marker tracking currently requires least-squares")
    if (
        args.output.exists()
        or args.nodes < 4
        or args.max_iterations < 1
        or not np.isfinite(args.finite_difference_step)
        or args.finite_difference_step <= 0.0
        or not np.isfinite(args.chart_check_step)
        or args.chart_check_step <= 0.0
    ):
        raise ValueError(
            "Output must be new with at least four nodes and one iteration"
        )
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = tuple(spec["coordinate_order"])
    source = json.loads(args.path.read_text())
    times = np.asarray(source["times_s"], dtype=float)[: args.nodes]
    references = np.asarray(source["coordinates"], dtype=float)[: args.nodes]
    if references.shape != (args.nodes, len(names)) or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid source path window")
    primitive = {
        item["coordinate"]: item["primitive"]
        for joint in spec["joints"]
        for item in joint["primitives"]
    }
    scales = np.asarray(
        [0.1 if primitive[name].startswith("P") else 1.0 for name in names]
    )
    model = NativePinocchioModel(spec)
    if tracking:
        candidate = NativeReplayCandidate.from_document(
            json.loads(args.candidate.read_text()),
            names,
            hashlib.sha256(raw).hexdigest(),
        ).document
        payload = json.loads(args.payload.read_text())
        indices = [
            payload["labels"].index(label) for label in candidate["marker_labels"]
        ]
        clock = np.asarray(payload["time_s"], dtype=float)
        frames = [int(np.argmin(abs(clock - time))) for time in times]
        if not np.allclose(clock[frames], times, atol=1e-8, rtol=0):
            raise ValueError("Path nodes must coincide with capture times")
        targets = np.asarray(payload["points_world_m"], dtype=float)[frames][:, indices]
        masks = np.asarray(payload["valid"], dtype=bool)[frames][:, indices]
        if not masks.any() or not np.isfinite(targets[masks]).all():
            raise ValueError("Invalid observed marker targets")
        references[0] = np.asarray(candidate["q0"], dtype=float)

    def mapping(value: np.ndarray) -> dict[str, float]:
        return dict(zip(names, value.tolist(), strict=True))

    linear = [model.closure_position_linearization(mapping(q)) for q in references]
    bases = [scaled_tangent_basis(item.jacobian, scales) for item in linear]
    if any(basis.shape != bases[0].shape for basis in bases):
        raise ValueError("Native node chart dimension changed")
    dimension = bases[0].shape[1]
    retraction_radius = 0.2
    free_start = dimension if tracking else 0
    variable_count = args.nodes * dimension - free_start

    def nodes(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chart = np.concatenate((np.zeros(free_start), flat)).reshape(
            args.nodes, dimension
        )
        retractions = []
        for reference, basis, coordinate in zip(references, bases, chart, strict=True):

            def closure(value: np.ndarray) -> np.ndarray:
                return model.closure_position_linearization(mapping(value)).position

            def jacobian(value: np.ndarray) -> np.ndarray:
                return model.closure_position_linearization(mapping(value)).jacobian

            retractions.append(
                retract_node(
                    reference,
                    basis,
                    coordinate,
                    closure,
                    jacobian,
                    state_scales=scales,
                    residual_scales=np.ones(6),
                    # Bound feasibility is enforced by the solver and audited
                    # independently before exporting a returned candidate.
                    radius=retraction_radius,
                )
            )
        return (
            np.asarray([item.state for item in retractions]),
            np.asarray([item.state_jacobian for item in retractions]),
        )

    initial_rate = np.asarray(candidate["qd0"], dtype=float) if tracking else None
    derivative_spline = make_path_spline(
        times, np.eye(args.nodes), np.zeros(args.nodes) if tracking else None
    )
    first, second = derivative_spline(times, 1), derivative_spline(times, 2)

    def state(
        flat: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        q, node_jacobians = nodes(flat)
        spline = make_path_spline(times, q, initial_rate)
        qd, qdd = spline_chart_derivative_jacobians(first, second, node_jacobians)
        return (
            q,
            np.asarray(spline(times, 1)),
            np.asarray(spline(times, 2)),
            node_jacobians,
            qd,
            qdd,
        )

    def marker_terms(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q, node_jacobians = nodes(flat)
        errors, derivatives = [], []
        for i, qi in enumerate(q):
            value = model.marker_derivatives(
                mapping(qi),
                candidate["marker_bodies"],
                np.asarray(candidate["marker_offsets_m"]),
            )
            errors.append(
                (
                    (value.positions_m - targets[i])[masks[i]] / args.marker_scale_m
                ).ravel()
            )
            local = np.einsum("mkq,qa->mka", value.dposition_dq, node_jacobians[i])[
                masks[i]
            ].reshape(-1, dimension)
            block = np.zeros((local.shape[0], args.nodes * dimension))
            block[:, i * dimension : (i + 1) * dimension] = local / args.marker_scale_m
            derivatives.append(block[:, free_start:])
        return np.concatenate(errors), np.vstack(derivatives)

    def residual(flat: np.ndarray) -> np.ndarray:
        q, velocity, acceleration, _, _, _ = state(flat)
        result = []
        for qi, vi, ai in zip(q, velocity, acceleration, strict=True):
            value = model.closure_trajectory_residuals(
                mapping(qi), mapping(vi), mapping(ai)
            )
            result.extend(value.rate)
            result.extend(value.acceleration)
        closure = np.asarray(result)
        return np.concatenate((closure, marker_terms(flat)[0])) if tracking else closure

    def jacobian(flat: np.ndarray) -> np.ndarray:
        q, velocity, acceleration, node_jacobians, qd, qdd = state(flat)
        local = [
            model.closure_trajectory_linearization(
                mapping(qi),
                mapping(vi),
                mapping(ai),
                finite_difference_step=args.finite_difference_step,
            )
            for qi, vi, ai in zip(q, velocity, acceleration, strict=True)
        ]
        derivative = compose_chart_residual_jacobian(
            np.asarray([item.dq[6:] for item in local]),
            np.asarray([item.dv[6:] for item in local]),
            np.asarray([item.da[6:] for item in local]),
            node_jacobians,
            qd,
            qdd,
        )
        closure = derivative.reshape(args.nodes * 12, args.nodes * dimension)[
            :, free_start:
        ]
        return np.vstack((closure, marker_terms(flat)[1])) if tracking else closure

    initial = np.zeros(variable_count)
    assembled = jacobian(initial)
    direct = np.empty_like(assembled)
    for column in range(initial.size):
        direction = np.zeros_like(initial)
        direction[column] = args.chart_check_step
        direct[:, column] = (
            residual(initial + direction) - residual(initial - direction)
        ) / (2.0 * args.chart_check_step)
    jacobian_difference = assembled - direct
    jacobian_scale = np.maximum(np.maximum(abs(assembled), abs(direct)), 1.0)
    constraint: dict[str, object] = {"type": "eq", "fun": residual}
    if args.use_constraint_jacobian:
        constraint["jac"] = jacobian
    initial_defect = residual(initial)
    started = perf_counter()
    if args.solver == "least-squares":
        result = least_squares(
            residual,
            initial,
            jac=jacobian if args.use_constraint_jacobian else "2-point",
            bounds=(-0.01, 0.01),
            max_nfev=args.max_iterations,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
    else:
        result = minimize(
            lambda x: float(x @ x),
            initial,
            method="trust-constr",
            constraints=constraint,
            bounds=Bounds(
                -0.01 * np.ones_like(initial),
                0.01 * np.ones_like(initial),
                keep_feasible=True,
            ),
            options={"maxiter": args.max_iterations, "gtol": 1e-12},
        )
    elapsed = perf_counter() - started
    chart_max = validate_chart_bounds(np.asarray(result.x), 0.01)
    values, _ = nodes(np.asarray(result.x))
    defect = residual(np.asarray(result.x))[: args.nodes * 12]
    audit_times = np.linspace(times[0], times[-1], 20 * (args.nodes - 1) + 1)
    audit_spline = make_path_spline(times, values, initial_rate)
    audit_values = np.asarray(
        [
            model.closure_trajectory_residuals(
                mapping(audit_spline(time)),
                mapping(audit_spline(time, 1)),
                mapping(audit_spline(time, 2)),
            )
            for time in audit_times
        ]
    )
    args.output.write_text(
        json.dumps(
            {
                "nodes": args.nodes,
                "marker_tracking": tracking,
                "first_pose_fixed": tracking,
                "initial_velocity_enforced": tracking,
                "initial_velocity_max_abs_error": float(
                    np.max(abs(audit_spline(times[0], 1) - initial_rate))
                )
                if tracking
                else None,
                "marker_scale_m": args.marker_scale_m if tracking else None,
                "marker_rms_m": float(
                    np.sqrt(np.mean(marker_terms(result.x)[0] ** 2) * 3)
                    * args.marker_scale_m
                )
                if tracking
                else None,
                "initial_pose_max_abs_error": float(
                    np.max(abs(values[0] - references[0]))
                ),
                "times_s": times.tolist(),
                "dense_audit_samples": len(audit_times),
                "dense_position_max_abs": float(np.max(abs(audit_values[:, 0]))),
                "dense_rate_max_abs": float(np.max(abs(audit_values[:, 1]))),
                "dense_acceleration_max_abs": float(np.max(abs(audit_values[:, 2]))),
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "path_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "solver_elapsed_s": elapsed,
                "function_evaluations": int(result.nfev),
                "chart_max_abs": chart_max,
                "chart_bound": 0.01,
                "chart_bounds_verified": True,
                "chart_coordinates": np.asarray(result.x).tolist(),
                "chart_dimension": dimension,
                "solver": args.solver,
                "iterations": int(result.nit)
                if args.solver == "trust-constr"
                else None,
                "optimizer_converged": bool(result.success),
                "message": str(result.message),
                "rate_acceleration_closure_max_abs": float(np.max(abs(defect))),
                "initial_rate_acceleration_closure_max_abs": float(
                    np.max(abs(initial_defect))
                ),
                "used_constraint_jacobian": args.use_constraint_jacobian,
                "residual_derivative": "node-level centered q/v differences with exact acceleration Jacobian",
                "finite_difference_step": args.finite_difference_step,
                "chart_check_step": args.chart_check_step,
                "chart_jacobian_max_abs_error": float(np.max(abs(jacobian_difference))),
                "chart_jacobian_max_relative_error": float(
                    np.max(abs(jacobian_difference) / jacobian_scale)
                ),
                "retraction_trial_radius": retraction_radius,
                "coordinates": values.tolist(),
                "scope": "Bounded path experiment; optional node marker tracking with fixed first pose. No initial velocity constraint, effort fit, or forward replay.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
