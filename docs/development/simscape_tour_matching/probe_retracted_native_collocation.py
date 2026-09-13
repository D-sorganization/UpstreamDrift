"""Bounded first-window Pinocchio collocation on retracted native weld nodes."""

from __future__ import annotations

import argparse
import json
import hashlib
from time import perf_counter
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import Bounds, minimize

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.motion_matching.node_retraction import (
    retract_node,
    scaled_tangent_basis,
)
from src.shared.python.motion_matching.constrained_trajectory import (
    compose_chart_residual_jacobian,
    spline_chart_derivative_jacobians,
    spline_node_derivative_maps,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--chart-check-step", type=float, default=1e-6)
    parser.add_argument("--use-constraint-jacobian", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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

    def mapping(value: np.ndarray) -> dict[str, float]:
        return dict(zip(names, value.tolist(), strict=True))

    linear = [model.closure_position_linearization(mapping(q)) for q in references]
    bases = [scaled_tangent_basis(item.jacobian, scales) for item in linear]
    if any(basis.shape != bases[0].shape for basis in bases):
        raise ValueError("Native node chart dimension changed")
    dimension = bases[0].shape[1]
    retraction_radius = 0.2

    def nodes(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chart = flat.reshape(args.nodes, dimension)
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

    first, second = spline_node_derivative_maps(times)

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
        spline = CubicSpline(times, q, axis=0)
        qd, qdd = spline_chart_derivative_jacobians(first, second, node_jacobians)
        return (
            q,
            np.asarray(spline(times, 1)),
            np.asarray(spline(times, 2)),
            node_jacobians,
            qd,
            qdd,
        )

    def residual(flat: np.ndarray) -> np.ndarray:
        q, velocity, acceleration, _, _, _ = state(flat)
        result = []
        for qi, vi, ai in zip(q, velocity, acceleration, strict=True):
            value = model.closure_trajectory_residuals(
                mapping(qi), mapping(vi), mapping(ai)
            )
            result.extend(value.rate)
            result.extend(value.acceleration)
        return np.asarray(result)

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
        return derivative.reshape(args.nodes * 12, args.nodes * dimension)

    initial = np.zeros(args.nodes * dimension)
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
    defect = residual(np.asarray(result.x))
    args.output.write_text(
        json.dumps(
            {
                "nodes": args.nodes,
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
                "iterations": int(result.nit),
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
                "scope": "Bounded retracted-node preflight only; no marker objective, effort fit, or forward replay.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
