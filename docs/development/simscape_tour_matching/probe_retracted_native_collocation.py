"""Bounded first-window Pinocchio collocation on retracted native weld nodes."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.nodes < 4 or args.max_iterations < 1:
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

    def nodes(flat: np.ndarray) -> np.ndarray:
        chart = flat.reshape(args.nodes, dimension)
        values = []
        for reference, basis, coordinate in zip(references, bases, chart, strict=True):

            def closure(value: np.ndarray) -> np.ndarray:
                return model.closure_position_linearization(mapping(value)).position

            def jacobian(value: np.ndarray) -> np.ndarray:
                return model.closure_position_linearization(mapping(value)).jacobian

            values.append(
                retract_node(
                    reference,
                    basis,
                    coordinate,
                    closure,
                    jacobian,
                    state_scales=scales,
                    residual_scales=np.ones(6),
                    radius=0.1,
                ).state
            )
        return np.asarray(values)

    def residual(flat: np.ndarray) -> np.ndarray:
        q = nodes(flat)
        spline = CubicSpline(times, q, axis=0)
        result = []
        for qi, vi, ai in zip(q, spline(times, 1), spline(times, 2), strict=True):
            value = model.closure_trajectory_residuals(
                mapping(qi), mapping(vi), mapping(ai)
            )
            result.extend(value.rate)
            result.extend(value.acceleration)
        return np.asarray(result)

    initial = np.zeros(args.nodes * dimension)
    result = minimize(
        lambda x: float(x @ x),
        initial,
        method="trust-constr",
        constraints={"type": "eq", "fun": residual},
        bounds=Bounds(-0.01 * np.ones_like(initial), 0.01 * np.ones_like(initial)),
        options={"maxiter": args.max_iterations, "gtol": 1e-12},
    )
    values = nodes(np.asarray(result.x))
    defect = residual(np.asarray(result.x))
    args.output.write_text(
        json.dumps(
            {
                "nodes": args.nodes,
                "chart_dimension": dimension,
                "iterations": int(result.nit),
                "optimizer_converged": bool(result.success),
                "message": str(result.message),
                "rate_acceleration_closure_max_abs": float(np.max(abs(defect))),
                "coordinates": values.tolist(),
                "scope": "Bounded retracted-node preflight only; no marker objective, effort fit, or forward replay.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
