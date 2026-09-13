"""Qualify one bounded Pinocchio weld-node chart retraction on ControlTower."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--chart-step", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.chart_step) or args.chart_step <= 0:
        raise ValueError("Output must be new and chart step positive")
    raw = args.model.read_bytes()
    specification = json.loads(raw)
    names = tuple(specification["coordinate_order"])
    path = json.loads(args.path.read_text())
    samples = np.asarray(path["coordinates"], dtype=float)
    if (
        samples.ndim != 2
        or samples.shape[1] != len(names)
        or not 0 <= args.sample < len(samples)
    ):
        raise ValueError("Path sample does not match native coordinate inventory")
    primitives = {
        item["coordinate"]: item["primitive"]
        for joint in specification["joints"]
        for item in joint["primitives"]
    }
    scales = np.asarray(
        [0.1 if primitives[name].startswith("P") else 1.0 for name in names],
        dtype=float,
    )
    model = NativePinocchioModel(specification)

    def mapping(values: np.ndarray) -> dict[str, float]:
        return dict(zip(names, values.tolist(), strict=True))

    reference = samples[args.sample]
    linearization = model.closure_position_linearization(mapping(reference))
    basis = scaled_tangent_basis(linearization.jacobian, scales)
    chart = np.zeros(basis.shape[1])
    chart[0] = args.chart_step

    def closure(values: np.ndarray) -> np.ndarray:
        return model.closure_position_linearization(mapping(values)).position

    def jacobian(values: np.ndarray) -> np.ndarray:
        return model.closure_position_linearization(mapping(values)).jacobian

    result = retract_node(
        reference,
        basis,
        chart,
        closure,
        jacobian,
        state_scales=scales,
        residual_scales=np.ones(6),
        radius=0.1,
    )
    args.output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "path_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
                "sample": args.sample,
                "coordinate_count": len(names),
                "chart_dimension": basis.shape[1],
                "chart_step": args.chart_step,
                "closure_max_abs": result.closure_max_abs,
                "scaled_displacement": result.scaled_displacement,
                "state_jacobian_shape": list(result.state_jacobian.shape),
                "scope": "Local weld-node chart qualification only; no trajectory, effort identification, or forward replay.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
