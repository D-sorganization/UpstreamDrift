"""Qualify Pinocchio's native weld pose Jacobian against centered differences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--step", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.step) or args.step <= 0:
        raise ValueError("Output must be new and finite difference step positive")
    raw = args.model.read_bytes()
    names = tuple(json.loads(raw)["coordinate_order"])
    path = json.loads(args.path.read_text())
    coordinates = np.asarray(path["coordinates"], dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != len(names)
        or not 0 <= args.sample < coordinates.shape[0]
    ):
        raise ValueError("Path sample does not match native coordinate inventory")
    state = coordinates[args.sample]
    model = NativePinocchioModel(json.loads(raw))

    def mapping(values: np.ndarray) -> dict[str, float]:
        return dict(zip(names, values.tolist(), strict=True))

    linearization = model.closure_position_linearization(mapping(state))
    differences = []
    for index in range(len(names)):
        plus, minus = state.copy(), state.copy()
        plus[index] += args.step
        minus[index] -= args.step
        residual_plus, _ = model.closure_residuals(mapping(plus))
        residual_minus, _ = model.closure_residuals(mapping(minus))
        differences.append(
            (residual_plus - residual_minus) / (2 * args.step)
            - linearization.jacobian[:, index]
        )
    error = np.asarray(differences, dtype=float)
    args.output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "path_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
                "sample": args.sample,
                "finite_difference_step": args.step,
                "jacobian_shape": list(linearization.jacobian.shape),
                "position_closure_max_abs": float(np.max(abs(linearization.position))),
                "finite_difference_max_abs": float(np.max(abs(error))),
                "scope": "Kinematic weld Jacobian qualification only; no trajectory, effort identification, or forward replay.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
