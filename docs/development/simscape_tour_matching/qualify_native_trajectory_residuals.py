"""Evaluate all Pinocchio native weld residual levels along a sampled path."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    specification = json.loads(raw)
    names = tuple(specification["coordinate_order"])
    path = json.loads(args.path.read_text())
    position, rate, acceleration = (
        np.asarray(path[key], dtype=float)
        for key in ("coordinates", "rates", "finite_difference_accelerations")
    )
    if (
        position.shape != rate.shape
        or position.shape != acceleration.shape
        or position.ndim != 2
        or position.shape[1] != len(names)
    ):
        raise ValueError("Path does not match native state inventory")
    model = NativePinocchioModel(specification)

    def mapping(values: np.ndarray) -> dict[str, float]:
        return dict(zip(names, values.tolist(), strict=True))

    residuals = [
        model.closure_trajectory_residuals(mapping(q), mapping(v), mapping(a))
        for q, v, a in zip(position, rate, acceleration, strict=True)
    ]
    args.output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "path_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
                "samples": len(residuals),
                "position_closure_max_abs": float(
                    max(np.max(abs(value.position)) for value in residuals)
                ),
                "rate_closure_max_abs": float(
                    max(np.max(abs(value.rate)) for value in residuals)
                ),
                "acceleration_closure_max_abs": float(
                    max(np.max(abs(value.acceleration)) for value in residuals)
                ),
                "scope": "Native residual-oracle qualification only; no trajectory correction, effort identification, or forward replay.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
