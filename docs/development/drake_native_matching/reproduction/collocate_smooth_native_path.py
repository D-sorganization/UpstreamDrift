"""Construct a C2 native path with Drake pose, rate, and acceleration closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel
from src.shared.python.motion_matching.constrained_trajectory import (
    collocate_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--closure-tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    names = tuple(json.loads(raw)["coordinate_order"])
    source = json.loads(args.path.read_text())
    times = np.asarray(source["times_s"], dtype=float)
    seed = np.asarray(source["coordinates"], dtype=float)
    if seed.shape != (len(times), len(names)):
        raise ValueError("Smooth path coordinates do not match native inventory")
    engine = NativeDrakeModel(args.urdf.read_bytes(), args.sidecar.read_bytes(), raw)
    plant = engine.plant
    indices = [plant.GetJointByName(name).velocity_start() for name in names]
    frame_a = plant.GetFrameByName("native_closure_a")
    frame_b = plant.GetFrameByName("native_closure_b")
    zeros = dict.fromkeys(names, 0.0)

    def mapping(values: np.ndarray) -> dict[str, float]:
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError("Expected finite native coordinate vector")
        return dict(zip(names, values.tolist(), strict=True))

    def state(position: np.ndarray, rate: np.ndarray) -> None:
        engine.accelerations(mapping(position), mapping(rate), zeros)

    def position_closure(position: np.ndarray) -> np.ndarray:
        state(position, np.zeros(len(names)))
        transform = plant.CalcRelativeTransform(engine.context, frame_a, frame_b)
        return np.concatenate(
            (
                transform.translation(),
                Rotation.from_matrix(transform.rotation().matrix()).as_rotvec(),
            )
        )

    def rate_closure(position: np.ndarray, rate: np.ndarray) -> np.ndarray:
        state(position, rate)
        jacobian = plant.CalcJacobianSpatialVelocity(
            engine.context,
            JacobianWrtVariable.kV,
            frame_b,
            np.zeros(3),
            frame_a,
            frame_a,
        )[:, indices]
        spatial = jacobian @ rate
        return np.concatenate((spatial[3:], spatial[:3]))

    def acceleration_closure(
        position: np.ndarray, rate: np.ndarray, acceleration: np.ndarray
    ) -> np.ndarray:
        state(position, rate)
        jacobian = plant.CalcJacobianSpatialVelocity(
            engine.context,
            JacobianWrtVariable.kV,
            frame_b,
            np.zeros(3),
            frame_a,
            frame_a,
        )[:, indices]
        gamma = plant.CalcBiasSpatialAcceleration(
            engine.context,
            JacobianWrtVariable.kV,
            frame_b,
            np.zeros(3),
            frame_a,
            frame_a,
        ).get_coeffs()
        return jacobian @ acceleration + gamma

    result = collocate_positions(
        times,
        seed,
        position_closure,
        rate_closure,
        acceleration_closure,
        max_iterations=args.max_iterations,
        closure_tolerance=args.closure_tolerance,
    )
    report = result.closure
    qualified = (
        result.optimizer_converged
        and max(
            report.position_max_abs,
            report.rate_max_abs,
            report.acceleration_max_abs,
        )
        <= args.closure_tolerance
    )
    args.output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "source_path_sha256": hashlib.sha256(
                    args.path.read_bytes()
                ).hexdigest(),
                "times_s": times.tolist(),
                "coordinates": result.coordinates.tolist(),
                "rates": report.rates.tolist(),
                "accelerations": report.accelerations.tolist(),
                "position_closure_max_abs": report.position_max_abs,
                "rate_closure_max_abs": report.rate_max_abs,
                "acceleration_closure_max_abs": report.acceleration_max_abs,
                "optimizer_converged": result.optimizer_converged,
                "optimizer_message": result.message,
                "iterations": result.iterations,
                "qualified": qualified,
                "scope": "C2 constrained-path construction only; no effort identification or forward replay.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
