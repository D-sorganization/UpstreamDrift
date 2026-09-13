"""Project smooth-path accelerations onto Drake's instantaneous weld constraint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable

from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    names = tuple(json.loads(raw)["coordinate_order"])
    path = json.loads(args.path.read_text())
    q, qd, qdd = (
        np.asarray(path[key], dtype=float)
        for key in ("coordinates", "rates", "finite_difference_accelerations")
    )
    if q.shape != qd.shape or q.shape != qdd.shape or q.shape[1:] != (len(names),):
        raise ValueError("Smooth path state shape does not match native coordinates")
    engine = NativeDrakeModel(args.urdf.read_bytes(), args.sidecar.read_bytes(), raw)
    plant = engine.plant
    indices = [plant.GetJointByName(name).velocity_start() for name in names]
    frame_a = plant.GetFrameByName("native_closure_a")
    frame_b = plant.GetFrameByName("native_closure_b")
    corrected = []
    before = []
    after = []
    corrections = []
    for position, rate, acceleration in zip(q, qd, qdd, strict=True):
        engine.accelerations(
            dict(zip(names, position, strict=True)),
            dict(zip(names, rate, strict=True)),
            dict.fromkeys(names, 0.0),
        )
        jacobian = plant.CalcJacobianSpatialVelocity(
            engine.context, JacobianWrtVariable.kV, frame_b, np.zeros(3), frame_a, frame_a
        )[:, indices]
        gamma = plant.CalcBiasSpatialAcceleration(
            engine.context, JacobianWrtVariable.kV, frame_b, np.zeros(3), frame_a, frame_a
        ).get_coeffs()
        residual = jacobian @ acceleration + gamma
        delta, _, rank, _ = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)
        if rank != jacobian.shape[0]:
            raise ValueError("Weld acceleration Jacobian lost rank")
        projected = acceleration + delta
        corrected.append(projected)
        before.append(residual)
        after.append(jacobian @ projected + gamma)
        corrections.append(delta)
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    correction_array = np.asarray(corrections)
    args.output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "path_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
                "projected_accelerations": np.asarray(corrected).tolist(),
                "before_max_abs": float(np.max(abs(before_array))),
                "after_max_abs": float(np.max(abs(after_array))),
                "correction_max_abs": float(np.max(abs(correction_array))),
                "correction_rms": float(np.sqrt(np.mean(correction_array**2))),
                "scope": "Minimum-norm instantaneous acceleration projection; it does not establish derivative consistency, effort identification, or forward motion.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
