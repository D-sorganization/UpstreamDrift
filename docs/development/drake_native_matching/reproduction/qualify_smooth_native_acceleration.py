"""Check Drake weld acceleration compatibility of a native smooth-path receipt."""

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
    specification = json.loads(raw)
    names = tuple(specification["coordinate_order"])
    path = json.loads(args.path.read_text())
    q = np.asarray(path["coordinates"], dtype=float)
    qd = np.asarray(path["rates"], dtype=float)
    qdd = np.asarray(path["finite_difference_accelerations"], dtype=float)
    if q.shape != qd.shape or q.shape != qdd.shape or q.shape[1:] != (len(names),):
        raise ValueError("Smooth path state shape does not match native coordinates")
    engine = NativeDrakeModel(args.urdf.read_bytes(), args.sidecar.read_bytes(), raw)
    plant = engine.plant
    indices = [plant.GetJointByName(name).velocity_start() for name in names]
    frame_a = plant.GetFrameByName("native_closure_a")
    frame_b = plant.GetFrameByName("native_closure_b")
    residuals = []
    for position, rate, acceleration in zip(q, qd, qdd, strict=True):
        engine.accelerations(
            dict(zip(names, position, strict=True)),
            dict(zip(names, rate, strict=True)),
            dict.fromkeys(names, 0.0),
        )
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
        residuals.append(jacobian @ acceleration + gamma)
    residual = np.asarray(residuals)
    args.output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "path_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
                "samples": len(residual),
                "acceleration_closure_max_abs": float(np.max(abs(residual))),
                "acceleration_closure_rms": float(np.sqrt(np.mean(residual**2))),
                "scope": "Independent Drake acceleration-compatibility diagnostic of finite-difference native qdd; not inverse dynamics, torque identification, or forward validation.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
