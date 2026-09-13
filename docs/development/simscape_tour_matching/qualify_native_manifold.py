"""Qualify alternate Pinocchio coordinates at saved feedback swing states."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_manifold_model import (
    NativeManifoldPinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.model.read_text())
    native = NativePinocchioModel(specification)
    manifold = NativeManifoldPinocchioModel(specification)
    pin = manifold.pin
    names = specification["coordinate_order"]
    rows = []
    with np.load(args.states, allow_pickle=False) as saved:
        for requested_time in (0.6, 0.9, 1.3):
            index = int(np.argmin(abs(saved["time"] - requested_time)))
            state, effort = saved["state"][index], saved["primitive_efforts"][index]
            q, v, tau = (
                dict(zip(names, values, strict=True))
                for values in (state[:27], state[27:], effort)
            )
            start = time.perf_counter()
            mq, mv, mtau = manifold.native_state(q, v, tau)
            poses, expected_poses = manifold.frame_poses(mq), native.frame_poses(q)
            pose_error = max(
                float(np.max(abs(poses[n] - expected_poses[n]))) for n in poses
            )
            sq, sv = native.configuration(q), native._velocity_vector(v)
            energy = pin.computeKineticEnergy(manifold.model, manifold.data, mq, mv)
            expected_energy = pin.computeKineticEnergy(
                native.model, native.data, sq, sv
            )
            expected_power = sum(v[n] * tau[n] for n in names)
            power_error = abs(float(mv @ mtau) - expected_power)
            pin.forwardKinematics(native.model, native.data, sq, sv)
            velocities = manifold.frame_velocities(mq, mv)
            velocity_error = max(
                float(
                    np.max(
                        abs(
                            velocities[n]
                            - pin.getFrameVelocity(
                                native.model,
                                native.data,
                                j,
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                            ).vector
                        )
                    )
                )
                for n, j in native._frames.items()
            )
            native_a = native.accelerations(q, v, tau)
            # A displaced reference must select a branch, never supply the
            # configuration used for torque conversion.
            reference = {n: value + 0.02 for n, value in q.items()}
            a = manifold.acceleration_from_native_efforts(mq, mv, tau, reference)
            actual_a = manifold.adapter.restore(manifold._native_state(mq, mv, a), q)[2]
            expected = np.array([native_a[n] for n in names])
            actual = np.array([actual_a[n] for n in names])
            acceleration_error = float(np.max(abs(actual - expected)))
            scaled_acceleration_error = float(
                np.max(abs(actual - expected) / (2e-7 + 2e-8 * abs(expected)))
            )
            pose, rate = native.closure_errors()
            contact = manifold.constraint_data[0]
            closure_pose_error = float(
                np.max(abs(contact.contact_placement_error.vector - pose))
            )
            closure_rate_error = float(
                np.max(abs(contact.contact_velocity_error.vector - rate))
            )
            energy_error = abs(float(energy - expected_energy))
            passed = bool(
                pose_error < 1e-10
                and velocity_error < 1e-9
                and energy_error < 1e-11 * max(1, abs(expected_energy))
                and power_error < 1e-11 * max(1, abs(expected_power))
                and scaled_acceleration_error <= 1
                and closure_pose_error < 1e-10
                and closure_rate_error < 1e-9
            )
            rows.append(
                {
                    "requested_time_s": requested_time,
                    "actual_time_s": float(saved["time"][index]),
                    "frame_index": index,
                    "max_transform_error": pose_error,
                    "max_frame_velocity_error": velocity_error,
                    "kinetic_energy_error_j": energy_error,
                    "power_error_w": power_error,
                    "native_acceleration_max_error": acceleration_error,
                    "native_acceleration_scaled_gate": scaled_acceleration_error,
                    "closure_pose_error": closure_pose_error,
                    "closure_rate_error": closure_rate_error,
                    "elapsed_seconds": time.perf_counter() - start,
                    "passed": passed,
                }
            )
    report = {
        "status": "passed" if all(r["passed"] for r in rows) else "failed",
        "pinocchio_version": pin.__version__,
        "specification_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "states_sha256": hashlib.sha256(args.states.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "states_source": str(args.states),
        "thresholds": {
            "transform_abs": 1e-10,
            "frame_velocity_abs": 1e-9,
            "kinetic_energy_scaled": 1e-11,
            "power_scaled": 1e-11,
            "native_acceleration_atol": 2e-7,
            "native_acceleration_rtol": 2e-8,
            "closure_pose_abs": 1e-10,
            "closure_rate_abs": 1e-9,
        },
        "rows": rows,
        "scope": "Pointwise saved feedback-state equivalence only; neither open-loop trajectory nor Simscape R2025b acceptance.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise RuntimeError("Saved-state manifold qualification failed; inspect report")


if __name__ == "__main__":
    main()
