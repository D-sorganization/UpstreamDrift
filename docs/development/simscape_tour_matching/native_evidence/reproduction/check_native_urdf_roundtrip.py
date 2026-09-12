"""Actual URDF-derived closed dynamics versus the qualified native-derived tree."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_urdf_model import (
    NativeUrdfModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "urdf",
        "sidecar",
        "model",
        "candidate",
        "fixture",
        "reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    xml = args.urdf.read_bytes()
    side = args.sidecar.read_bytes()
    spec = json.loads(raw)
    fixture = json.loads(args.fixture.read_text())
    original = NativePinocchioModel(spec)
    loaded = NativeUrdfModel(xml, side, raw)
    names = spec["coordinate_order"]

    def mapping(row: list) -> dict:
        return dict(zip(names, row, strict=True))

    pose_errors = []
    acceleration_errors = []
    for q, v, tau in zip(
        fixture["q"], fixture["qd"], fixture["primitive_efforts"], strict=True
    ):
        expected = original.frame_poses(mapping(q))
        observed = loaded.frame_poses(mapping(q))
        pose_errors.append(
            max(float(np.max(np.abs(expected[n] - observed[n]))) for n in expected)
        )
        a = original.accelerations(mapping(q), mapping(v), mapping(tau))
        b = loaded.accelerations(mapping(q), mapping(v), mapping(tau))
        acceleration_errors.append(
            max(abs(a[n] - b[n]) / (1 + abs(a[n])) for n in names)
        )
    q0 = mapping(fixture["q"][0])
    zero = dict.fromkeys(names, 0.0)
    pulse_errors = []
    for name in [None] + names:
        effort = zero.copy()
        if name is not None:
            effort[name] = 1.0
        a = original.accelerations(q0, zero, effort)
        b = loaded.accelerations(q0, zero, effort)
        pulse_errors.append(max(abs(a[n] - b[n]) / (1 + abs(a[n])) for n in names))
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    reference = json.loads(args.reference.read_text())
    result = replay_candidate(
        raw,
        candidate,
        np.asarray(reference["time_s"]),
        model_factory=lambda _: NativeUrdfModel(xml, side, raw),
        rtol=1e-11,
        atol=1e-13,
        max_step=0.00025,
    )
    n = len(names)
    q_error = float(np.max(np.abs(result.integration.state[:, :n] - reference["q"])))
    v_error = float(np.max(np.abs(result.integration.state[:, n:] - reference["qd"])))
    com = []
    for model in (original, loaded):
        com.append(
            pin.centerOfMass(model.model, model.data, model.configuration(q0)).copy()
        )
    receipt = {
        "qualification": "URDF plus sidecar round-trip diagnostic; full-swing unqualified",
        "mass_kg": float(sum(i.mass for i in loaded.model.inertias)),
        "com_max_difference_m": float(np.max(np.abs(com[1] - com[0]))),
        "frame_matrix_max_differences": pose_errors,
        "acceleration_scaled_max_differences": acceleration_errors,
        "pulse_scaled_max_differences": pulse_errors,
        "q_max_difference": q_error,
        "qd_max_difference": v_error,
        "integration_s": result.integration.elapsed_s,
        "duration_s": float(result.integration.time[-1]),
        "closure_pose_max_abs": result.closure_pose_max_abs,
        "closure_velocity_max_abs": result.closure_velocity_max_abs,
        "input_sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in (
                "urdf",
                "sidecar",
                "model",
                "candidate",
                "fixture",
                "reference",
            )
        },
    }
    receipt["passed"] = (
        max(pose_errors) < 1e-10
        and max(acceleration_errors + pulse_errors) < 1e-7
        and q_error < 1e-5
        and v_error < 0.01
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["passed"]:
        raise AssertionError("URDF round-trip diagnostic failed")


if __name__ == "__main__":
    main()
