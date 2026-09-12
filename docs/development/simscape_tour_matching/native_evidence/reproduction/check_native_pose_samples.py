"""Compare actual Pinocchio frame poses to exact-state R2025b fixtures."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("module", "spec", "fixture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--check-accelerations", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loader = importlib.util.spec_from_file_location("native_port", args.module)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    spec = json.loads(args.spec.read_text())
    fixture = json.loads(args.fixture.read_text())
    if fixture["release"] != "2025b":
        raise ValueError("Require native R2025b evidence")
    model = module.NativePinocchioModel(spec)
    import pinocchio as pin

    initial_q = model.configuration(
        dict(zip(fixture["coordinate_names"], fixture["q"][0], strict=True))
    )
    initial_com = pin.centerOfMass(model.model, model.data, initial_q).copy()
    expected = np.asarray(fixture["poses"])
    results = []
    for index, q in enumerate(fixture["q"]):
        poses = model.frame_poses(
            dict(zip(fixture["coordinate_names"], q, strict=True))
        )
        observed = np.array([poses[name] for name in fixture["frame_names"]])
        error = observed - expected[index]
        results.append(
            {
                "time_s": fixture["time_s"][index],
                "translation_max_m": float(np.max(np.abs(error[:, :3, 3]))),
                "rotation_matrix_max_error": float(np.max(np.abs(error[:, :3, :3]))),
            }
        )
        if args.check_accelerations:
            names = fixture["coordinate_names"]
            acceleration = model.accelerations(
                dict(zip(names, q, strict=True)),
                dict(zip(names, fixture["qd"][index], strict=True)),
                dict(zip(names, fixture["primitive_efforts"][index], strict=True)),
            )
            reference = np.asarray(fixture["qdd"][index])
            delta = np.array([acceleration[name] for name in names]) - reference
            results[-1]["acceleration_max_abs_error"] = float(np.max(np.abs(delta)))
            results[-1]["acceleration_scaled_max_error"] = float(
                np.max(np.abs(delta) / (1.0 + np.abs(reference)))
            )
            results[-1]["acceleration_difference"] = delta.tolist()
            contact = model.constraint_data[0]
            jacobian = pin.getConstraintJacobian(
                model.model, model.data, model.constraints[0], contact
            )
            ordered_delta = np.zeros(model.model.nv)
            for name, value in zip(names, delta, strict=True):
                joint = model.model.joints[model.model.getJointId(name)]
                ordered_delta[joint.idx_v] = value
            results[-1]["closure_acceleration_difference"] = (
                jacobian @ ordered_delta
            ).tolist()
            results[-1]["closure_pose_error"] = (
                contact.contact_placement_error.vector.tolist()
            )
            results[-1]["closure_velocity_error"] = (
                contact.contact_velocity_error.vector.tolist()
            )
    passed = all(
        row["translation_max_m"] <= 1e-7 and row["rotation_matrix_max_error"] <= 1e-7
        for row in results
    )
    receipt = {
        "qualification": "multi-pose FK only; force/rollout parity unqualified",
        "passed": passed,
        "initial_com_world_m": initial_com.tolist(),
        "total_mass_kg": float(sum(inertia.mass for inertia in model.model.inertias)),
        "samples": results,
        "sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("module", "spec", "fixture")
        },
    }
    if args.check_accelerations:
        # Diagnostic gate; full qualification additionally requires pulse and
        # native solver convergence studies, then continuous rollout parity.
        acceleration_passed = all(
            row["acceleration_scaled_max_error"] <= 1e-4 for row in results
        )
        receipt["acceleration_passed"] = acceleration_passed
        receipt["qualification"] = (
            "same-state FK and acceleration diagnostic; rollout parity unqualified"
        )
        passed = passed and acceleration_passed
        receipt["passed"] = passed
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not passed:
        raise AssertionError(
            "Native pose/acceleration diagnostic failed; inspect receipt"
        )


if __name__ == "__main__":
    main()
