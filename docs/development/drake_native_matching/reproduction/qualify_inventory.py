"""Qualify explicit native physical inventory and COM independently of tracking."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--reference", type=Path)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
raw = (a.assets / "native_geometry_spec_9967.json").read_bytes()
spec = json.loads(raw)
names = spec["coordinate_order"]
state = np.load(a.assets / "drake-native-reference-10022-01/reference.npz")["state"][
    [0, 40, 60, 70, 75, 80]
]
report = {
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "coordinate_order": names,
    "frame_names": [f["name"] for f in spec["frames"]],
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
if a.reference is None:
    import pinocchio as pin
    from src.engines.physics_engines.pinocchio.python.native_model import (
        NativePinocchioModel,
    )

    engine = NativePinocchioModel(spec)
    report["com_world_m"] = [
        pin.centerOfMass(
            engine.model,
            engine.data,
            engine.configuration(dict(zip(names, row[:27], strict=True))),
        )
        .copy()
        .tolist()
        for row in state
    ]
    report["total_mass_kg"] = float(pin.computeTotalMass(engine.model))
else:
    from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel

    meta = json.loads((a.assets / "native-golf-9967-01.sidecar.json").read_bytes())
    engine = NativeDrakeModel(
        (a.assets / "native-golf-9967-01.urdf").read_bytes(),
        (a.assets / "native-golf-9967-01.sidecar.json").read_bytes(),
        raw,
    )
    report["com_world_m"] = []
    for row in state:
        engine.frame_poses(dict(zip(names, row[:27], strict=True)))
        report["com_world_m"].append(
            engine.plant.CalcCenterOfMassPositionInWorld(engine.context).tolist()
        )
    report["total_mass_kg"] = engine.plant.CalcTotalMass(engine.context)
    inertia_errors = []
    mass_errors = []
    com_errors = []
    for group in spec["bodies"]:
        for solid in group["solids"]:
            body = engine.plant.GetBodyByName(meta["solid_links"][solid["name"]])
            mass = body.default_mass()
            com = body.default_com()
            inertia = body.default_rotational_inertia().CopyToFullMatrix3() - mass * (
                np.dot(com, com) * np.eye(3) - np.outer(com, com)
            )
            mass_errors.append(abs(mass - solid["mass_kg"]))
            com_errors.append(np.max(np.abs(com - solid["com_m"])))
            inertia_errors.append(np.max(np.abs(inertia - solid["inertia_com_kg_m2"])))
    report["solid_count"] = len(mass_errors)
    report["solid_mass_max_abs"] = float(max(mass_errors))
    report["solid_com_max_abs"] = float(max(com_errors))
    report["solid_inertia_com_max_abs"] = float(max(inertia_errors))
    report["damping_max_abs"] = float(
        max(
            np.max(
                np.abs(
                    engine.plant.GetJointByName(name).GetDampingVector(engine.context)
                )
            )
            for name in names
        )
    )
    report["unbounded_positions"] = bool(
        np.isneginf(engine.plant.GetPositionLowerLimits()).all()
        and np.isposinf(engine.plant.GetPositionUpperLimits()).all()
    )
    report["unbounded_velocities"] = bool(
        np.isneginf(engine.plant.GetVelocityLowerLimits()).all()
        and np.isposinf(engine.plant.GetVelocityUpperLimits()).all()
    )
    report["gravity_m_s2"] = engine.plant.gravity_field().gravity_vector().tolist()
    reference = json.loads(a.reference.read_bytes())
    report["com_max_abs_m"] = float(
        np.max(np.abs(np.asarray(report["com_world_m"]) - reference["com_world_m"]))
    )
    report["mass_difference_kg"] = abs(
        report["total_mass_kg"] - reference["total_mass_kg"]
    )
    report["passed"] = (
        all(
            report[key] < 1e-12
            for key in (
                "solid_mass_max_abs",
                "solid_com_max_abs",
                "solid_inertia_com_max_abs",
                "com_max_abs_m",
                "mass_difference_kg",
            )
        )
        and report["damping_max_abs"] == 0
        and report["unbounded_positions"]
        and report["unbounded_velocities"]
        and report["solid_count"] == 31
    )
    report["execution_mode"] = "drake-tree-custom-rigid-kkt"
a.output.write_text(json.dumps(report, indent=2))
sys.stdout.write(json.dumps(report) + "\n")
if a.reference and not report["passed"]:
    raise ValueError("Physical inventory mismatch")
