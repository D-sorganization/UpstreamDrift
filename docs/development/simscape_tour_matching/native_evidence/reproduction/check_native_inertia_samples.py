"""Diagnose native sensor inertias against reconstructed rigid body groups.

Matches by numeric COM/inertia signature, not signal routing identity. Symmetric
groups can have multiple matching sensors; this is diagnostic evidence only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.shared.python.model_generation.inertia import mcI
from src.shared.python.motion_matching.native_assembly import (
    assemble_native_frames,
    solid_reference,
)
from src.shared.python.motion_matching.native_inventory import uncommented_blocks
from src.shared.python.motion_matching.native_solids import NativeParameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "bindings", "fixture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cut-weld", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    document = json.loads(args.inventory.read_text())
    bindings = json.loads(args.bindings.read_text())["bindings"]
    fixture = json.loads(args.fixture.read_text())
    assembly = assemble_native_frames(document, bindings, args.cut_weld)
    assembly.graph.connect(
        args.cut_weld + ":LConn1", args.cut_weld + ":RConn1", np.eye(4)
    )
    logs = {entry["path"]: entry["initial_value"] for entry in fixture["inertia_logs"]}
    candidates = {
        name: (np.asarray(value), np.asarray(logs[name[:-7] + "COM"]).reshape(3))
        for name, value in logs.items()
        if name.endswith("Inertia") and name[:-7] + "COM" in logs
    }
    results = []
    for path, block in uncommented_blocks(document).items():
        if block["library_reference"] != "sm_lib/Body Elements/Inertia Sensor":
            continue
        parameters = NativeParameters(block)
        if (
            parameters.text("SensorExtent") != "BodyGroup"
            or parameters.text("MeasurementFrame") != "Attached"
        ):
            continue
        poses = assembly.graph.component(path + ":LConn1")
        inertia = np.zeros((3, 3))
        weighted_com = np.zeros(3)
        mass = 0.0
        for solid_path, solid in assembly.solids.items():
            reference = solid_reference(solid_path)
            if reference not in poses:
                continue
            pose = poses[reference]
            rotation = pose[:3, :3]
            com = rotation @ solid.com_m + pose[:3, 3]
            inertia += mcI(
                solid.mass_kg, com, rotation @ solid.inertia_com_kg_m2 @ rotation.T
            )[:3, :3]
            mass += solid.mass_kg
            weighted_com += solid.mass_kg * com
        if mass <= 0:
            raise ValueError(f"Empty native sensor group {path}")
        com = weighted_com / mass
        matches = [
            name
            for name, (native_inertia, native_com) in candidates.items()
            if np.allclose(inertia, native_inertia, atol=1e-10, rtol=0)
            and np.allclose(com, native_com, atol=1e-10, rtol=0)
        ]
        results.append(
            {
                "sensor": path,
                "mass_kg": mass,
                "com": com.tolist(),
                "inertia": inertia.tolist(),
                "numeric_signature_matches": matches,
            }
        )
    args.output.write_text(json.dumps({"sensors": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
