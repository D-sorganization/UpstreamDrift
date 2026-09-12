"""Project same-clock engine trajectories through the qualified native FK map."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "module",
        "projection",
        "spec",
        "seed",
        "reference",
        "replay",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = {
        name: json.loads(getattr(args, name).read_text())
        for name in ("spec", "seed", "reference", "replay")
    }
    names = data["spec"]["coordinate_order"]
    for name in ("seed", "reference", "replay"):
        if data[name]["coordinate_names"] != names:
            raise ValueError("Coordinate inventory mismatch")
    clock = np.asarray(data["replay"]["time_s"])
    trajectory = data["reference"]["trajectory"]
    np.testing.assert_array_equal(clock, trajectory["time_s"])
    model = runpy.run_path(str(args.module))["NativePinocchioModel"](data["spec"])
    project = runpy.run_path(str(args.projection))["project_markers"]
    seed = data["seed"]
    differences = []
    for ref, pred in zip(trajectory["q"], data["replay"]["q"], strict=True):
        positions = [
            project(
                model.frame_poses(dict(zip(names, q, strict=True))),
                seed["body_names"],
                seed["offsets_m"],
            )
            for q in (ref, pred)
        ]
        differences.append(np.linalg.norm(positions[1] - positions[0], axis=1))
    error = np.asarray(differences)
    receipt = {
        "qualification": "same-clock Cartesian discrepancy through shared qualified FK; not independent native marker validation or C3D fit",
        "samples": len(clock),
        "duration_s": float(clock[-1]),
        "labels": seed["labels"],
        "maximum_marker_distance_m": float(error.max()),
        "max_distance_by_marker_m": error.max(axis=0).tolist(),
        "time_weighted_marker_rms_m": float(
            np.sqrt(
                np.trapezoid(np.mean(error**2, axis=1), clock) / (clock[-1] - clock[0])
            )
        ),
        "terminal_marker_distances_m": error[-1].tolist(),
        "sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("module", "projection", "spec", "seed", "reference", "replay")
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
