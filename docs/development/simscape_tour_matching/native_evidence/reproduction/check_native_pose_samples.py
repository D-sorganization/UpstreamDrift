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
    passed = all(
        row["translation_max_m"] <= 1e-7 and row["rotation_matrix_max_error"] <= 1e-7
        for row in results
    )
    receipt = {
        "qualification": "multi-pose FK only; force/rollout parity unqualified",
        "passed": passed,
        "samples": results,
        "sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("module", "spec", "fixture")
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not passed:
        raise AssertionError("Native multi-pose FK parity failed; inspect receipt")


if __name__ == "__main__":
    main()
