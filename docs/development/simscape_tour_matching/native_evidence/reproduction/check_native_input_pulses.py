"""Compare native stationary effort-response matrices with actual Pinocchio."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("module", "spec", "cases", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--require-parity", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loader = importlib.util.spec_from_file_location("native_port", args.module)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    spec = json.loads(args.spec.read_text())
    model = module.NativePinocchioModel(spec)
    names = spec["coordinate_order"]
    native, predicted, coordinates, rates, hashes = [], [], [], [], {}
    for index in range(len(names) + 1):
        path = args.cases / f"case-{index:02d}.json"
        case = json.loads(path.read_text())
        if case["release"] != "2025b" or case["coordinate_names"] != names:
            raise ValueError("Native release or coordinate identity mismatch")
        if case["case_index"] != index or case["time_s"] != 0:
            raise ValueError("Require ordered initial-state pulse cases")
        coordinate = dict(zip(names, case["q"], strict=True))
        velocity = dict(zip(names, case["qd"], strict=True))
        effort = dict(zip(names, case["primitive_efforts"], strict=True))
        acceleration = model.accelerations(coordinate, velocity, effort)
        native.append(case["qdd"])
        predicted.append([acceleration[name] for name in names])
        coordinates.append(case["q"])
        rates.append(case["qd"])
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    native = np.asarray(native)
    predicted = np.asarray(predicted)
    native_response = (native[1:] - native[0]).T
    pin_response = (predicted[1:] - predicted[0]).T
    difference = pin_response - native_response
    row_scale = np.maximum(1.0, np.max(np.abs(native_response), axis=1))
    result = {
        "qualification": "stationary input-response diagnostic; rollout unqualified",
        "coordinate_names": names,
        "native_baseline_qdd": native[0].tolist(),
        "pinocchio_baseline_qdd": predicted[0].tolist(),
        "baseline_max_abs_error": float(np.max(np.abs(predicted[0] - native[0]))),
        "response_scaled_max_error": float(
            np.max(np.abs(difference) / row_scale[:, None])
        ),
        "native_response_matrix": native_response.tolist(),
        "pinocchio_response_matrix": pin_response.tolist(),
        "assembled_q_max_variation": float(
            np.max(np.abs(np.asarray(coordinates) - coordinates[0]))
        ),
        "assembled_rate_max_abs": float(np.max(np.abs(rates))),
        "case_sha256": hashes,
        "module_sha256": hashlib.sha256(args.module.read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
    }
    result["passed"] = bool(
        result["baseline_max_abs_error"] <= 1e-7
        and result["response_scaled_max_error"] <= 1e-7
        and result["assembled_q_max_variation"] <= 1e-10
        and result["assembled_rate_max_abs"] <= 1e-10
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.require_parity and not result["passed"]:
        raise AssertionError("Native stationary input-response parity failed")


if __name__ == "__main__":
    main()
