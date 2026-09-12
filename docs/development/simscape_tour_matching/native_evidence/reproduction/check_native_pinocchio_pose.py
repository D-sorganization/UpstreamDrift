"""Verify real Pinocchio FK against the qualified native initial marker pose."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-free-fall", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loader = importlib.util.spec_from_file_location(
        "native_pinocchio_port", args.module
    )
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    spec = json.loads(args.spec.read_text())
    seed = json.loads(args.seed.read_text())
    native = module.NativePinocchioModel(spec)
    coordinates = dict(zip(seed["coordinate_names"], seed["q"], strict=True))
    poses = native.frame_poses(coordinates)
    predicted = np.array(
        [
            poses[body][:3, :3] @ np.asarray(offset) + poses[body][:3, 3]
            for body, offset in zip(seed["body_names"], seed["offsets_m"], strict=True)
        ]
    )
    trajectory = np.asarray(seed["prediction_m"])
    if trajectory.ndim != 3 or trajectory.shape[1:] != predicted.shape:
        raise ValueError("Expected native prediction as time-by-marker-by-xyz")
    reference = trajectory[0]
    error = predicted - reference
    result = {
        "qualification": "initial-pose FK only; dynamics unqualified",
        "native_sample_index": 0,
        "nq": native.model.nq,
        "nv": native.model.nv,
        "marker_rms_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "marker_max_coordinate_error_m": float(np.max(np.abs(error))),
        "predicted_m": predicted.tolist(),
        "native_reference_m": reference.tolist(),
    }
    if args.check_free_fall:
        zeros = dict.fromkeys(spec["coordinate_order"], 0.0)
        acceleration = native.accelerations(coordinates, zeros, zeros)
        base = spec["joints"][0]
        assert [p["primitive"] for p in base["primitives"]] == [
            "Px",
            "Py",
            "Pz",
            "Rx",
            "Ry",
            "Rz",
        ]
        expected = zeros.copy()
        gravity = np.asarray(base["parent_to_base"])[:3, :3].T @ np.asarray(
            spec["gravity_m_s2"]
        )
        for primitive, value in zip(base["primitives"][:3], gravity, strict=True):
            expected[primitive["coordinate"]] = float(value)
        difference = np.array(
            [acceleration[name] - expected[name] for name in expected]
        )
        result["free_fall_max_acceleration_error"] = float(np.max(np.abs(difference)))
        result["qualification"] = (
            "initial FK and free-fall invariant only; native force parity unqualified"
        )
        np.testing.assert_allclose(difference, 0, atol=1e-7, rtol=0)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.testing.assert_allclose(predicted, reference, atol=1e-7, rtol=0)


if __name__ == "__main__":
    main()
