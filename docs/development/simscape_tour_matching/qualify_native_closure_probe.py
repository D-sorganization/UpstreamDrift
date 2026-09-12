"""Qualify the native single-state weld-residual probe on fixed input bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import NativePinocchioModel
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def _values(names: tuple[str, ...], values: object) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(names),) or not np.isfinite(array).all():
        raise ValueError("Candidate state does not match native coordinate inventory")
    return dict(zip(names, array.tolist(), strict=True))


def _maximum_difference(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Closure probe returned invalid residuals")
    return float(np.max(np.abs(first - second)))


def _direct_closure(
    model: NativePinocchioModel,
    coordinates: dict[str, float],
    rates: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    model.accelerations(coordinates, rates, dict.fromkeys(coordinates, 0.0))
    return model.closure_errors()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    specification = json.loads(raw)
    names = tuple(specification["coordinate_order"])
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    coordinates = _values(names, candidate.document["q0"])
    rates = _values(names, candidate.document["qd0"])
    model = NativePinocchioModel(specification)

    probe_pose, probe_rate = model.closure_residuals(coordinates, rates)
    direct_pose, direct_rate = _direct_closure(model, coordinates, rates)
    zero_probe_pose, zero_probe_rate = model.closure_residuals(coordinates)
    zero = dict.fromkeys(names, 0.0)
    zero_direct_pose, zero_direct_rate = _direct_closure(model, coordinates, zero)
    result = {
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_sha256": candidate.sha256,
        "recorded_rate": {
            "pose_max_abs_difference": _maximum_difference(probe_pose, direct_pose),
            "rate_max_abs_difference": _maximum_difference(probe_rate, direct_rate),
        },
        "zero_rate": {
            "pose_max_abs_difference": _maximum_difference(
                zero_probe_pose, zero_direct_pose
            ),
            "rate_max_abs_difference": _maximum_difference(
                zero_probe_rate, zero_direct_rate
            ),
        },
    }
    if max(value for group in result.values() if isinstance(group, dict) for value in group.values()) > 1e-14:
        raise ValueError("Closure probe differs from direct constrained dynamics")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
