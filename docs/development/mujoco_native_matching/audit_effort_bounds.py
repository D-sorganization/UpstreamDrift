"""Read-only effort correction audit; no optimization or dynamics execution."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
    recover_native_bernstein,
)
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile


def extrema(coefficients: np.ndarray, horizon: float) -> dict[str, float]:
    """Evaluate endpoints and real derivative roots in the closed interval."""
    coefficients = np.asarray(coefficients, dtype=float)
    if (
        coefficients.ndim != 1
        or not coefficients.size
        or not np.isfinite(coefficients).all()
        or not np.isfinite(horizon)
        or horizon <= 0
    ):
        raise ValueError("Finite polynomial and positive horizon required")
    roots = np.roots(np.polyder(coefficients))
    times = [0.0, horizon] + [
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-10 and 0 < root.real < horizon
    ]
    values = np.polyval(coefficients, times)
    low, high = int(np.argmin(values)), int(np.argmax(values))
    return {
        "minimum": float(values[low]),
        "minimum_time_s": times[low],
        "maximum": float(values[high]),
        "maximum_time_s": times[high],
        "maximum_absolute": float(np.max(np.abs(values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "parent", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.run / "config.json",
        args.run / "returned-candidate.json",
        args.run / "bound-audit.json",
        args.parent,
        args.model,
        Path(__file__),
    ]
    config, document, previous, parent, spec = [
        json.loads(path.read_bytes()) for path in paths[:5]
    ]
    names = spec["coordinate_order"]
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    original = NativeReplayCandidate.from_document(parent, names, model_hash)
    # Coverage is metadata, not polynomial time rescaling. This must match run config.
    parent["duration_s"] = document["duration_s"]
    base = NativeReplayCandidate.from_document(parent, names, model_hash)
    candidate = NativeReplayCandidate.from_document(document, names, model_hash)
    if (
        base.sha256 != config["base"]
        or candidate.sha256 != previous["candidate_sha256"]
        or config["first_control"] != 0
    ):
        raise ValueError("Run identities or effort layout do not match")
    controls = recover_native_bernstein(
        base, candidate, basis_duration_s=config["basis_duration_s"]
    )
    restored = increment_native_bernstein(
        base, controls, basis_duration_s=config["basis_duration_s"]
    )
    reconstruction = float(
        np.max(
            np.abs(
                np.asarray(restored.document["coefficients"]) - document["coefficients"]
            )
        )
    )
    lower, upper = [
        np.asarray(config[key]).reshape(len(names), 7)
        for key in ("lower_effort", "upper_effort")
    ]
    active = np.isclose(controls, lower, atol=1e-8, rtol=0) | np.isclose(
        controls, upper, atol=1e-8, rtol=0
    )
    expected = {
        (row["coordinate"], row["bernstein_index"]) for row in previous["efforts"]
    }
    actual = {(names[i], int(j)) for i, j in zip(*np.where(active), strict=True)}
    if actual != expected or int(active.sum()) != previous["active_effort_controls"]:
        raise ValueError("Independent correction recovery disagrees with bound audit")
    horizon = document["duration_s"]
    times = np.unique(np.r_[0, 0.2, 0.4, 0.6, 0.7, 0.8, horizon])
    times = times[times <= horizon]
    root = next(joint for joint in spec["joints"] if joint["parent"] == "world")
    rotation = np.asarray(root["parent_to_base"])[:3, :3].T
    profile = NativeEffortProfile(names, document["coefficients"], rotation)
    coefficients, parent_coefficients = (
        np.asarray(document["coefficients"]),
        np.asarray(parent["coefficients"]),
    )
    commands = np.array([np.polyval(row, times) for row in coefficients]).T
    primitive = np.array(
        [list(profile.evaluate(float(time)).values()) for time in times]
    )
    primitive_coefficients = coefficients.copy()
    primitive_coefficients[:3] = rotation @ coefficients[:3]
    rows = []
    for i, name in enumerate(names):
        difference = coefficients[i] - parent_coefficients[i]
        rows.append(
            {
                "coordinate": name,
                "unit": "N" if i < 3 else "Nm",
                "command_frame": "world Cartesian force"
                if i < 3
                else "joint generalized torque (not a Cartesian moment component)",
                "active_controls": [
                    {
                        "index": int(j),
                        "value": float(controls[i, j]),
                        "bound": "lower"
                        if np.isclose(controls[i, j], lower[i, j], atol=1e-8, rtol=0)
                        else "upper",
                    }
                    for j in np.flatnonzero(active[i])
                ],
                "correction_controls": controls[i].tolist(),
                "correction_lower": lower[i].tolist(),
                "correction_upper": upper[i].tolist(),
                "total_effort_extrema": extrema(coefficients[i], horizon),
                "parent_effort_extrema": extrema(parent_coefficients[i], horizon),
                "correction_extrema_full_horizon": extrema(difference, horizon),
                "correction_extrema_basis_interval": extrema(
                    difference, min(horizon, config["basis_duration_s"])
                ),
                "primitive_effort_extrema": extrema(primitive_coefficients[i], horizon),
                "command_samples": commands[:, i].tolist(),
                "primitive_samples": primitive[:, i].tolist(),
            }
        )
    report = {
        "qualification": "read-only effort audit; numerical correction bounds are not physical or biological actuator limits",
        "candidate_sha256": candidate.sha256,
        "original_parent_sha256": original.sha256,
        "coverage_adjusted_base_sha256": base.sha256,
        "model_sha256": model_hash,
        "horizon_s": horizon,
        "basis_duration_s": config["basis_duration_s"],
        "reconstruction_coefficient_max_abs": reconstruction,
        "active_effort_controls": int(active.sum()),
        "active_channels": int(active.any(axis=1).sum()),
        "sample_times_s": times.tolist(),
        "world_to_root_primitive_force_rotation": rotation.tolist(),
        "channels": rows,
        "warning": "Bernstein convex-hull bounds apply on the basis interval only; coverage beyond it extrapolates the same global polynomial",
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sys.stdout.write(
        json.dumps(
            {
                "active_controls": int(active.sum()),
                "active_channels": report["active_channels"],
                "reconstruction": reconstruction,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
