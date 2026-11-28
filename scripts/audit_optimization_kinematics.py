"""Audit recorded joint ROM, speed and tracking margins (Refs #15493).

Run ``python -m scripts.audit_optimization_kinematics BUNDLE LIMITS.json``.
Limits are explicit per-joint angular-coordinate budgets in the bundle frame.
See docs/research/optimization_kinematic_audit/README.md for the contract.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.optimization.solution_bundle import (
    SolutionBundle,
    load_solution_bundle,
)

_LIMITS = (
    "position_lower_rad",
    "position_upper_rad",
    "speed_limit_rad_s",
    "position_error_limit_rad",
    "velocity_error_limit_rad_s",
)


def _array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != shape:
        raise ValueError(f"{name} must be real numeric with shape {shape}")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def audit_bundle(bundle: SolutionBundle, limits: Mapping[str, Any]) -> dict[str, Any]:
    """Return signed per-joint minimum margins over recorded samples.

    Preconditions: at least two increasing finite times; finite (N,J) actual
    and reference angular q/qd; finite (J,) limits in the exact joint order;
    lower <= upper; nonnegative speed and absolute tracking-error budgets.
    All coordinates must have a shared unwrapped radian convention. Caller
    owns the biomechanical validity of limits; none are inferred from names.

    Postconditions: strict-JSON report with finite margins; sampled_feasible
    iff every margin >= 0. No inter-sample or forward-dynamics qualification.
    Overflow or malformed data raises ValueError rather than passing.
    """
    if not isinstance(limits, Mapping):
        raise ValueError("limits must be an object")
    missing = {*_LIMITS, "joint_order"} - limits.keys()
    if missing:
        raise ValueError(f"missing limits: {sorted(missing)}")
    names = bundle.joint_order
    if not names or len(set(names)) != len(names):
        raise ValueError("bundle joint_order must be nonempty and unique")
    if limits["joint_order"] != list(names):
        raise ValueError("limits joint_order must match bundle joint_order exactly")
    arrays = bundle.arrays
    required = ("time", "q_rollout", "qd_rollout", "q_ref", "qd_ref")
    if set(required) - arrays.keys():
        raise ValueError("missing kinematic arrays")
    times = np.asarray(arrays["time"])
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time must be one-dimensional with at least two samples")
    times = _array(times, "time", times.shape)
    if np.any(times[1:] <= times[:-1]):
        raise ValueError("time must be strictly increasing")
    shape = (times.size, len(names))
    q, qd, ref, refd = (_array(arrays[key], key, shape) for key in required[1:])
    lower, upper, speed, qerr, v_err = (
        _array(limits[key], key, (len(names),)) for key in _LIMITS
    )
    if np.any(lower > upper):
        raise ValueError("position lower bounds must not exceed upper bounds")
    if any(np.any(vector < 0) for vector in (speed, qerr, v_err)):
        raise ValueError("speed and tracking limits must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        margins = {
            "position_margin_rad": np.minimum(q - lower, upper - q).min(axis=0),
            "speed_margin_rad_s": speed - np.abs(qd).max(axis=0),
            "position_tracking_margin_rad": qerr - np.abs(q - ref).max(axis=0),
            "velocity_tracking_margin_rad_s": v_err - np.abs(qd - refd).max(axis=0),
        }
    if not all(np.isfinite(value).all() for value in margins.values()):
        raise ValueError("derived margins must be finite")
    return {
        "solution_id": bundle.solution_id,
        "model_hash": bundle.model_hash,
        "frame": bundle.frame,
        "joint_order": list(names),
        "samples": times.size,
        "scope": "recorded angular-coordinate samples only",
        "sampled_feasible": all(bool(np.all(m >= 0)) for m in margins.values()),
        **{name: value.tolist() for name, value in margins.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Emit strict JSON; exit 0 if feasible, 1 if violated, 2 for invalid input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("limits", type=Path)
    args = parser.parse_args(argv)
    try:
        bundle = load_solution_bundle(args.bundle)
        limits = json.loads(args.limits.read_text(encoding="utf-8"))
        report = audit_bundle(bundle, limits)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid kinematic audit input: {exc}\n")
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0 if report["sampled_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
