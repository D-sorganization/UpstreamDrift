"""Audit recorded Solution Bundle total applied effort (Refs #15493).

Run as ``python3 -m scripts.audit_optimization_effort BUNDLE --limits limits.json``.
Uses the canonical codec and sampled torque/slew audit; adds no dynamics model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.shared.python.optimization.applied_effort_audit import audit_applied_effort
from src.shared.python.optimization.solution_bundle import load_solution_bundle

_FIELDS = ("lower_nm", "upper_nm", "slew_nm_s")


def audit_bundle(bundle_dir: Path, limits: Mapping[str, Any]) -> dict[str, Any]:
    """Return strict-JSON-compatible sampled margins in bundle joint order.

    Preconditions: a codec-valid bundle and exactly one limit record per joint.
    Records contain only lower_nm, upper_nm and slew_nm_s, all finite real
    numbers (not booleans). Bounds refer to total applied rotational torque
    in the bundle's joint signs/frame; limits are caller-declared, not inferred.

    Postcondition: feasibility concerns sampled total torque and adjacent-sample
    secant slew only. No replay, inter-sample or physiological claim is made.
    Invalid data raises ValueError; codec integrity errors propagate.
    """
    bundle = load_solution_bundle(bundle_dir)
    joints = bundle.joint_order
    if not isinstance(limits, Mapping) or set(limits) != set(joints):
        raise ValueError("limits must contain exactly the bundle joint names")
    vectors: dict[str, list[float]] = {field: [] for field in _FIELDS}
    for joint in joints:
        record = limits[joint]
        if not isinstance(record, Mapping) or set(record) != set(_FIELDS):
            raise ValueError(f"limits for {joint} must contain exactly {_FIELDS}")
        for field in _FIELDS:
            value = record[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{joint}.{field} must be a finite real number")
            vectors[field].append(float(value))
    arrays = bundle.arrays
    times = arrays["time"]
    audit = audit_applied_effort(
        times_s=times,
        applied_torque_nm=arrays["tau_applied"],
        torque_lower_nm=vectors["lower_nm"],
        torque_upper_nm=vectors["upper_nm"],
        slew_limit_nm_s=vectors["slew_nm_s"],
    )
    rows = [
        {
            "joint": joint,
            "lower_nm": vectors["lower_nm"][i],
            "upper_nm": vectors["upper_nm"][i],
            "slew_limit_nm_s": vectors["slew_nm_s"][i],
            "torque_margin_nm": audit.torque_margin_nm[i],
            "slew_margin_nm_s": audit.slew_margin_nm_s[i],
        }
        for i, joint in enumerate(joints)
    ]
    return {
        "schema_version": "optimization-effort-audit/1",
        "solution_id": bundle.solution_id,
        "git_sha": bundle.git_sha,
        "model_hash": bundle.model_hash,
        "frame": bundle.frame,
        "sample_count": len(times),
        "duration_s": float(times[-1] - times[0]),
        "torque_source": "tau_applied",
        "sampled_effort_feasible": audit.feasible,
        "forward_replay_verified": False,
        "joints": rows,
        "violating_joints": [
            joint
            for joint, torque, slew in zip(
                joints, audit.torque_margin_nm, audit.slew_margin_nm_s, strict=True
            )
            if torque < 0 or slew < 0
        ],
        "limitations": [
            "Caller-declared rotational torque limits; no clinical safety claim.",
            "Sampled torque and secant slew only; no inter-sample or ZOH jump guarantee.",
            "No work, ROM, tracking, contact, flight or forward-replay qualification.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Emit JSON; return 0 for sampled pass, 1 for violation, 2 for invalid input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--limits", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        limits = json.loads(args.limits.read_text(encoding="utf-8"))
        report = audit_bundle(args.bundle, limits)
        output = json.dumps(report, allow_nan=False, indent=2)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    print(output)
    return 0 if report["sampled_effort_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
