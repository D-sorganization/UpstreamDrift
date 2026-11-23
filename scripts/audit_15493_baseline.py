"""Diagnostic v2 forward-tracking audit; it never qualifies physical feasibility.

The bundle's embedded target drives the existing MuJoCo controller with no
ball force. Output contains no capture path or private bundle identifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.same_input import InputBundle, closed_loop

V2_SPEC = Path(__file__).resolve().parents[1] / (
    "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
)


def audit_bundle(bundle: Any, pinned_spec: dict[str, Any]) -> dict[str, Any]:
    """Compare a replay to its original target; missing gates remain unverified."""
    result: dict[str, Any] = {
        "schema": "v2-forward-tracking-diagnostic/1",
        "status": "unqualified",
        "tracking": {"status": "unavailable"},
        "gates": {
            "target_and_ledger_provenance": "unverified",
            "anatomical_rom": "unverified",
            "joint_speed": "unverified",
            "total_applied_torque": "unverified",
            "mechanical_work": "unverified",
            "first_on_face_contact": "unverified",
            "flight_and_carry": "unverified",
        },
    }
    if canonical_sha256(json.loads(bundle.spec_bytes)) != canonical_sha256(pinned_spec):
        result["tracking"]["reason"] = "bundle model does not match pinned v2 spec"
        return result
    times, target = bundle.target_time_s, bundle.target_q
    if times is None or target is None:
        result["tracking"]["reason"] = "original controller target is unavailable"
        return result
    # InputBundle validates the embedded target; the clock is rebased exactly as
    # in the existing controller. Do not invent a target from rollout states.
    times = np.asarray(times, dtype=float) - float(times[0])
    target = np.asarray(target, dtype=float)
    rollout = closed_loop(
        "mujoco",
        bundle.spec_bytes,
        times,
        target,
        duration_s=bundle.steps * bundle.dt_s,
        impact=None,
        policy=bundle.step_policy(),
    )
    t, q = np.asarray(rollout.time_s, dtype=float), np.asarray(rollout.q, dtype=float)
    if (
        t.ndim != 1
        or not t.size
        or q.shape != (t.size, len(bundle.coordinate_order))
        or not np.isfinite(t).all()
        or not np.isfinite(q).all()
        or np.any(np.diff(t) <= 0)
        or t[0] < 0
        or t[-1] > times[-1] + bundle.dt_s / 2
    ):
        raise ValueError("forward rollout has invalid or unsupported samples")
    interpolated = np.column_stack(
        [np.interp(t, times, target[:, i]) for i in range(target.shape[1])]
    )
    errors = q - interpolated
    if not np.isfinite(errors).all():
        raise ValueError("tracking error is nonfinite")
    result["tracking"] = {
        "status": "measured",
        "method": "computed-torque target tracking; controller initializes and projects its own state",
        "samples": len(t),
        "duration_s": float(t[-1]),
        "rollout_failure": rollout.failure,
        "per_coordinate": {
            name: {
                "unit": "m" if name.startswith("Translation") else "rad",
                "rms": float(np.sqrt(np.mean(errors[:, i] ** 2))),
                "max_abs": float(np.max(np.abs(errors[:, i]))),
            }
            for i, name in enumerate(bundle.coordinate_order)
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_bundle(
        InputBundle.load(args.bundle), json.loads(V2_SPEC.read_bytes())
    )
    args.out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    raise SystemExit(2)  # Diagnostic evidence is never an admitted baseline.


if __name__ == "__main__":
    main()
