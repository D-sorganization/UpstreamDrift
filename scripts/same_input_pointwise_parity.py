"""Write the pointwise same-input parity receipt (#11606, epic #11605).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_pointwise_parity

Projects every ``--stride``-th recorded state of a full-body dynamics record
onto the dual-grip closure manifold and compares MuJoCo (exact KKT), Drake and
Pinocchio accelerations there under the recorded efforts.  Also records the
off-manifold disagreement that motivates the projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.same_input import (
    PARITY_ENGINES,
    VectorPlant,
    project_to_closure,
)

EVIDENCE = Path("docs/development/full_body_models/evidence")
DEFAULT_SPEC = EVIDENCE / "ground_support/full_body_spec_hipcal_scaled.json"
DEFAULT_RECORD = EVIDENCE / "ground_support/dynamics_record.npz"
DEFAULT_OUT = EVIDENCE / "same_input/pointwise_parity_receipt.json"
# Epic #11605 level L1: |da| <= ATOL + RTOL * max|a|.  Contact spikes reach
# max|a| ~ 1.3e5 where cond(M) ~ 6e8, so eps * cond(M) ~ 1.4e-7 bounds round-off;
# RTOL sits 14x inside that and 10x outside the observed 1.1e-9.
ACCELERATION_ATOL = 1e-6
ACCELERATION_RTOL = 1e-8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt(spec_path: Path, record_path: Path, stride: int) -> dict:
    """Return the parity receipt; ``stride`` must be a positive integer."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    spec = spec_path.read_bytes()
    plants = {engine: VectorPlant(engine, spec) for engine in PARITY_ENGINES}
    with np.load(record_path) as data:
        q_all, v_all, tau_all = data["q"], data["v"], data["tau"]
    rows = []
    for frame in range(0, q_all.shape[0], stride):
        q, v, tau = q_all[frame], v_all[frame], tau_all[frame]
        states = {e: project_to_closure(p, q, v) for e, p in plants.items()}
        ref = states["pinocchio"]
        acc = {e: p.acceleration(ref.q, ref.v, tau) for e, p in plants.items()}
        raw = {e: p.acceleration(q, v, tau) for e, p in plants.items()}
        rows.append(
            {
                "frame": frame,
                "pose_residual_before": ref.pose_residual_before,
                "state_spread_q": max(
                    float(np.abs(s.q - ref.q).max()) for s in states.values()
                ),
                "state_spread_v": max(
                    float(np.abs(s.v - ref.v).max()) for s in states.values()
                ),
                "max_abs_acceleration": float(np.abs(acc["pinocchio"]).max()),
                "projected": {
                    e: float(np.abs(acc[e] - acc["pinocchio"]).max())
                    for e in ("mujoco", "drake")
                },
                "off_manifold": {
                    e: float(np.abs(raw[e] - raw["pinocchio"]).max())
                    for e in ("mujoco", "drake")
                },
            }
        )
    for row in rows:
        bound = ACCELERATION_ATOL + ACCELERATION_RTOL * row["max_abs_acceleration"]
        row["passed"] = max(row["projected"].values()) <= bound
    worst = max(max(r["projected"].values()) for r in rows)
    worst_relative = max(
        max(r["projected"].values()) / max(r["max_abs_acceleration"], 1.0) for r in rows
    )
    return {
        "schema": "same-input-pointwise-parity/v1",
        "issue": 11606,
        "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "record": {"path": str(record_path), "sha256": _sha256(record_path)},
        "engines": list(PARITY_ENGINES),
        "mujoco_kkt_regularization": 0.0,
        "acceleration_atol": ACCELERATION_ATOL,
        "acceleration_rtol": ACCELERATION_RTOL,
        "frames_checked": len(rows),
        "worst_projected_acceleration_difference": worst,
        "worst_projected_relative_difference": worst_relative,
        "worst_state_spread": max(
            max(r["state_spread_q"], r["state_spread_v"]) for r in rows
        ),
        "worst_off_manifold_difference": max(
            max(r["off_manifold"].values()) for r in rows
        ),
        "passed": all(r["passed"] for r in rows),
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    receipt = build_receipt(args.spec, args.record, args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(  # noqa: T201 - CLI summary
        f"{receipt['frames_checked']} frames, worst projected "
        f"{receipt['worst_projected_acceleration_difference']:.2e} "
        f"(relative {receipt['worst_projected_relative_difference']:.1e}), "
        f"off-manifold "
        f"{receipt['worst_off_manifold_difference']:.2e}, passed={receipt['passed']}"
    )


if __name__ == "__main__":
    main()
