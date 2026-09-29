"""Build nondegenerate cross-engine numeric-comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
SOURCE = DATA / "spatial_forward_contact_study.npz"
OUTPUT = DATA / "claim_numeric_comparison_evidence.json"


def build_record() -> dict[str, object]:
    with np.load(SOURCE, allow_pickle=False) as arrays:
        reference = np.asarray(
            arrays["mujoco_killswitch_swing_normal_couple"], dtype=float
        )
        candidate = np.asarray(
            arrays["pinocchio_killswitch_swing_normal_couple"], dtype=float
        )
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("cross-engine comparison arrays must be aligned vectors")
    if reference.size == 0 or np.array_equal(reference, candidate):
        raise ValueError("cross-engine comparison must be nonempty and nondegenerate")
    return {
        "schema_version": "claim-numeric-comparison-evidence-v1",
        "source_artifact": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "spatial_forward_contact": {
            "quantity": "killswitch_swing_normal_couple_nm",
            "reference_engine": "mujoco",
            "candidate_engine": "pinocchio",
            "reference": reference.tolist(),
            "candidate": candidate.tolist(),
        },
        "boundary": (
            "Cross-engine numerical agreement is model-conditional and is not "
            "independent empirical or human validation."
        ),
    }


def validate_record() -> dict[str, int | str]:
    expected = build_record()
    if not OUTPUT.is_file():
        raise ValueError("numeric comparison evidence is stale")
    try:
        record = json.loads(OUTPUT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("numeric comparison evidence is invalid JSON") from exc
    if record != expected:
        raise ValueError("numeric comparison evidence is stale")
    comparison = record["spatial_forward_contact"]
    return {
        "source_sha256": str(record["source_sha256"]),
        "comparison_sample_count": len(comparison["reference"]),
        "completion_status": "complete",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "check"), nargs="?", default="write")
    args = parser.parse_args()
    rendered = json.dumps(build_record(), indent=2, ensure_ascii=False) + "\n"
    if args.mode == "check":
        validate_record()
    else:
        OUTPUT.write_text(rendered, encoding="utf-8")
    print(json.dumps({"mode": args.mode, "output": str(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
