"""Emit source-bound native CustomJoint MTP derivation and sampled receipt."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from src.engines.physics_engines.opensim.python.native_custom_mtp_reduction import (
    ZeroCustomMtpReductionRequest,
    derive_zero_custom_mtp_model,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    if not args.receipt.parent.is_dir():
        raise ValueError("receipt parent directory missing")
    result = derive_zero_custom_mtp_model(
        ZeroCustomMtpReductionRequest(
            source_model_path=args.source,
            source_sha256=_sha(args.source),
            derived_model_path=args.derived,
            declared_target_rad=(("mtp_angle_r", 0.0), ("mtp_angle_l", 0.0)),
        )
    )
    payload = {
        "issue": 12176,
        "scope": "sampled-zero-CustomJoint-MTP-mechanics-only",
        "reducer": asdict(result),
        "receipt_script_sha256": _sha(Path(__file__)),
    }
    args.receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
