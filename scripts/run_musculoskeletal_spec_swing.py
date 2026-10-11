"""Run the phase-2 spec-driven musculoskeletal pipeline (issue #11617).

    MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen PYTHONPATH=.:src \
        python3 scripts/run_musculoskeletal_spec_swing.py --bundle driver.npz \
        --out-dir /tmp/msk2_driver --receipt receipt_v2_driver.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.engines.physics_engines.opensim.python.musculoskeletal_pipeline_v2 import (
    V2Config,
    compare_with_v1,
    run_v2,
)

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--id-stride", type=int, default=10)
    parser.add_argument("--v1-receipt", type=Path, default=None)
    parser.add_argument("--so-stride", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    receipt = run_v2(
        V2Config(
            args.bundle, args.out_dir, args.base_model, args.id_stride, args.so_stride
        )
    )
    if args.v1_receipt:
        receipt["comparison_with_v1"] = compare_with_v1(
            receipt, json.loads(args.v1_receipt.read_text())
        )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    logger.info("receipt written to %s", args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
