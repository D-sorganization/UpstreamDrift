"""Run the MyoFullBody mapping and static optimisation for one bundle (#11643-#11645).

    MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen PYTHONPATH=.:src \
        python3 scripts/run_myofullbody_swing.py --bundle driver.npz \
        --receipt receipt.json --solution solution.npz

Needs the asset cache: ``python3 scripts/fetch_myofullbody.py`` first.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shlex
import sys

import numpy as np

from src.shared.python.myofullbody.swing_pipeline import SwingConfig, run_swing

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--reserve-weight", type=float, default=100.0)
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = SwingConfig(
        args.bundle,
        args.stride,
        args.reserve_weight,
        args.cache_root,
        invocation=shlex.join(["python3", *sys.argv]),
    )
    result = run_swing(config)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result.receipt, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.solution, **result.arrays)
    logger.info(
        "status %s; receipt %s", result.receipt["qualification"]["status"], args.receipt
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
