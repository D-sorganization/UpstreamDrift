"""Replay saved Pinocchio controls in MuJoCo; unverified inputs require diagnostics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.engines.physics_engines.mujoco.python.replay_contract import ReplaySettings
from src.engines.physics_engines.mujoco.python.replay_evidence import (
    ReplayFiles,
    generate_replay,
)


logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate",
        "source-receipt",
        "document",
        "capture",
        "attachments",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--armature", type=float, required=True)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Record unverified legacy configuration; never grant parity",
    )
    parser.add_argument("--max-evaluations", type=int, default=20000)
    args = parser.parse_args()
    files = ReplayFiles(
        args.candidate,
        args.source_receipt,
        args.document,
        args.capture,
        args.attachments,
        args.output,
        args.candidate_sha256,
    )
    settings = ReplaySettings(
        armature_kg_m2=args.armature, max_evaluations=args.max_evaluations
    )
    receipt = generate_replay(files, settings, diagnostic=args.diagnostic)
    logger.warning("G1 acceptance: %s", receipt["acceptance"]["status"])
    return 0 if receipt["acceptance"]["is_physically_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
