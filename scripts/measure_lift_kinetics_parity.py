"""Measure lift kinetics parity across engines (LIFT-7, OSL-8, #11747, #13045).

Usage (headless, from the repository root)::

    MUJOCO_GL=egl python3 scripts/measure_lift_kinetics_parity.py
    MUJOCO_GL=egl python3 scripts/measure_lift_kinetics_parity.py --lift deadlift

Every engine evaluates the ground-reaction breakdown and the grip wrench at
the same reference states of each registered lift bundle; the record holds
the largest difference from MuJoCo per quantity, plus the MuJoCo values for
scale. An engine that cannot be imported, or cannot supply a quantity, is
recorded as unavailable with a reason, never as zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.lifting.kinetics_parity import (  # noqa: E402
    DEFAULT_STRIDE,
    evaluate_lift_kinetics_parity,
    lift_kinetics_bundle_path,
    lift_kinetics_record_path,
    unavailable_kinetics_parity_record,
)
from src.shared.python.lifting.pack_audit.names import LIFTS  # noqa: E402
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402

logger = logging.getLogger("measure_lift_kinetics_parity")
DEFAULT_OUT = lift_kinetics_record_path("squat", ROOT)


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lift",
        default="all",
        help="lift to measure ('all' for every registered lift, or a specific lift)",
    )
    parser.add_argument(
        "--bundle", type=Path, default=None, help="override bundle path"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="override output JSON path"
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="state stride"
    )
    parser.add_argument("--source-commit", default=None)
    return parser


def _measure_one(
    bundle_path: Path | None,
    out_path: Path,
    *,
    lift_name: str,
    stride: int,
    source_commit: str,
) -> None:
    if bundle_path is None or not bundle_path.is_file():
        record = unavailable_kinetics_parity_record(
            lift_name,
            f"no bundle registered in SAME_INPUT_LIFT_BUNDLES for {lift_name}",
            source_commit=source_commit,
        )
    else:
        bundle = InputBundle.load(bundle_path)
        rel_bundle = str(bundle_path.resolve().relative_to(ROOT))
        record = evaluate_lift_kinetics_parity(
            bundle,
            stride=stride,
            source_commit=source_commit,
            bundle_rel_path=rel_bundle,
        )
        for engine, diff in record["max_difference_from_mujoco"].items():
            logger.info("[%s] %s %s", lift_name, engine, json.dumps(diff))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    source_commit = args.source_commit or _git("rev-parse", "HEAD")

    if args.bundle is not None:
        out = args.out or DEFAULT_OUT
        bundle = InputBundle.load(args.bundle)
        lift_name = str(bundle.provenance.get("lift") or "back_squat")
        _measure_one(
            args.bundle,
            out,
            lift_name=lift_name,
            stride=args.stride,
            source_commit=source_commit,
        )
        return 0

    target_lifts = list(LIFTS) if args.lift in ("all", None) else [args.lift]

    for lift in target_lifts:
        bundle_path = lift_kinetics_bundle_path(lift)
        out_path = (
            args.out
            if (len(target_lifts) == 1 and args.out is not None)
            else lift_kinetics_record_path(lift, ROOT)
        )
        _measure_one(
            bundle_path,
            out_path,
            lift_name=lift,
            stride=args.stride,
            source_commit=source_commit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
