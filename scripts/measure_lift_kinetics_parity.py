"""Measure back-squat kinetics parity across engines (LIFT-7, #11747).

Usage (headless, from the repository root)::

    MUJOCO_GL=egl python3 scripts/measure_lift_kinetics_parity.py \
        --out docs/development/lifting/evidence/back_squat/kinetics_parity.json

Every engine evaluates the ground-reaction breakdown and the grip wrench at
the same reference states of the committed back-squat bundle; the record
holds the largest difference from MuJoCo per quantity, plus the MuJoCo
values for scale.  An engine that cannot be imported, or cannot supply a
quantity, is recorded as unavailable with a reason, never as zero.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE  # noqa: E402
from src.shared.python.lifting.kinetics_parity import (  # noqa: E402
    KineticsSample,
    bundle_kinetics,
    kinetics_differences,
)
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402

logger = logging.getLogger("measure_lift_kinetics_parity")
DEFAULT_OUT = ROOT / "docs/development/lifting/evidence/back_squat/kinetics_parity.json"
ENGINE_MODULES = {"drake": "pydrake", "pinocchio": "pinocchio", "opensim": "opensim"}


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


def _values(sample: KineticsSample) -> dict[str, Any]:
    grip = sample.grip
    return {
        "index": sample.index,
        "time_s": sample.time_s,
        "grf_left_n": sample.per_foot["left"].force_n.tolist(),
        "grf_right_n": sample.per_foot["right"].force_n.tolist(),
        "grf_net_n": sample.net.force_n.tolist(),
        "cop_net_m": None if sample.net.cop_m is None else sample.net.cop_m.tolist(),
        "grip_net_force_n": None
        if grip.net_force_n is None
        else list(grip.net_force_n),
    }


def _worst(
    reference: list[KineticsSample], other: list[KineticsSample]
) -> dict[str, Any]:
    worst: dict[str, Any] = {}
    for ref, oth in zip(reference, other, strict=True):
        for key, gap in kinetics_differences(ref, oth).items():
            if gap is None:
                worst.setdefault(key, None)
            else:
                worst[key] = max(worst.get(key) or 0.0, gap)
    reasons = {s.grip.unavailable_reason for s in other if s.grip.unavailable_reason}
    if reasons:
        worst["grip_unavailable_reason"] = sorted(reasons)[0]
    return worst


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=BACK_SQUAT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=20, help="state stride")
    parser.add_argument("--source-commit", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    bundle = InputBundle.load(args.bundle)
    indices = list(range(0, bundle.steps + 1, args.stride))
    reference = bundle_kinetics("mujoco", bundle, indices)
    engines: dict[str, Any] = {}
    for engine, module in ENGINE_MODULES.items():
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            engines[engine] = {"unavailable": True, "reason": f"{module}: {exc}"}
            continue
        engines[engine] = _worst(reference, bundle_kinetics(engine, bundle, indices))
        logger.info("%s %s", engine, json.dumps(engines[engine]))
    record = {
        "lift": bundle.provenance.get("lift"),
        "bundle": str(args.bundle.resolve().relative_to(ROOT)),
        "bundle_spec_sha256": bundle.spec_sha256,
        "indices": indices,
        "reference_engine": "mujoco",
        "source_commit": args.source_commit or _git("rev-parse", "HEAD"),
        "generator": "scripts/measure_lift_kinetics_parity.py",
        "mujoco_values": [_values(s) for s in reference],
        "max_difference_from_mujoco": engines,
    }
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
