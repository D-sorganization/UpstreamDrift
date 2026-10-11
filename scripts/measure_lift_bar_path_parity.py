"""Measure cross-engine bar-path parity of the back-squat bundle (LIFT-5, #11745).

Usage (headless, from the repository root)::

    MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py \
        --out docs/development/lifting/evidence/back_squat/bar_path_parity.json

Each engine replays the whole committed bundle open loop; an engine whose
Python package is not importable is recorded as unavailable with the reason,
never as zero deviation.  The 200-step replay takes minutes per engine, so
run it on an unloaded host.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.lifting.bar_path_parity import (  # noqa: E402
    BAR_PATH_PARITY_RECORD,
    replay_bar_path,
)
from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE  # noqa: E402
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402

logger = logging.getLogger("measure_lift_bar_path_parity")
ENGINE_MODULES = {
    "mujoco": "mujoco",
    "drake": "pydrake",
    "pinocchio": "pinocchio",
    "opensim": "opensim",
}


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


def _measure(engine: str, bundle: InputBundle) -> dict[str, Any]:
    try:
        importlib.import_module(ENGINE_MODULES[engine])
    except (ImportError, OSError) as exc:
        return {"unavailable": True, "reason": f"{ENGINE_MODULES[engine]}: {exc}"}
    started = time.perf_counter()
    summary = replay_bar_path(engine, bundle).summary()
    summary["wall_time_s"] = round(time.perf_counter() - started, 1)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=BACK_SQUAT_BUNDLE)
    parser.add_argument("--out", type=Path, default=BAR_PATH_PARITY_RECORD)
    parser.add_argument(
        "--engines", default=",".join(ENGINE_MODULES), help="comma separated"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    bundle = InputBundle.load(args.bundle)
    engines: dict[str, Any] = {}
    for engine in args.engines.split(","):
        engines[engine] = _measure(engine, bundle)
        logger.info("%s %s", engine, json.dumps(engines[engine]))
    record = {
        "lift": bundle.provenance.get("lift"),
        "bundle": str(args.bundle.resolve().relative_to(ROOT)),
        "bundle_spec_sha256": bundle.spec_sha256,
        "bundle_steps": bundle.steps,
        "dt_s": bundle.dt_s,
        "reference": "window_bar_path_m (MuJoCo, recorded at bundle generation)",
        "source_commit": _git("rev-parse", "HEAD"),
        "generator": "scripts/measure_lift_bar_path_parity.py",
        "engines": engines,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
