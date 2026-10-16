"""Measure cross-engine bar-path parity of a lift bundle (LIFT-5, #11745).

Usage (headless, from the repository root)::

    MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift squat
    MUJOCO_GL=egl python3 scripts/measure_lift_bar_path_parity.py --lift deadlift

``--lift`` reads the bundle and writes the record at the lift recipe's
evidence paths (``src/shared/python/lifting/lift_recipes.py``); ``--bundle``
and ``--out`` override them.

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_SRC = ROOT / "vendor" / "ud-tools" / "src"
if str(TOOLS_SRC) not in sys.path:
    # Ahead of site-packages so the pinned Tools tree resolves the retired
    # shared roots (sidekick utils) imported by the same-input pipeline.
    sys.path.insert(1, str(TOOLS_SRC))

from src.shared.python.lifting.bar_path_parity import replay_bar_path  # noqa: E402
from src.shared.python.lifting.lift_recipes import (  # noqa: E402
    bundle_lifts,
    recipe,
)
from src.shared.python.motion_matching.same_input import (  # noqa: E402
    InputBundle,
    VectorPlant,
    segmented_replay,
)

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


def _measure(engine: str, bundle: InputBundle, segment_steps: int) -> dict[str, Any]:
    try:
        importlib.import_module(ENGINE_MODULES[engine])
    except (ImportError, OSError) as exc:
        return {"unavailable": True, "reason": f"{ENGINE_MODULES[engine]}: {exc}"}
    started = time.perf_counter()
    replay = replay_bar_path(engine, bundle)
    summary = replay.summary()
    worst = int(np.argmax(replay.final_coordinate_error_rad))
    summary["worst_final_coordinate"] = bundle.coordinate_order[worst]
    segments = segmented_replay(
        VectorPlant(engine, bundle.spec_bytes), bundle, segment_steps=segment_steps
    )
    summary["segmented"] = {
        "segment_steps": segment_steps,
        "segments": len(segments),
        "max_coordinate_error_rad": max(s["coordinate_error_rad"] for s in segments),
        "max_frame_error_m": max(s["frame_error_m"] for s in segments),
        "worst_segment_start_s": max(segments, key=lambda s: s["coordinate_error_rad"])[
            "start_s"
        ],
        "coordinate_error_rad_by_segment": [
            s["coordinate_error_rad"] for s in segments
        ],
    }
    summary["wall_time_s"] = round(time.perf_counter() - started, 1)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lift", choices=bundle_lifts(), default="squat")
    parser.add_argument("--bundle", type=Path, default=None, help="default: recipe")
    parser.add_argument("--out", type=Path, default=None, help="default: recipe")
    parser.add_argument(
        "--source-commit",
        default=None,
        help="commit measured (default: git rev-parse HEAD)",
    )
    parser.add_argument(
        "--segment-steps",
        type=int,
        default=20,
        help="re-anchored segment length for the segmented replay",
    )
    parser.add_argument(
        "--engines", default=",".join(ENGINE_MODULES), help="comma separated"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    lift = recipe(args.lift)
    bundle_path: Path = args.bundle or lift.bundle_path
    bundle = InputBundle.load(bundle_path)
    engines: dict[str, Any] = {}
    for engine in args.engines.split(","):
        engines[engine] = _measure(engine, bundle, args.segment_steps)
        logger.info("%s %s", engine, json.dumps(engines[engine]))
    record = {
        "lift": bundle.provenance.get("lift"),
        "bundle": str(bundle_path.resolve().relative_to(ROOT)),
        "bundle_spec_sha256": bundle.spec_sha256,
        "bundle_steps": bundle.steps,
        "dt_s": bundle.dt_s,
        "reference": "window_bar_path_m (MuJoCo, recorded at bundle generation)",
        "source_commit": args.source_commit or _git("rev-parse", "HEAD"),
        "generator": "scripts/measure_lift_bar_path_parity.py",
        "engines": engines,
    }
    out: Path = args.out or lift.parity_record_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
