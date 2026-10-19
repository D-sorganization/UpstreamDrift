"""Barbell-on-bushings evidence for the barbell bundles (LIFT-4, #11744).

Usage (headless, from the repository root; each part can run alone)::

    MUJOCO_GL=egl python3 scripts/generate_barbell_squat_evidence.py \
        --lift back_squat --part hold --part bundle --part parity --source-commit <sha>

* ``hold``: quasi-static hold of the lift's standing pose in MuJoCo with the
  bar on both hands (``barbell_hold.json``).
* ``bundle``: the lift window re-tracked with the bar attached
  (``<lift>_barbell_bundle.npz``).
* ``parity``: per-hand bushing wrenches and coupled accelerations of Drake,
  Pinocchio and OpenSim against MuJoCo at every 20th reference state of that
  bundle (``barbell_parity.json``); needs the ``bundle`` output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco  # noqa: F401
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.lifting.barbell_bushing import Barbell  # noqa: E402
from src.shared.python.lifting.barbell_runs import (  # noqa: E402
    build_barbell_bundle,
    hold_pose,
    instantaneous_parity,
    quasi_static_hold,
)
from src.shared.python.lifting.bundles import (  # noqa: E402
    BACK_SQUAT_BARBELL_BUNDLE,
    BACK_SQUAT_BUNDLE,
    BENCH_PRESS_BARBELL_BUNDLE,
    BENCH_PRESS_BUNDLE,
    CLEAN_AND_JERK_BARBELL_BUNDLE,
    CLEAN_AND_JERK_BUNDLE,
    DEADLIFT_BARBELL_BUNDLE,
    DEADLIFT_BUNDLE,
    SNATCH_BARBELL_BUNDLE,
    SNATCH_BUNDLE,
)
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402

logger = logging.getLogger("generate_barbell_squat_evidence")
PARITY_ENGINES = ("drake", "pinocchio", "opensim")
HOLD_DURATION_S = 1.5
SETTLE_WINDOW_S = 0.5
PARITY_STRIDE = 20

LIFT_BUNDLES: dict[str, tuple[Path, Path]] = {
    "back_squat": (BACK_SQUAT_BUNDLE, BACK_SQUAT_BARBELL_BUNDLE),
    "deadlift": (DEADLIFT_BUNDLE, DEADLIFT_BARBELL_BUNDLE),
    "bench_press": (BENCH_PRESS_BUNDLE, BENCH_PRESS_BARBELL_BUNDLE),
    "snatch": (SNATCH_BUNDLE, SNATCH_BARBELL_BUNDLE),
    "clean_and_jerk": (CLEAN_AND_JERK_BUNDLE, CLEAN_AND_JERK_BARBELL_BUNDLE),
}


def _bar_mass(source: InputBundle) -> float:
    return float(source.provenance["optimizer"]["bar_mass_kg"])


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", path)


def run_hold(source: InputBundle, commit: str, lift: str, hold_json: Path) -> None:
    barbell = Barbell(_bar_mass(source))
    q_hold = hold_pose(source)
    result = quasi_static_hold(
        source.spec_bytes, q_hold, barbell, duration_s=HOLD_DURATION_S
    )
    gen = (
        "scripts/generate_barbell_squat_evidence.py --part hold"
        if lift == "back_squat"
        else f"scripts/generate_barbell_squat_evidence.py --lift {lift} --part hold"
    )
    record = {
        "lift": lift,
        "pose": (
            "balanced IK standing pose (barbell_runs.hold_pose), root in contact "
            "equilibrium for body plus bar (barbell_runs.hold_start, #11744)"
        ),
        "source_commit": commit,
        "bundle_spec_sha256": source.spec_sha256,
        "generator": gen,
        "summary": result.summary(SETTLE_WINDOW_S),
        "series": {
            "time_s": result.time_s.tolist(),
            "grf_vertical_n": result.grf_n[:, 2].tolist(),
            "grf_forward_n": result.grf_n[:, 0].tolist(),
            "hand_vertical_on_bar_n": {
                s: result.hand_force_n[s][:, 2].tolist() for s in ("L", "R")
            },
            "max_deflection_m": result.deflection_m.tolist(),
        },
    }
    _write_json(hold_json, record)


def run_bundle(source: InputBundle, commit: str, target_path: Path) -> None:
    bundle = build_barbell_bundle(source, Barbell(_bar_mass(source)))
    provenance = dict(bundle.provenance)
    provenance["barbell_source_commit"] = commit
    # replace() keeps the step policy (substeps, effort holds and the hold
    # efforts) the re-track was integrated with (#13104).
    replace(bundle, provenance=provenance).save(target_path)
    logger.info("wrote %s", target_path)


def run_parity(commit: str, lift: str, bundle_path: Path, parity_json: Path) -> None:
    bundle = InputBundle.load(bundle_path)
    indices = list(range(0, bundle.steps + 1, PARITY_STRIDE))
    report = instantaneous_parity(bundle, PARITY_ENGINES, indices)
    gen = (
        "scripts/generate_barbell_squat_evidence.py --part parity"
        if lift == "back_squat"
        else f"scripts/generate_barbell_squat_evidence.py --lift {lift} --part parity"
    )
    report |= {
        "lift": lift,
        "bundle": str(bundle_path.relative_to(ROOT)),
        "bundle_spec_sha256": bundle.spec_sha256,
        "source_commit": commit,
        "generator": gen,
        "tolerance": "|dX| <= 1e-6 + 1e-8 * max|X| (L1, epic #11605)",
    }
    _write_json(parity_json, report)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lift",
        choices=(
            "back_squat",
            "squat",
            "deadlift",
            "bench_press",
            "snatch",
            "clean_and_jerk",
        ),
        default="back_squat",
    )
    parser.add_argument(
        "--part", action="append", choices=("hold", "bundle", "parity"), required=True
    )
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)

    lift_key = "back_squat" if args.lift == "squat" else args.lift
    source_path, target_path = LIFT_BUNDLES[lift_key]
    evidence_dir = source_path.parent
    hold_json = evidence_dir / "barbell_hold.json"
    parity_json = evidence_dir / "barbell_parity.json"

    source = InputBundle.load(source_path)
    if "hold" in args.part:
        run_hold(source, args.source_commit, lift_key, hold_json)
    if "bundle" in args.part:
        run_bundle(source, args.source_commit, target_path)
    if "parity" in args.part:
        run_parity(args.source_commit, lift_key, target_path, parity_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
