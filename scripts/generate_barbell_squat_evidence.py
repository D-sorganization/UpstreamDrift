"""Barbell-on-bushings evidence for the back-squat bundle (LIFT-4, #11744).

Usage (headless, from the repository root; each part can run alone)::

    MUJOCO_GL=egl python3 scripts/generate_barbell_squat_evidence.py \
        --part hold --part bundle --part parity --source-commit <sha>

* ``hold``: quasi-static hold of the lift's standing pose in MuJoCo with the
  bar on both hands (``barbell_hold.json``).
* ``bundle``: the back-squat window re-tracked with the bar attached
  (``back_squat_barbell_bundle.npz``).
* ``parity``: per-hand bushing wrenches and coupled accelerations of Drake,
  Pinocchio and OpenSim against MuJoCo at every 20th reference state of that
  bundle (``barbell_parity.json``); needs the ``bundle`` output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.lifting.barbell_bushing import Barbell  # noqa: E402
from src.shared.python.lifting.barbell_runs import (  # noqa: E402
    build_barbell_bundle,
    instantaneous_parity,
    quasi_static_hold,
)
from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE  # noqa: E402
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402

logger = logging.getLogger("generate_barbell_squat_evidence")
EVIDENCE = BACK_SQUAT_BUNDLE.parent
HOLD_JSON = EVIDENCE / "barbell_hold.json"
BARBELL_BUNDLE = EVIDENCE / "back_squat_barbell_bundle.npz"
PARITY_JSON = EVIDENCE / "barbell_parity.json"
PARITY_ENGINES = ("drake", "pinocchio", "opensim")
HOLD_DURATION_S = 1.5
SETTLE_WINDOW_S = 0.5
PARITY_STRIDE = 20


def _bar_mass(source: InputBundle) -> float:
    return float(source.provenance["optimizer"]["bar_mass_kg"])


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", path)


def run_hold(source: InputBundle, commit: str) -> None:
    barbell = Barbell(_bar_mass(source))
    q_hold = np.asarray(source.provenance["ik"]["q_standing"], dtype=float)
    result = quasi_static_hold(
        source.spec_bytes, q_hold, barbell, duration_s=HOLD_DURATION_S
    )
    record = {
        "lift": "back_squat",
        "pose": "IK standing pose (provenance ik.q_standing)",
        "source_commit": commit,
        "bundle_spec_sha256": source.spec_sha256,
        "generator": "scripts/generate_barbell_squat_evidence.py --part hold",
        "summary": result.summary(SETTLE_WINDOW_S),
        "series": {
            "time_s": result.time_s.tolist(),
            "grf_vertical_n": result.grf_n[:, 2].tolist(),
            "hand_vertical_on_bar_n": {
                s: result.hand_force_n[s][:, 2].tolist() for s in ("L", "R")
            },
            "max_deflection_m": result.deflection_m.tolist(),
        },
    }
    _write_json(HOLD_JSON, record)


def run_bundle(source: InputBundle, commit: str) -> None:
    bundle = build_barbell_bundle(source, Barbell(_bar_mass(source)))
    provenance = dict(bundle.provenance)
    provenance["barbell_source_commit"] = commit
    InputBundle(
        spec_bytes=bundle.spec_bytes,
        coordinate_order=bundle.coordinate_order,
        dt_s=bundle.dt_s,
        q0=bundle.q0,
        v0=bundle.v0,
        efforts=bundle.efforts,
        reference_q=bundle.reference_q,
        reference_v=bundle.reference_v,
        reference_engine=bundle.reference_engine,
        provenance=provenance,
    ).save(BARBELL_BUNDLE)
    logger.info("wrote %s", BARBELL_BUNDLE)


def run_parity(commit: str) -> None:
    bundle = InputBundle.load(BARBELL_BUNDLE)
    indices = list(range(0, bundle.steps + 1, PARITY_STRIDE))
    report = instantaneous_parity(bundle, PARITY_ENGINES, indices)
    report |= {
        "lift": "back_squat",
        "bundle": str(BARBELL_BUNDLE.relative_to(ROOT)),
        "bundle_spec_sha256": bundle.spec_sha256,
        "source_commit": commit,
        "generator": "scripts/generate_barbell_squat_evidence.py --part parity",
        "tolerance": "|dX| <= 1e-6 + 1e-8 * max|X| (L1, epic #11605)",
    }
    _write_json(PARITY_JSON, report)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--part", action="append", choices=("hold", "bundle", "parity"), required=True
    )
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    source = InputBundle.load(BACK_SQUAT_BUNDLE)
    if "hold" in args.part:
        run_hold(source, args.source_commit)
    if "bundle" in args.part:
        run_bundle(source, args.source_commit)
    if "parity" in args.part:
        run_parity(args.source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
