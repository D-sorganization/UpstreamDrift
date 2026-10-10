"""OpenSim address toe-out against the tour captures (OSV-6, #11737).

Fits the OpenSim golf address (``tour_matching.address.fit_address_pose``) on
each public tour capture with ``toe_out_deg`` set to the capture's measured
toe-out (``evidence/foot_progression/capture_report.json``) and writes the
achieved per-foot angle and error. OpenSim's hip rotation is mirrored natively,
so this pathway is independent of the spec's hip axes.

Usage::

    python3 -m scripts.opensim_address_foot_progression \
        --out docs/development/full_body_models/evidence/foot_progression/opensim_address.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.engines.physics_engines.opensim.python.tour_matching.address import (
    fit_address_pose,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    align_tour_capture_to_golf_world,
)
from src.shared.python.motion_matching.pipeline.constants import CAPTURES
from src.shared.python.motion_matching.tour_capture_contract import (
    load_tour_capture,
)

LOGGER = logging.getLogger(__name__)
MODEL = Path("src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim")
REPORT = Path(
    "docs/development/full_body_models/evidence/foot_progression/capture_report.json"
)
#: OSV-6 acceptance: each foot within this many degrees of the capture.
TOLERANCE_DEG = 2.0


def capture_block(name: str, targets: dict[str, float]) -> dict[str, Any]:
    """Fit the address for capture ``name`` and score each foot against ``targets``."""
    capture, _ = align_tour_capture_to_golf_world(load_tour_capture(CAPTURES[name]))
    result = fit_address_pose(MODEL, capture, toe_out_deg=targets)
    feet = result.foot_progression or {}
    if feet.get("method") != "fk":
        raise RuntimeError(f"{name}: toe-out was not solved against OpenSim FK")
    achieved = feet["model_deg"]
    errors = {side: achieved[side] - targets[side] for side in targets}
    return {
        "targets_deg": targets,
        "foot_progression": feet,
        "error_deg": errors,
        "within_tolerance": all(abs(e) <= TOLERANCE_DEG for e in errors.values()),
        "address_qualified": result.is_qualified,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"model": str(MODEL), "tolerance_deg": TOLERANCE_DEG}
    for name in CAPTURES:
        targets = {s: f["angle_deg"] for s, f in report[name]["feet"].items()}
        out[name] = capture_block(name, targets)
        LOGGER.info("%s: error_deg %s", name, out[name]["error_deg"])
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
