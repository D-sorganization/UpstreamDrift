"""Regenerate (or ``--check``) every loop-nullspace golden JSON file.

Run from the repository root::

    python3 docs/research/loop_nullspace/derivations/generate_goldens.py
    python3 docs/research/loop_nullspace/derivations/generate_goldens.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as cm  # noqa: E402
import c_pm_identity  # noqa: E402
import closure_convention  # noqa: E402
import envelope_case  # noqa: E402
import five_bar_goldens  # noqa: E402
import grasp_map  # noqa: E402
import lower_body_counts  # noqa: E402
import planar_golfer  # noqa: E402
import restriction_counts  # noqa: E402
import spatial_counts  # noqa: E402
import variability_goldens  # noqa: E402

GENERATORS: dict[str, Callable[[], dict[str, Any]]] = {
    "planar_two_arm_full": planar_golfer.generate_full,
    "planar_two_arm_club_unactuated": planar_golfer.generate_club_unactuated,
    "planar_straight_arm_rank_deficient": planar_golfer.generate_straight_arm,
    "spatial_counts": spatial_counts.generate,
    "restriction_counts": restriction_counts.generate,
    "lower_body_observation": lower_body_counts.generate,
    "envelope_analytic_lp": envelope_case.generate,
    "grasp_map_worked_example": grasp_map.generate,
    "c_pm_identity": c_pm_identity.generate,
    "closure_convention": closure_convention.generate,
    "variability_sensitivity": variability_goldens.generate,
    "planar_five_bar_tangent_state_space": five_bar_goldens.generate,
}


def generate_all() -> dict[str, dict[str, Any]]:
    return {name: gen() for name, gen in GENERATORS.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if files differ")
    args = parser.parse_args()
    status = 0
    for name, payload in generate_all().items():
        target = cm.GOLDENS_DIR / f"{name}.json"
        if args.check:
            stored = (
                json.loads(target.read_text(encoding="utf-8"))
                if target.exists()
                else None
            )
            fresh = json.loads(json.dumps(payload, allow_nan=False))
            if stored != fresh:
                sys.stdout.write(f"STALE {target}\n")
                status = 1
        else:
            sys.stdout.write(f"{cm.write_json(name, payload)}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
