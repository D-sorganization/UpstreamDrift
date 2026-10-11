"""Capture foot progression report (OSV-4, #11730).

Prints the per-foot address toe-out of the tour captures (and the owner capture
O when the private data is available) as JSON:

    python3 -m scripts.foot_progression_report [--out PATH]

The owner capture resolves through ``CAPTURE_DATA_DIR`` and is skipped, with a
reason, when it is not present; it is never replaced by a guess.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.shared.python.motion_matching.foot_progression import (
    capture_foot_progression,
)
from src.shared.python.motion_matching.ground_support import capture_to_native_world
from src.shared.python.motion_matching.pipeline.address_feet import (
    NATIVE_TARGET_AXIS,
    NATIVE_UP_AXIS,
)
from src.shared.python.motion_matching.pipeline.constants import capture_path
from src.shared.python.motion_matching.tour_capture_contract import load_tour_capture

logger = logging.getLogger(__name__)
CAPTURES = ("driver", "iron", "owner")


def report_capture(name: str) -> dict[str, Any]:
    """Foot progression of one named capture, or why it is unavailable."""
    try:
        path = capture_path(name)
        capture = load_tour_capture(path)
    except (OSError, ValueError, KeyError) as exc:
        return {"capture": name, "available": False, "reason": str(exc)}
    points = capture_to_native_world(capture.points_m)
    feet = capture_foot_progression(
        points,
        capture.valid,
        capture.labels,
        up=NATIVE_UP_AXIS,
        target_axis=NATIVE_TARGET_AXIS,
    )
    stance = capture_foot_progression(
        points, capture.valid, capture.labels, up=NATIVE_UP_AXIS
    )
    return {
        "capture": name,
        "available": True,
        "source_sha256": capture.source_sha256,
        "target_axis": "native -Y (verified: lead foot at -Y, toes toward -X)",
        "feet": {side: fp.to_receipt() for side, fp in feet.items()},
        "stance_line_target_angle_deg": {
            side: round(fp.angle_deg, 2) for side, fp in stance.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = {name: report_capture(name) for name in CAPTURES}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.seterr(all="warn")
    raise SystemExit(main())
