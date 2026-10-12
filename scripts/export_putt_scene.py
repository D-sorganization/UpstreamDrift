"""Write the public-explorer putt export artifact (issue #12319).

Usage::

    python -m scripts.export_putt_scene --out putt_scene.json [--speed 2.5 ...]

The artifact is the canonical ``ud.putt_scene/1`` file that the public
explorer's Putting tab vendors through the mirror pattern.  Headless: no Qt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.tools.putting_green_gui._scene_builder import PuttConfig, build_putt_scene
from src.tools.putting_green_gui._scene_export import (
    PuttExportError,
    export_putt_scene,
)

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    defaults = PuttConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--speed", type=float, default=defaults.putter_speed_ms)
    parser.add_argument("--aim-deg", type=float, default=defaults.aim_deg)
    parser.add_argument("--distance-m", type=float, default=defaults.cup_distance_m)
    parser.add_argument("--stimp", type=float, default=defaults.stimp)
    parser.add_argument("--slope-deg", type=float, default=defaults.slope_deg)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build a scene from CLI controls and export it; return a process code."""
    args = _parser().parse_args(argv)
    config = PuttConfig(
        putter_speed_ms=args.speed,
        aim_deg=args.aim_deg,
        cup_distance_m=args.distance_m,
        stimp=args.stimp,
        slope_deg=args.slope_deg,
    )
    try:
        export_putt_scene(build_putt_scene(config), args.out)
    except (ValueError, PuttExportError) as exc:  # PuttExportError is a ValueError
        logger.error("putt export failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
