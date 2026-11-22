"""Command line of the shot dispersion study (#15150): JSON record and figure.

``python -m src.tools.native_viewer_export.shot_dispersion_cli --bundle B
--swing S --club C --out DIR`` replays the swing with random club roll and ball
position, writes ``dispersion.json`` and ``dispersion.png`` and renders no video.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
from typing import Any

from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.display_units import (
    DISTANCE_UNITS,
    DisplayUnits,
    load_display_units,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Parser; defaults come from ``shot_dispersion_settings.json``."""
    d = sd.load_dispersion_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="same-input bundle .npz")
    p.add_argument("--swing", required=True, help="swing name")
    p.add_argument("--club", default="Driver", help="club label")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--n", type=int, default=d.n_shots, help="number of shots")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--roll-sd", type=float, default=d.club_roll_sd_deg, help="deg")
    p.add_argument("--ball-sd", type=float, default=d.ball_position_sd_m, help="m")
    p.add_argument("--base-roll", type=float, default=d.base_club_roll_deg)
    p.add_argument("--base-ball-shift", type=float, default=d.base_ball_shift_m)
    p.add_argument(
        "--distance-unit",
        choices=DISTANCE_UNITS,
        default=None,
        help="unit of the figure's distances (default: display units settings)",
    )
    return p


def record_for(
    cfg: sd.DispersionSettings, result: sd.DispersionResult, units: DisplayUnits
) -> dict[str, Any]:
    """JSON record: SI values (``*_m``) plus the unit names of the figure."""
    return {
        "seed": cfg.seed,
        **result.to_dict(),
        "units": {
            "distance_values": "m",
            "angle_values": "deg",
            "display_distance_unit": units.distance_unit,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.tools.native_viewer_export.core import ExportSettings, load_swing_input
    from src.tools.native_viewer_export.shot_dispersion_plot import draw_dispersion

    try:
        cfg = replace(
            sd.load_dispersion_settings(),
            n_shots=args.n,
            seed=args.seed,
            club_roll_sd_deg=args.roll_sd,
            ball_position_sd_m=args.ball_sd,
            base_club_roll_deg=args.base_roll,
            base_ball_shift_m=args.base_ball_shift,
        )
        units = load_display_units()
        if args.distance_unit:
            units = replace(units, distance_unit=args.distance_unit)
        swing = load_swing_input(args.bundle, swing=args.swing, club=args.club)
        result = sd.run_dispersion(swing, ExportSettings(), sd.sample_setups(cfg))
        args.out.mkdir(parents=True, exist_ok=True)
        record = record_for(cfg, result, units)
        (args.out / "dispersion.json").write_text(json.dumps(record, indent=2))
        draw_dispersion(result, cfg, args.out / "dispersion.png", units)
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
