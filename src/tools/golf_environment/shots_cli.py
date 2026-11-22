"""Model shots on a hole from a swing bundle, no video rendering (#15150).

``python -m src.tools.golf_environment.shots_cli --bundle B --swing S --club C
--roll -4 --roll 0 --labels Draw,Straight --out DIR`` runs each roll through the
exporter's own steps (club roll, ball placement, contact refinement, flight) and
writes ``hole_shots.png`` and ``hole_shots.json`` to ``DIR``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.tools.golf_environment import shots as sh

logger = logging.getLogger(__name__)
CARD_KEYS = {"face_angle_deg": "face_deg", "club_path_deg": "path_deg"}


def flight_for_roll(
    bundle: Path, swing_name: str, club: str, roll_deg: float, shift_m: float
) -> tuple[sh.TargetFlight, dict[str, Any]]:
    """Flight and table row of one shot, in the exporter's own order.

    Raises ``ValueError`` when the models give no drawable flight.
    """
    from src.tools.native_viewer_export import contact_frame, flight_tracer, shot_setup
    from src.tools.native_viewer_export.ball import attach_ball
    from src.tools.native_viewer_export.core import (
        ExportSettings,
        impact_time_s,
        load_swing_input,
    )
    from src.tools.native_viewer_export.impact_card import ImpactCard, impact_card_for

    settings = ExportSettings(
        club_roll_in_hands_deg=roll_deg,
        ball_position_shift_m=shift_m,
        ball_flight=flight_tracer.load_tracer_settings(),
    )
    swing = load_swing_input(bundle, swing=swing_name, club=club)
    swing = shot_setup.apply_club_roll(swing, roll_deg)
    swing, _ = attach_ball(swing, settings)
    swing, contact = contact_frame.refine(swing, impact_time_s(swing, settings))
    if contact is not None:
        settings = replace(settings, impact_time_s=contact)
    playback = flight_tracer.flight_for_export(swing, settings)
    if (
        playback is None
        or playback.path_m is None
        or playback.carry_m is None
        or playback.lateral_m is None
        or playback.axes_m is None
    ):
        raise ValueError(f"no flight: {getattr(playback, 'status', 'unavailable')}")
    path, axes = np.asarray(playback.path_m), np.asarray(playback.axes_m)
    flight = sh.TargetFlight(
        (path - path[0]) @ axes.T, float(playback.carry_m), float(playback.lateral_m)
    )
    row: dict[str, Any] = {
        "roll_deg": roll_deg,
        "ball_shift_m": shift_m,
        "carry_m": flight.carry_m,
        "side_m_right_positive": flight.lateral_m,
        "apex_m": playback.apex_m,
    }
    card = impact_card_for(swing, impact_time_s(swing, settings), swing.ball)
    if isinstance(card, ImpactCard):
        values = {r.key: r.value for r in card.rows}
        row.update({name: values.get(key) for key, name in CARD_KEYS.items()})
    return flight, row


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--swing", required=True)
    p.add_argument("--club", required=True)
    p.add_argument("--roll", type=float, action="append", default=None)
    p.add_argument("--ball-shift", type=float, action="append", default=None)
    p.add_argument("--labels", default=None, help="comma-separated, one per roll")
    p.add_argument("--out", type=Path, required=True)
    return p


def _per_shot(values: list[float] | None, count: int, name: str) -> list[float]:
    values = values or [0.0]
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} must be given once or once per roll")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.tools.golf_environment.hole_figure import write_hole_figure

    rolls = args.roll or [0.0]
    shifts = _per_shot(args.ball_shift, len(rolls), "--ball-shift")
    labels = args.labels.split(",") if args.labels else [f"Roll {r:+g}" for r in rolls]
    settings = sh.load_shots_settings()
    palette = settings["shot_colours_rgba"]
    if len(labels) != len(rolls) or len(rolls) > len(palette):
        raise ValueError("need one label per roll and at most one colour per roll")
    hole = sh.hole_from_settings(settings)
    results = [
        flight_for_roll(args.bundle, args.swing, args.club, r, s)
        for r, s in zip(rolls, shifts, strict=True)
    ]
    shots = sh.shots_on_hole(
        [f for f, _ in results], hole, labels, [tuple(c) for c in palette[: len(rolls)]]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_hole_figure(shots, hole, args.out / "hole_shots.png", settings)
    table = {
        "hole_sources": settings["sources"],
        "shots": [
            {"label": lab, **row} for lab, (_, row) in zip(labels, results, strict=True)
        ],
    }
    (args.out / "hole_shots.json").write_text(json.dumps(table, indent=2))
    for entry in table["shots"]:
        logger.info("%s", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
