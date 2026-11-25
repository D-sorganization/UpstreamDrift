"""Ball position in the stance: impact factors and straight roll per position.

``python -m src.tools.native_viewer_export.ball_position_sweep --bundle B
--swing S --out sweep.json`` moves the address ball along the target line
(``ball_position_sweep_settings.json`` ``shifts_m``, or ``--shifts``), keeps
contact on the face centre (``--ball-position-shift`` semantics), and for
each position records the impact factors and flight at zero club roll and the
club roll that lands the ball on the target line (``shot_setup``'s straight
roll solve). Replay only: the body motion is unchanged, nothing is rendered
and nothing is refitted (#15150 epic item "Ball position in the stance").
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

from src.tools.native_viewer_export import shot_setup
from src.tools.native_viewer_export.shot_dispersion import (
    SampleOutcome,
    ShotSetup,
    slopes,
)

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).with_name("ball_position_sweep_settings.json")
FORWARD_NOTE = (
    "forward of the valid range: the measured club was already slowed by the "
    "real ball here"
)
Evaluate = Callable[[ShotSetup], SampleOutcome]


@dataclass(frozen=True)
class SweepSettings:
    """Positions swept (m along the target line) and the valid forward range."""

    shifts_m: tuple[float, ...]
    forward_valid_m: float
    bracket_halvings: int = 2

    def __post_init__(self) -> None:
        limit = shot_setup.load_shot_setup_settings().max_abs_ball_shift_m
        shifts = tuple(float(s) for s in self.shifts_m)
        if not shifts or any(not math.isfinite(s) or abs(s) > limit for s in shifts):
            raise ValueError(f"every shift must be finite and within +-{limit} m")
        if not (math.isfinite(self.forward_valid_m) and self.forward_valid_m >= 0):
            raise ValueError("forward_valid_m must be finite and >= 0")
        if self.bracket_halvings < 0:
            raise ValueError("bracket_halvings must be >= 0")
        object.__setattr__(self, "shifts_m", tuple(sorted(set(shifts))))


def load_sweep_settings(path: Path | None = None) -> SweepSettings:
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return SweepSettings(
        tuple(raw["shifts_m"]), float(raw["forward_valid_m"]), raw["bracket_halvings"]
    )


@dataclass(frozen=True)
class PositionRow:
    """One ball position: factors at zero roll and the roll that flies straight.

    ``reason`` says why the flight or the straight roll is missing; ``note``
    flags a position outside the valid forward range.
    """

    shift_m: float
    metrics: dict[str, float | None] = field(default_factory=dict)
    straight_roll_deg: float | None = None
    side_at_straight_m: float | None = None
    reason: str | None = None
    note: str = ""


class NoLanding(ValueError):
    """A roll whose shot has no landing (no flight, or outside its envelope)."""


def _side(evaluate: Evaluate, shift_m: float) -> Callable[[float], float]:
    def side(roll_deg: float) -> float:
        outcome = evaluate(ShotSetup(roll_deg, shift_m))
        value = outcome.metrics.get("side_m") if outcome.reason is None else None
        if value is None:
            raise NoLanding(f"no landing at roll {roll_deg:g} deg: {outcome.reason}")
        return float(value)

    return side


def _position(
    evaluate: Evaluate,
    shift: float,
    limits: shot_setup.ShotSetupSettings,
    centre_deg: float,
    note: str,
) -> PositionRow:
    base = evaluate(ShotSetup(0.0, shift))
    if base.reason is not None:
        return PositionRow(shift, reason=base.reason, note=note)
    metrics = dict(base.metrics)
    try:
        roll, side = _straight_roll(evaluate, shift, limits, centre_deg)
    except ValueError as exc:
        return PositionRow(shift, metrics, reason=str(exc), note=note)
    return PositionRow(shift, metrics, roll, side, None, note)


def _straight_roll(
    evaluate: Evaluate,
    shift: float,
    limits: shot_setup.ShotSetupSettings,
    centre_deg: float,
) -> tuple[float, float]:
    """``shot_setup.solve_straight_roll`` about ``centre_deg``; when a bracket
    end has no landing (outside the flight model's envelope) the span is
    halved, at most ``bracket_halvings`` times, before the error is raised."""
    halvings = load_sweep_settings().bracket_halvings
    for attempt in range(halvings + 1):
        try:
            return shot_setup.solve_straight_roll(
                _side(evaluate, shift), limits, centre_deg
            )
        except NoLanding:
            if attempt == halvings:
                raise
            limits = replace(limits, solve_span_deg=limits.solve_span_deg / 2.0)
    raise AssertionError("unreachable")  # pragma: no cover


def sweep(
    evaluate: Evaluate,
    shifts_m: Sequence[float],
    limits: shot_setup.ShotSetupSettings,
    forward_valid_m: float | None = None,
) -> list[PositionRow]:
    """One :class:`PositionRow` per shift, in the order given.

    Positions are solved outward from the one nearest the address ball; each
    straight-roll bracket is centred on its solved neighbour's roll, so a
    bracket end does not leave the flight model's validity envelope just
    because the straight roll has drifted with the ball. A position without a
    flight, or without a straight roll inside its bracket, keeps a row with
    its ``reason``; the sweep never stops early.
    """
    valid = forward_valid_m
    if valid is None:
        valid = load_sweep_settings().forward_valid_m
    shifts = [float(s) for s in shifts_m]
    start = min(range(len(shifts)), key=lambda i: abs(shifts[i]))
    order = [start, *range(start + 1, len(shifts)), *range(start - 1, -1, -1)]
    rows: dict[int, PositionRow] = {}
    for i in order:
        neighbour = rows.get(i - 1 if i > start else i + 1)
        centre = 0.0
        if neighbour is not None and neighbour.straight_roll_deg is not None:
            centre = neighbour.straight_roll_deg
        note = FORWARD_NOTE if shifts[i] > valid else ""
        rows[i] = _position(evaluate, shifts[i], limits, centre, note)
    return [rows[i] for i in range(len(shifts))]


def record_for(rows: Sequence[PositionRow]) -> dict[str, Any]:
    """JSON record: the rows and each quantity's slope per centimetre of shift
    (least squares over the positions that have it; ``None`` below two)."""
    keys = sorted({k for r in rows for k in r.metrics} | {"straight_roll_deg"})
    slope_per_cm: dict[str, float | None] = {}
    for key in keys:
        pairs = [
            (
                r.shift_m,
                r.straight_roll_deg
                if key == "straight_roll_deg"
                else r.metrics.get(key),
            )
            for r in rows
        ]
        pairs = [(s, v) for s, v in pairs if v is not None]
        fit = slopes({"shift": [s for s, _ in pairs]}, [v for _, v in pairs])["shift"]
        slope_per_cm[key] = None if fit is None else fit / 100.0
    return {
        "positions": [r.__dict__ for r in rows],
        "slope_per_cm": slope_per_cm,
        "units": {"shift": "m", "angles": "deg", "distances": "m", "spin": "rpm"},
        "method": "kinematic replay with the ball moved along the target line; "
        "contact kept on the face centre; straight roll from shot_setup",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="same-input bundle .npz")
    p.add_argument("--swing", required=True, help="swing name")
    p.add_argument("--club", default="Driver", help="club label")
    p.add_argument("--out", type=Path, required=True, help="output JSON file")
    p.add_argument(
        "--shifts",
        type=float,
        nargs="+",
        default=None,
        help="ball shifts in metres (default: the settings file)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_sweep_settings()
        if args.shifts is not None:
            cfg = SweepSettings(tuple(args.shifts), cfg.forward_valid_m)
        from src.tools.native_viewer_export.core import ExportSettings, load_swing_input
        from src.tools.native_viewer_export.shot_dispersion import make_evaluator

        swing = load_swing_input(args.bundle, swing=args.swing, club=args.club)
        evaluate = make_evaluator(swing, ExportSettings())
        rows = sweep(
            evaluate,
            cfg.shifts_m,
            shot_setup.load_shot_setup_settings(),
            cfg.forward_valid_m,
        )
        args.out.write_text(json.dumps(record_for(rows), indent=2), encoding="utf-8")
    except (ValueError, FileNotFoundError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
