"""Opt-in shot setup of the native export: club roll and ball position (#15150).

Two display settings, both off by default (zero changes nothing):

* ``club_roll_in_hands_deg`` -- a rigid turn of the club about its shaft axis
  relative to the hands, replaying the same body motion. It is written into
  the spec as ``club["face_roll_deg"]`` (plus ``club["roll_in_hands_deg"]``),
  which every engine and the impact card, flight and ball placement read.
  Positive opens the face: on the matched right-handed driver swing +1 degree
  of roll turns the face about +0.73 degrees to the right and adds about 0.67
  degrees of loft. The shaft is treated as rigid (no bend); no joint angle
  changes.
* ``ball_position_shift_m`` -- moves the nominal address ball along the
  horizontal target line (positive toward the target) before the strike
  placement, so the ball is still placed where the face centre passes.
* ``shot_shape`` (``straight``, ``draw``, ``fade``) -- solves the club roll in
  the hands instead of taking a number. ``straight`` is the roll at which the
  flight model lands the ball on the target line; ``draw`` and ``fade`` are
  that roll minus / plus ``shape_roll_step_deg``. The names are the ball curve
  of a right-handed golfer: a draw curves left (more closed, more negative
  roll), a fade curves right (more open, more positive roll). It excludes a
  non-zero ``club_roll_in_hands_deg`` and combines with the ball shift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from src.tools.native_viewer_export.core import ExportSettings
    from src.tools.native_viewer_export.ball import BallPlacement
    from src.tools.native_viewer_export.core import SwingInput

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).with_name("shot_setup_settings.json")
POSITION_NOTE = (
    "forward of the captured ball the measured club was already slowed by the "
    "real ball; forward positions are valid only for the first few centimetres"
)


SHAPES = ("straight", "draw", "fade")
_SHAPE_SIGN = {"straight": 0.0, "draw": -1.0, "fade": 1.0}


@dataclass(frozen=True)
class ShotSetupSettings:
    """Values read from ``shot_setup_settings.json`` (all positive, finite)."""

    max_abs_roll_deg: float
    max_abs_ball_shift_m: float
    straight_tolerance_m: float = 0.25
    solve_span_deg: float = 6.0
    max_solve_steps: int = 12
    shape_roll_step_deg: float = 2.0


def load_shot_setup_settings(path: Path | None = None) -> ShotSetupSettings:
    """Read the limits; ``KeyError`` for a missing key, ``ValueError`` if invalid."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    limits = ShotSetupSettings(**raw)
    for name, value in vars(limits).items():
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    return limits


def _check(name: str, value: Any, limit: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not (math.isfinite(value) and abs(value) <= limit):
        raise ValueError(
            f"{name} must be finite with |value| <= {limit:g}, got {value}"
        )


def validate_shot_setup(roll_deg: Any, shift_m: Any, shape: Any = None) -> None:
    """Raise ``TypeError`` for a non-number (or a non-string ``shape``),
    ``ValueError`` for a value that is not finite or beyond the limits of
    ``shot_setup_settings.json``, an unknown ``shape`` or a ``shape`` together
    with a non-zero roll (the shape solves the roll)."""
    limits = load_shot_setup_settings()
    _check("club_roll_in_hands_deg", roll_deg, limits.max_abs_roll_deg)
    _check("ball_position_shift_m", shift_m, limits.max_abs_ball_shift_m)
    if shape is None:
        return
    if not isinstance(shape, str):
        raise TypeError(f"shot_shape must be a string, got {type(shape).__name__}")
    if shape not in SHAPES:
        raise ValueError(f"shot_shape must be one of {SHAPES}, got {shape!r}")
    if roll_deg != 0.0:
        raise ValueError(
            "shot_shape solves the roll: club_roll_in_hands_deg must be 0, "
            f"got {roll_deg}"
        )


def spec_with_roll_in_hands(spec_bytes: bytes, offset_deg: float) -> bytes:
    """``spec_bytes`` whose club roll carries ``offset_deg`` more degrees.

    Postcondition: ``club["face_roll_deg"]`` is the resolved roll plus the
    offset and ``club["roll_in_hands_deg"]`` records the offset. Raises
    ``ValueError`` for a spec without a club body.
    """
    from src.shared.python.model_appearance.club_assembly import assembly_from_spec

    spec = json.loads(spec_bytes)
    assembly = assembly_from_spec(spec) if "bodies" in spec else None
    if assembly is None:
        raise ValueError("spec has no club body to roll")
    club = dict(spec.get("club") or {})
    club.update(
        face_roll_deg=assembly.face_roll_deg + offset_deg,
        roll_in_hands_deg=float(offset_deg),
    )
    return json.dumps({**spec, "club": club}).encode()


def apply_club_roll(swing: SwingInput, offset_deg: float) -> SwingInput:
    """``swing`` with its club turned ``offset_deg`` about the shaft.

    Returns ``swing`` itself for a zero offset. Only the bundle's spec changes,
    so every consumer (render, card, flight, ball) sees the same rolled club.
    """
    if offset_deg == 0.0:
        return swing
    spec_bytes = spec_with_roll_in_hands(swing.bundle.spec_bytes, offset_deg)
    return replace(swing, bundle=replace(swing.bundle, spec_bytes=spec_bytes))


def target_direction(swing: SwingInput) -> np.ndarray:
    """Unit horizontal target-line direction of the address-square face."""
    from src.tools.native_viewer_export.impact_card import address_target_direction

    return np.asarray(address_target_direction(swing), dtype=float)


def shift_ball(swing: SwingInput, ball: BallPlacement, shift_m: float) -> BallPlacement:
    """``ball`` moved ``shift_m`` along the target line (ground height kept).

    Returns ``ball`` itself for a zero shift. Raises ``ValueError`` when the
    swing has no target line.
    """
    if shift_m == 0.0:
        return ball
    moved = np.asarray(ball.position_m) + shift_m * target_direction(swing)
    return replace(ball, position_m=(float(moved[0]), float(moved[1]), float(moved[2])))


def solve_straight_roll(
    side_m: Callable[[float], float],
    limits: ShotSetupSettings,
    centre_deg: float = 0.0,
) -> tuple[float, float]:
    """The roll (degrees) where ``side_m`` is zero, and the side found there.

    Bracketed false position (Illinois) on ``centre_deg +- solve_span_deg``
    with at most ``max_solve_steps`` evaluations, ending when
    ``|side| <= straight_tolerance_m``. Postcondition: the returned roll lies in
    the bracket. Raises ``ValueError`` for a non-finite side, a bracket without
    a sign change or an exhausted step budget.
    """
    steps = 0

    def probe(roll: float) -> float:
        nonlocal steps
        steps += 1
        value = float(side_m(roll))
        if not math.isfinite(value):
            raise ValueError(f"landing side must be finite, got {value} at {roll} deg")
        return value

    if not math.isfinite(centre_deg):
        raise ValueError(f"centre_deg must be finite, got {centre_deg}")
    lo, hi = centre_deg - limits.solve_span_deg, centre_deg + limits.solve_span_deg
    f_lo, f_hi = probe(lo), probe(hi)
    for roll, value in ((lo, f_lo), (hi, f_hi)):
        if abs(value) <= limits.straight_tolerance_m:
            return roll, value
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"no sign change of the landing side in {centre_deg:g} "
            f"+-{limits.solve_span_deg:g} deg of roll ({f_lo:.1f} m at {lo:g}, "
            f"{f_hi:.1f} m at {hi:g})"
        )
    last = 0
    while steps < limits.max_solve_steps:
        mid = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        f_mid = probe(mid)
        if abs(f_mid) <= limits.straight_tolerance_m:
            return mid, f_mid
        if f_mid * f_hi > 0.0:
            hi, f_hi = mid, f_mid
            f_lo = f_lo / 2.0 if last == -1 else f_lo
            last = -1
        else:
            lo, f_lo = mid, f_mid
            f_hi = f_hi / 2.0 if last == 1 else f_hi
            last = 1
    raise ValueError(
        f"landing side not within {limits.straight_tolerance_m:g} m after {steps} steps"
    )


def roll_for_shape(
    shape: str, side_m: Callable[[float], float], limits: ShotSetupSettings
) -> tuple[float, float, float]:
    """``(roll, straight_roll, side_at_straight)`` of ``shape``.

    ``draw`` / ``fade`` are the straight roll minus / plus
    ``shape_roll_step_deg`` (right-handed golfer; a draw curves left).
    """
    validate_shot_setup(0.0, 0.0, shape)
    straight, side = solve_straight_roll(side_m, limits)
    return shape_roll(shape, straight, limits), straight, side


def shape_roll(shape: str, straight_deg: float, limits: ShotSetupSettings) -> float:
    """Club roll (degrees) of ``shape`` given the solved straight roll.

    ``draw`` / ``fade`` are ``straight_deg`` minus / plus
    ``shape_roll_step_deg``; ``ValueError`` for an unknown ``shape``.
    """
    validate_shot_setup(0.0, 0.0, shape)
    return straight_deg + _SHAPE_SIGN[shape] * limits.shape_roll_step_deg


def lateral_of_flight(flight: Any) -> float:
    """Right-positive landing side (metres) of a ``FlightPlayback``.

    Raises ``ValueError`` stating the flight's status when it has no landing.
    """
    if flight is None or flight.lateral_m is None:
        why = "not requested" if flight is None else flight.status
        raise ValueError(f"cannot solve shot_shape, ball flight unavailable: {why}")
    return float(flight.lateral_m)


def _trial_lateral(swing: SwingInput, settings: ExportSettings, roll: float) -> float:
    """Landing side of the export's flight for ``roll`` (no rendering)."""
    from src.tools.native_viewer_export import contact_frame, core, flight_tracer
    from src.tools.native_viewer_export.ball import attach_ball

    tracer = settings.ball_flight or flight_tracer.load_tracer_settings()
    trial = replace(
        settings, club_roll_in_hands_deg=roll, ball_flight=tracer, shot_shape=None
    )
    rolled, _ = attach_ball(apply_club_roll(swing, roll), trial)
    if trial.impact_time_s is not None or rolled.has_impact_provenance:
        rolled, contact = contact_frame.refine(
            rolled, core.impact_time_s(rolled, trial)
        )
        if contact is not None:
            trial = replace(trial, impact_time_s=contact)
    return lateral_of_flight(flight_tracer.flight_for_export(rolled, trial))


def solve_shot_shape(
    swing: SwingInput, settings: ExportSettings
) -> tuple[ExportSettings, dict[str, Any] | None]:
    """``settings`` with ``shot_shape`` turned into a club roll, plus a record.

    Returns ``(settings, None)`` unchanged without a shape. Otherwise the
    solved roll replaces ``club_roll_in_hands_deg`` (and ``shot_shape`` is
    cleared, the roll now being explicit) and the record holds
    ``shot_shape``, ``club_roll_in_hands_deg``, ``straight_roll_deg`` and
    ``side_m_at_straight``. Raises ``ValueError`` when no flight can be
    computed or the roll cannot be bracketed.
    """
    shape = settings.shot_shape
    if shape is None:
        return settings, None
    roll, straight, side = roll_for_shape(
        shape,
        lambda r: _trial_lateral(swing, settings, r),
        load_shot_setup_settings(),
    )
    logger.info("shot shape %s: club roll %.2f deg in hands", shape, roll)
    record = {
        "shot_shape": shape,
        "club_roll_in_hands_deg": roll,
        "straight_roll_deg": straight,
        "side_m_at_straight": side,
    }
    return replace(settings, club_roll_in_hands_deg=roll, shot_shape=None), record


def with_setup_record(
    ball_record: dict[str, Any] | None, record: dict[str, Any] | None
) -> dict[str, Any] | None:
    """``ball_record`` carrying ``record`` under ``shot_setup`` (if both exist)."""
    if ball_record is None or record is None:
        return ball_record
    return {**ball_record, "shot_setup": record}


def hud_lines(settings: ExportSettings) -> tuple[str, ...]:
    """One ``Setup: ...`` HUD line for a non-zero roll or ball shift, else ``()``."""
    parts = []
    if settings.club_roll_in_hands_deg:
        parts.append(f"club roll {settings.club_roll_in_hands_deg:+.1f} deg in hands")
    shift = settings.ball_position_shift_m
    if shift:
        where = "forward" if shift > 0 else "back"
        parts.append(f"ball {shift * 100:+.1f} cm {where}")
    return (f"Setup: {', '.join(parts)}",) if parts else ()
