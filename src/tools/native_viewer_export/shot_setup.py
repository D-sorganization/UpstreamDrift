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
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.tools.native_viewer_export.ball import BallPlacement
    from src.tools.native_viewer_export.core import SwingInput

SETTINGS_PATH = Path(__file__).with_name("shot_setup_settings.json")
POSITION_NOTE = (
    "forward of the captured ball the measured club was already slowed by the "
    "real ball; forward positions are valid only for the first few centimetres"
)


@dataclass(frozen=True)
class ShotSetupSettings:
    """Limits read from ``shot_setup_settings.json`` (both positive, finite)."""

    max_abs_roll_deg: float
    max_abs_ball_shift_m: float


def load_shot_setup_settings(path: Path | None = None) -> ShotSetupSettings:
    """Read the limits; ``KeyError`` for a missing key, ``ValueError`` if invalid."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    limits = ShotSetupSettings(
        float(raw["max_abs_roll_deg"]), float(raw["max_abs_ball_shift_m"])
    )
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


def validate_shot_setup(roll_deg: Any, shift_m: Any) -> None:
    """Raise ``TypeError`` for a non-number, ``ValueError`` for a value that is
    not finite or beyond the limits of ``shot_setup_settings.json``."""
    limits = load_shot_setup_settings()
    _check("club_roll_in_hands_deg", roll_deg, limits.max_abs_roll_deg)
    _check("ball_position_shift_m", shift_m, limits.max_abs_ball_shift_m)


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
