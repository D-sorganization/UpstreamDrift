"""Where the drawn club head sits along the shaft axis (#15150).

Two conventions meet at the club-body origin. ``motion_matching.club_models``
ends the club length at the head centre and rests the head ellipsoid on the
ground at address, its sole ``sole_offset_m`` farther down the shaft.
``model_appearance.club_head_mesh`` places the head mesh by its sole point on
the shaft axis and puts that point at the same origin. The drawn head
therefore sits one sole offset up the shaft from the head the fit grounded.
``club_head_seat_settings.json`` beside this module names the seat; the
default keeps the drawn head where it was.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(__file__).with_name("club_head_seat_settings.json")
HEAD_CENTRE = "head_centre"
SOLE_ON_SHAFT_AXIS = "sole_on_shaft_axis"
SEATS = (HEAD_CENTRE, SOLE_ON_SHAFT_AXIS)


def validate_seat(seat: object) -> str:
    """``seat`` when it is a known head seat.

    Raises ``TypeError`` for a non-string and ``ValueError`` for an unknown one.
    """
    if not isinstance(seat, str):
        raise TypeError("head_seat must be a string")
    if seat not in SEATS:
        raise ValueError(f"head_seat must be one of {SEATS}")
    return seat


@dataclass(frozen=True)
class HeadSeatSettings:
    """Validated ``club_head_seat_settings.json``."""

    seat: str

    def __post_init__(self) -> None:
        validate_seat(self.seat)


def load_head_seat_settings(path: Path | None = None) -> HeadSeatSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return HeadSeatSettings(raw["seat"])


def head_slide_m(club: Mapping[str, Any], length_m: float) -> float:
    """Distance the drawn head slides down the shaft (club ``+y``), metres.

    ``club`` is the spec club block; its ``head_seat`` overrides the settings
    file. ``head_centre`` gives 0. ``sole_on_shaft_axis`` gives the block's
    ``butt_to_sole_m`` minus ``length_m`` (the club length to the head centre).

    Raises ``TypeError`` / ``ValueError`` for a bad seat, and ``ValueError``
    when the sole seat is asked of a block whose ``butt_to_sole_m`` is missing,
    not finite or shorter than ``length_m``. Postcondition: finite and >= 0.
    """
    seat = validate_seat(club.get("head_seat", load_head_seat_settings().seat))
    if seat == HEAD_CENTRE:
        return 0.0
    stated = club.get("butt_to_sole_m")
    if isinstance(stated, bool) or not isinstance(stated, (int, float)):
        raise ValueError(f"head seat {seat!r} needs the club's butt_to_sole_m")
    slide = float(stated) - float(length_m)
    if not math.isfinite(slide) or slide < 0.0:
        raise ValueError("butt_to_sole_m must be finite and not less than length_m")
    return slide
