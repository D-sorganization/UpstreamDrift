"""Validity envelope for the Rust-backed ``BallFlightSimulator`` path (#14621).

The impact solver can produce launch spin well above measured driver ranges
(face-to-path 10–15 deg is typically under ~3500 rpm). Without spin decay or
input limits, extreme sidespin can turn the ball past 90 deg so down-range
distance reverses — a model artefact, not a result.

This module centralises the launch spin cap, trajectory monotonicity check,
and landing azimuth limit used by ``SwingBallFlightPipeline`` and reporting
layers (SHOW-C7 ``flight_report``).
"""

from __future__ import annotations

import functools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.physics.ball_launch_conditions import LaunchConditions

SETTINGS_PATH = Path(__file__).with_name("ball_flight_validity_settings.json")
_DRIVER_CLASS = "driver"


@dataclass(frozen=True)
class LaunchSpinEnvelope:
    """Launch-spin cap of one club class (#15443).

    Invariant: ``max_rpm`` is positive and finite.
    """

    club_class: str
    label: str
    max_rpm: float

    def __post_init__(self) -> None:
        require(
            math.isfinite(self.max_rpm) and self.max_rpm > 0.0,
            "max_rpm must be positive and finite",
            self.max_rpm,
        )


@functools.lru_cache(maxsize=1)
def _settings() -> dict[str, Any]:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _envelope_of(club_class: str) -> LaunchSpinEnvelope:
    entry = _settings()["classes"][club_class]
    return LaunchSpinEnvelope(
        club_class=club_class,
        label=str(entry["label"]),
        max_rpm=float(entry["max_launch_spin_rpm"]),
    )


def report_margin_rpm() -> float:
    """Spin excess [rpm] tolerated before a report calls a flight invalid."""
    return float(_settings()["report_margin_rpm"])


def launch_spin_envelope(club: str | None) -> LaunchSpinEnvelope:
    """Launch-spin envelope for the club named ``club``.

    Preconditions: ``club`` is a string or ``None``.
    Postcondition: an unknown, empty or ``None`` club gets the fallback class
    of the settings file (the driver envelope, the most conservative one).

    Raises:
        TypeError: ``club`` is neither a string nor ``None``.
    """
    club_class = club_class_of(club)
    if club_class is None:
        club_class = str(_settings()["fallback_club_class"])
    return _envelope_of(club_class)


def club_class_of(club: str | None) -> str | None:
    """Club class whose name pattern matches ``club``, or ``None`` if none does.

    ``None`` means :func:`launch_spin_envelope` would use the fallback class.

    Raises:
        TypeError: ``club`` is neither a string nor ``None``.
    """
    if club is not None and not isinstance(club, str):
        raise TypeError(f"club must be a string or None, got {type(club).__name__}")
    name = (club or "").strip().lower()
    for entry in _settings()["club_name_patterns"]:
        if re.search(entry["pattern"], name):
            return str(entry["club_class"])
    return None


# Tour-labelled driver launch spin for face-to-path 10–15 deg (issue #14621);
# alias of the driver value in the settings file.
MAX_DRIVER_LAUNCH_SPIN_RPM: float = _envelope_of(_DRIVER_CLASS).max_rpm

# Landing azimuth guard (SHOW-C7 flight card; margin over typical driver flights).
MAX_LANDING_AZIMUTH_DEG: float = 45.0

_DOWN_RANGE_REVERSAL_TOL_M: float = 1e-6


@dataclass(frozen=True)
class FlightValidityVerdict:
    """Whether a trajectory is inside the flight model validity envelope."""

    valid: bool
    invalid_reason: str = ""


def clamp_launch_spin_rate(
    launch: LaunchConditions, max_rpm: float = MAX_DRIVER_LAUNCH_SPIN_RPM
) -> tuple[LaunchConditions, bool]:
    """Cap launch spin magnitude while preserving spin axis direction."""
    require(max_rpm > 0.0, "max_rpm must be positive", max_rpm)
    if launch.spin_rate <= max_rpm:
        return launch, False
    return (
        LaunchConditions(
            velocity=launch.velocity,
            launch_angle=launch.launch_angle,
            azimuth_angle=launch.azimuth_angle,
            spin_rate=float(max_rpm),
            spin_axis=np.asarray(launch.spin_axis, dtype=float),
        ),
        True,
    )


def landing_azimuth_deg(path: np.ndarray) -> float:
    """Return landing azimuth magnitude in degrees (0 = straight down-range)."""
    require(path.ndim == 2 and path.shape[1] == 3, "path must be (n, 3)", path.shape)
    end_x = float(path[-1, 0])
    end_y = float(path[-1, 1])
    if end_x <= 0.0:
        return 90.0
    return float(math.degrees(math.atan2(abs(end_y), end_x)))


def assess_trajectory_validity(
    path: np.ndarray,
    *,
    max_landing_azimuth_deg: float = MAX_LANDING_AZIMUTH_DEG,
) -> FlightValidityVerdict:
    """Flag trajectories outside the model validity envelope."""
    require(
        path.ndim == 2 and path.shape[1] == 3 and len(path) >= 2, "path invalid", path
    )
    if float(np.min(np.diff(path[:, 0]))) < -_DOWN_RANGE_REVERSAL_TOL_M:
        return FlightValidityVerdict(
            valid=False,
            invalid_reason=(
                "down-range reverses (outside the flight model's validity)"
            ),
        )
    azimuth = landing_azimuth_deg(path)
    end_x = float(path[-1, 0])
    if end_x <= 0.0 or azimuth > max_landing_azimuth_deg:
        return FlightValidityVerdict(
            valid=False,
            invalid_reason=(
                f"ball lands {azimuth:.0f} deg off the target line "
                f"(limit {max_landing_azimuth_deg:.0f}; "
                "outside the flight model's validity)"
            ),
        )
    return FlightValidityVerdict(valid=True)
