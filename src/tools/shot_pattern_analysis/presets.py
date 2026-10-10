"""Illustrative club inputs shared by the desktop and command-line tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ClubPreset:
    """A user-facing club hypothesis, not a measured golfer average."""

    id: str
    label: str
    club_speed_mps: float
    loft_deg: float
    attack_angle_deg: float
    lie_deg: float
    clubhead_mass_kg: float
    shaft_lean_deg: float = 0.0
    illustrative: bool = True
    source_assumption: str = (
        "Illustrative research starting value; not measured delivery data."
    )


CLUB_PRESETS: Mapping[str, ClubPreset] = MappingProxyType(
    {
        "driver": ClubPreset("driver", "Driver", 45.0, 12.8, -0.9, 58.5, 0.200),
        "seven_iron": ClubPreset("seven_iron", "7-Iron", 36.0, 24.0, -4.0, 63.0, 0.272),
        "pitching_wedge": ClubPreset(
            "pitching_wedge", "Pitching Wedge", 32.0, 36.7, -5.0, 64.0, 0.300
        ),
    }
)


def get_club_preset(preset_id: str) -> ClubPreset:
    """Return an illustrative preset by stable ID."""
    try:
        return CLUB_PRESETS[preset_id]
    except KeyError as exc:
        raise ValueError(f"unknown club preset: {preset_id}") from exc


__all__ = ["CLUB_PRESETS", "ClubPreset", "get_club_preset"]
