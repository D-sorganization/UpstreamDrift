"""Impact-model inputs of the club that struck the ball (#15443).

The flight pipeline defaults describe a driver (head 0.200 kg, scalar moment
of inertia, driver COR). A 7-iron head is heavier, so launching an iron with
those defaults misreads the ball speed. :func:`club_impact_for` returns the
head mass, head inertia and COR of a catalogue club (``club_models.CLUBS``);
for the clubs in ``pipeline_default_clubs`` it returns ``None`` and the
pipeline defaults stay in force, so the driver path is unchanged. Numbers and
their sources: ``club_impact_settings.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).with_name("club_impact_settings.json")


@dataclass(frozen=True)
class ClubImpact:
    """Head mass [kg], scalar head inertia [kg m^2] and COR of one club."""

    head_mass_kg: float
    head_moi_kg_m2: float
    cor: float

    def __post_init__(self) -> None:
        require(self.head_mass_kg > 0.0, "head mass must be positive")
        require(self.head_moi_kg_m2 > 0.0, "head inertia must be positive")
        require(0.0 < self.cor <= 1.0, "COR must lie in (0, 1]")


def _load(path: Path | None) -> dict:
    return json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))


def _face_plane_k2(club: ClubSpec) -> float:
    """Mean squared gyration radius about the two axes normal to the face."""
    _, ky, kz = club.head_gyration_m
    return 0.5 * (ky * ky + kz * kz)


def club_impact_for(name: str | None, path: Path | None = None) -> ClubImpact | None:
    """Impact inputs of the catalogue club ``name``, ``None`` for pipeline defaults.

    ``None`` is returned for the clubs the pipeline defaults describe (the
    driver) and, with a logged warning, for an unknown or missing name, so a
    club the catalogue lacks is launched as before. Postcondition: a returned
    value has positive mass and inertia and a COR in (0, 1].
    """
    cfg = _load(path)
    if name in cfg["pipeline_default_clubs"]:
        return None
    club = CLUBS.get(name) if isinstance(name, str) else None
    if club is None or name not in cfg["cor_by_club"]:
        logger.warning(
            "club %r has no impact properties; launching with the driver defaults",
            name,
        )
        return None
    ref = CLUBS[cfg["reference_club"]]
    ratio = (club.head_mass_kg * _face_plane_k2(club)) / (
        ref.head_mass_kg * _face_plane_k2(ref)
    )
    result = ClubImpact(
        club.head_mass_kg,
        float(cfg["reference_head_moi_kg_m2"]) * ratio,
        float(cfg["cor_by_club"][name]),
    )
    ensure(result.head_moi_kg_m2 > 0.0, "inertia must be positive")
    return result


def club_impact_from_spec(club: Mapping[str, object] | None) -> ClubImpact | None:
    """:func:`club_impact_for` the ``name`` of a bundle spec's ``club`` entry."""
    name = None if club is None else club.get("name")
    return club_impact_for(name if isinstance(name, str) else None)
