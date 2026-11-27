"""Impact-model inputs of the club that struck the ball (#15443).

The flight pipeline defaults describe a driver (head 0.200 kg, scalar moment
of inertia, driver COR). A 7-iron head is heavier, so launching an iron with
those defaults misreads the ball speed. :func:`club_impact_for` returns the
head mass, head inertia and COR law of a catalogue club (``club_models.CLUBS``);
for the clubs in ``pipeline_default_clubs`` it returns ``None`` and the
pipeline defaults stay in force, so the driver path is unchanged. Numbers and
their sources: ``club_impact_settings.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec
from src.shared.python.physics.impact_model.types import RigidGearHead

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).with_name("club_impact_settings.json")


@dataclass(frozen=True)
class ClubImpact:
    """Head mass [kg], scalar head inertia [kg m^2] and COR law of one club.

    The COR is ``cor_intercept - cor_slope_s_per_m * v_n`` in the normal
    approach speed ``v_n`` [m/s] (:meth:`cor_at`); a zero slope is a constant.
    """

    head_mass_kg: float
    head_moi_kg_m2: float
    cor_intercept: float
    cor_slope_s_per_m: float = 0.0

    def __post_init__(self) -> None:
        require(self.head_mass_kg > 0.0, "head mass must be positive")
        require(self.head_moi_kg_m2 > 0.0, "head inertia must be positive")
        require(0.0 < self.cor_intercept <= 1.0, "COR intercept must lie in (0, 1]")
        require(self.cor_slope_s_per_m >= 0.0, "COR must not rise with speed")

    def cor_at(self, normal_speed_mps: float) -> float:
        """COR of a strike whose face approaches the ball at ``normal_speed_mps``.

        Precondition: the speed is finite and >= 0. Postcondition: the COR
        lies in (0, 1].
        """
        require(
            math.isfinite(normal_speed_mps) and normal_speed_mps >= 0.0,
            "normal approach speed must be finite and >= 0",
            normal_speed_mps,
        )
        cor = self.cor_intercept - self.cor_slope_s_per_m * normal_speed_mps
        ensure(0.0 < cor <= 1.0, "COR must lie in (0, 1]", cor)
        return cor


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
    value has positive mass and inertia and a COR intercept in (0, 1].
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
    law = cfg["cor_by_club"][name]
    result = ClubImpact(
        club.head_mass_kg,
        float(cfg["reference_head_moi_kg_m2"]) * ratio,
        float(law["intercept"]),
        float(law["slope_s_per_m"]),
    )
    ensure(result.head_moi_kg_m2 > 0.0, "inertia must be positive")
    return result


def club_impact_from_spec(club: Mapping[str, object] | None) -> ClubImpact | None:
    """:func:`club_impact_for` the ``name`` of a bundle spec's ``club`` entry."""
    name = None if club is None else club.get("name")
    return club_impact_for(name if isinstance(name, str) else None)


def rigid_gear_head_for(name: str | None, path: Path | None = None) -> RigidGearHead:
    """Rigid-contact gear head of the club ``name`` from the settings file.

    A club without an entry (or ``None``) gets ``rigid_gear_default_club``'s
    head, as :func:`club_impact_for` launches it with the driver defaults.
    Postcondition: the head has positive inertia and a CG depth >= 0.
    """
    cfg = _load(path)
    heads = cfg["rigid_gear_heads"]
    entry = heads.get(name) if isinstance(name, str) else None
    if entry is None:
        entry = heads[cfg["rigid_gear_default_club"]]
    inertia = tuple(float(v) for v in entry["inertia_diagonal_kg_m2"])
    require(len(inertia) == 3, "gear head inertia needs three values", inertia)
    return RigidGearHead(
        inertia_diagonal_kg_m2=(inertia[0], inertia[1], inertia[2]),
        cg_depth_m=float(entry["cg_depth_m"]),
    )
