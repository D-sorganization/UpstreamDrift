"""Explicitly unverified adapter for the historical default-filled loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .catalog import ClubIdentity, ClubRecord, PropertyClaim, SpecificationSource

if TYPE_CHECKING:
    from .loader import ClubSpecification

_FIELDS = (
    ("length_inches", "length", "assembled", "in"),
    ("head_mass_grams", "mass", "head", "g"),
    ("shaft_mass_grams", "mass", "shaft", "g"),
    ("grip_mass_grams", "mass", "grip", "g"),
    ("loft_degrees", "loft", "head", "deg"),
    ("lie_angle_degrees", "lie", "assembled", "deg"),
    ("moment_of_inertia", "moi", "head", "g*cm^2"),
    ("center_of_gravity_mm", "cg_distance", "head", "mm"),
)


def import_legacy_specification(specification: ClubSpecification) -> ClubRecord:
    """Preserve legacy values for review without presenting defaults as facts.

    The old dataclass does not record which values were supplied. Every imported
    quantity therefore needs explicit re-attribution before physical inference.
    """
    source = SpecificationSource(
        kind="legacy",
        title="Legacy ClubSpecification import",
        license="source-unknown",
        method="Default-filled loader; original attribution unavailable",
    )
    claims = tuple(
        PropertyClaim.model_validate(
            {
                "property": prop,
                "component": component,
                "value": getattr(specification, field),
                "unit": unit,
                "status": "unverified",
                "source": source,
            }
        )
        for field, prop, component, unit in _FIELDS
    )
    club_type = specification.club_type.lower()
    if club_type not in {"driver", "wood", "hybrid", "iron", "wedge", "putter"}:
        club_type = "other"
    identity = ClubIdentity.model_validate(
        {
            "model": specification.name,
            "club_type": club_type,
            "number": specification.number,
            "build": "legacy-unverified",
        }
    )
    return ClubRecord(
        identity=identity,
        claims=claims,
        notes="Imported values may be generic defaults, not measurements. "
        f"Legacy swing weight (unverified): {specification.swing_weight}. "
        + specification.description,
    )
