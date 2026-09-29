"""Named grind presets with per-number provenance (issue #8609).

This package has already had one honesty failure of exactly the kind
these records exist to prevent (#7999): a number that looked measured
but was fabricated.  Every preset therefore states, per field, whether
the value is a patent example, an OEM-published figure, derived from
one, or an engineering estimate - and any field without a record is
reported as an estimate rather than quietly inheriting credibility from
its neighbours.

Known gaps, stated rather than papered over:

* The Acushnet patent family (US10143900B2 / US10661131B2) is the source
  of the three example bounce angles.  The remaining sole dimensions of
  those presets are chosen inside the claimed preferred ranges - they
  are *not* the patent's own example tables, and are marked estimated.
* Retail presets carry published loft, lie, marketed bounce and head
  mass.  No OEM publishes sole entry height, camber area or sole radii,
  so those are estimated.
* No Ping "MS" grind could be verified in any generation, so none is
  offered here.  Ping used SS/WS/TS/ES on Glide 2.0-3.0 and single
  letters from Glide 4.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from .bounce import GeometricBounce, MarketedBounce, geometric_from_marketed
from .wedge import PATENT_DATUM_OFFSET_M, WedgeGeometry

__all__ = [
    "GRIND_PRESETS",
    "GrindPreset",
    "ParameterProvenance",
    "ProvenanceKind",
    "get_preset",
    "preset_names",
]

_ACUSHNET_CITATION = (
    "Acushnet US10143900B2 / US10661131B2 (Harrington & Gonzalez, 2017)"
)


class ProvenanceKind(Enum):
    """Where a preset's number came from."""

    MEASURED = "measured"
    """Measured on a physical head with a stated procedure."""

    PUBLISHED = "published"
    """Published by the manufacturer in a spec sheet or press material."""

    PATENT = "patent"
    """Taken from a patent's example table or claim."""

    DERIVED = "derived"
    """Computed from another sourced number by a stated relation."""

    ESTIMATED = "estimated"
    """An engineering estimate. Not a published figure."""


@dataclass(frozen=True, slots=True)
class ParameterProvenance:
    """Where one preset parameter came from."""

    kind: ProvenanceKind
    source: str
    citation: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProvenanceKind):
            raise TypeError(f"kind must be a ProvenanceKind, got {self.kind!r}")
        if not self.source.strip():
            raise ValueError("provenance source must not be empty")
        if self.kind is not ProvenanceKind.ESTIMATED and not self.citation.strip():
            raise ValueError(
                f"a {self.kind.value} value needs a citation; only estimates "
                "may omit one"
            )


@dataclass(frozen=True)
class GrindPreset:
    """A named sole grind: a design vector plus its paper trail."""

    name: str
    description: str
    geometry: WedgeGeometry
    provenance: Mapping[str, ParameterProvenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fields = set(WedgeGeometry.field_names())
        unknown = sorted(set(self.provenance) - fields)
        if unknown:
            raise ValueError(
                f"provenance key(s) {unknown} not a WedgeGeometry field; "
                "a preset may only attribute parameters that exist"
            )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def published_fields(self) -> tuple[str, ...]:
        """Fields backed by a patent, publication, measurement or derivation."""
        return tuple(
            sorted(
                name
                for name, record in self.provenance.items()
                if record.kind is not ProvenanceKind.ESTIMATED
            )
        )

    def estimated_fields(self) -> tuple[str, ...]:
        """Every other field: estimates, whether or not they are annotated."""
        published = set(self.published_fields())
        return tuple(
            sorted(
                name for name in WedgeGeometry.field_names() if name not in published
            )
        )


def _patent_example(
    name: str,
    bounce_deg: float,
    sole_width_mm: float,
    entry_height_mm: float,
    camber_area_mm2: float,
) -> GrindPreset:
    """One of the patent's worked bounce angles on a claim-band sole.

    The camber area is per-example rather than shared: the band a convex,
    monotone sole admits climbs steeply with bounce, so one number cannot be
    constructible at 15.99, 18.42 and 20.78 degrees at once (issue #8698).
    """
    geometry = WedgeGeometry.from_millimetres(
        loft_deg=56.0,
        lie_deg=64.0,
        geometric_bounce=GeometricBounce(bounce_deg),
        sole_width_mm=sole_width_mm,
        entry_height_mm=entry_height_mm,
        leading_edge_radius_mm=7.5,
        trailing_edge_radius_mm=42.0,
        sole_camber_area_mm2=camber_area_mm2,
        centre_rocker_radius_mm=250.0,
        heel_rocker_radius_mm=95.0,
        toe_rocker_radius_mm=135.0,
        trailing_relief_fraction=0.10,
        heel_relief_fraction=0.12,
        toe_relief_fraction=0.08,
        face_progression_mm=2.0,
        blade_length_mm=78.0,
        face_height_mm=38.0,
        topline_width_mm=4.0,
        head_mass_g=302.0,
    )
    return GrindPreset(
        name=name,
        description=(
            f"Acushnet sole-geometry example at {bounce_deg:.2f} deg of "
            "geometric bounce; remaining sole dimensions are engineering "
            "estimates inside the claimed preferred ranges."
        ),
        geometry=geometry,
        provenance={
            "geometric_bounce": ParameterProvenance(
                kind=ProvenanceKind.PATENT,
                source="patent example bounce angle",
                citation=_ACUSHNET_CITATION,
            ),
            "datum_offset_m": ParameterProvenance(
                kind=ProvenanceKind.PATENT,
                source="1.2 mm measurement datum",
                citation=_ACUSHNET_CITATION,
            ),
            "sole_width_m": ParameterProvenance(
                kind=ProvenanceKind.ESTIMATED,
                source="midpoint of the most-preferred 15-22 mm band",
                note="not the patent's own example value",
            ),
            "sole_camber_area_m2": ParameterProvenance(
                kind=ProvenanceKind.ESTIMATED,
                source="inside the claimed band and constructible at this bounce",
                note=(
                    "chosen so a convex monotone sole of this width can "
                    "actually realise it; the patent bands alone do not "
                    "guarantee that (issue #8698)"
                ),
            ),
        },
    )


def _retail_wedge(
    *,
    name: str,
    description: str,
    loft_deg: float,
    marketed_bounce_deg: float,
    head_mass_g: float,
    sole_width_mm: float,
    entry_height_mm: float,
    camber_area_mm2: float,
    heel_relief_fraction: float,
    toe_relief_fraction: float,
    trailing_relief_fraction: float,
    bounce_source: str,
    bounce_citation: str,
    mass_citation: str,
) -> GrindPreset:
    """A retail wedge whose published numbers are loft, bounce and mass."""
    geometry = WedgeGeometry.from_millimetres(
        loft_deg=loft_deg,
        lie_deg=64.0,
        geometric_bounce=geometric_from_marketed(
            MarketedBounce(marketed_bounce_deg),
            sole_width_m=sole_width_mm * 1e-3,
            entry_height_m=entry_height_mm * 1e-3,
            datum_offset_m=PATENT_DATUM_OFFSET_M,
        ),
        sole_width_mm=sole_width_mm,
        entry_height_mm=entry_height_mm,
        leading_edge_radius_mm=7.5,
        trailing_edge_radius_mm=42.0,
        sole_camber_area_mm2=camber_area_mm2,
        centre_rocker_radius_mm=250.0,
        heel_rocker_radius_mm=90.0,
        toe_rocker_radius_mm=130.0,
        trailing_relief_fraction=trailing_relief_fraction,
        heel_relief_fraction=heel_relief_fraction,
        toe_relief_fraction=toe_relief_fraction,
        face_progression_mm=2.0,
        blade_length_mm=78.0,
        face_height_mm=38.0,
        topline_width_mm=4.0,
        head_mass_g=head_mass_g,
    )
    return GrindPreset(
        name=name,
        description=description,
        geometry=geometry,
        provenance={
            "loft_deg": ParameterProvenance(
                kind=ProvenanceKind.PUBLISHED,
                source="manufacturer specification",
                citation=bounce_citation,
            ),
            "lie_deg": ParameterProvenance(
                kind=ProvenanceKind.PUBLISHED,
                source="manufacturer specification (64 deg standard)",
                citation=bounce_citation,
            ),
            "head_mass_kg": ParameterProvenance(
                kind=ProvenanceKind.PUBLISHED,
                source="published head mass",
                citation=mass_citation,
            ),
            "geometric_bounce": ParameterProvenance(
                kind=ProvenanceKind.DERIVED,
                source=bounce_source,
                citation=bounce_citation,
                note=(
                    "converted from the published marketed bounce with "
                    "geometric_from_marketed() using the estimated sole "
                    "width and entry height, so it inherits their uncertainty"
                ),
            ),
            "datum_offset_m": ParameterProvenance(
                kind=ProvenanceKind.PATENT,
                source="1.2 mm measurement datum",
                citation=_ACUSHNET_CITATION,
            ),
        },
    )


def _build_registry() -> dict[str, GrindPreset]:
    presets = [
        _patent_example("acushnet_example_1", 15.99, 22.0, 3.0, 44.0),
        _patent_example("acushnet_example_2", 18.42, 22.0, 3.0, 50.0),
        _patent_example("acushnet_example_3", 20.78, 22.0, 3.0, 55.0),
        _retail_wedge(
            name="sm9_54_f",
            description=(
                "54 deg full-sole wedge: published loft, lie, 12 deg marketed "
                "bounce and 304 g head mass; sole shape estimated."
            ),
            loft_deg=54.0,
            marketed_bounce_deg=12.0,
            head_mass_g=304.0,
            sole_width_mm=21.0,
            entry_height_mm=3.0,
            camber_area_mm2=48.0,
            heel_relief_fraction=0.10,
            toe_relief_fraction=0.10,
            trailing_relief_fraction=0.10,
            bounce_source="published marketed bounce, full-sole grind",
            bounce_citation="Titleist Vokey SM9 published specifications",
            mass_citation="Titleist Vokey SM9 published head mass (54 deg, 304 g)",
        ),
        _retail_wedge(
            name="sm9_58_m",
            description=(
                "58 deg crescent-sole wedge: published loft, lie, 8 deg "
                "marketed bounce and 300 g head mass; sole shape estimated."
            ),
            loft_deg=58.0,
            marketed_bounce_deg=8.0,
            head_mass_g=300.0,
            sole_width_mm=20.0,
            entry_height_mm=3.2,
            camber_area_mm2=42.0,
            heel_relief_fraction=0.28,
            toe_relief_fraction=0.20,
            trailing_relief_fraction=0.25,
            bounce_source="published marketed bounce, crescent grind",
            bounce_citation="Titleist Vokey SM9 published specifications",
            mass_citation="Titleist Vokey SM9 published head mass (58 deg, 300 g)",
        ),
        _retail_wedge(
            name="tour_shaved_heel_lob",
            description=(
                "60 deg lob wedge with a heavily shaved heel. Magnitude "
                "anchor: a tour 11 deg lob wedge measures about 4 deg with a "
                "shaved heel and the face open - roughly 7 deg from grind "
                "plus rotation. The dimensions here are estimates built to "
                "reproduce that behaviour, not a copy of any retail head. "
                "At 5 deg marketed bounce on a 20 mm sole the camber area "
                "is almost fully determined - a convex monotone sole admits "
                "a band barely 0.3 mm^2 wide - so it is not a free "
                "parameter of this grind (issue #8698)."
            ),
            loft_deg=60.0,
            marketed_bounce_deg=5.0,
            head_mass_g=302.0,
            sole_width_mm=20.0,
            entry_height_mm=2.6,
            camber_area_mm2=29.4,
            heel_relief_fraction=0.25,
            toe_relief_fraction=0.15,
            trailing_relief_fraction=0.35,
            bounce_source="tour grind magnitude datapoint",
            bounce_citation=(
                "TaylorMade MG2 lob wedge, 11 deg standard vs about 4 deg "
                "shaved-heel with the face open"
            ),
            mass_citation="typical tour lob-wedge head mass, 290-310 g band",
        ),
    ]
    return {preset.name: preset for preset in presets}


GRIND_PRESETS: Mapping[str, GrindPreset] = MappingProxyType(_build_registry())
"""Read-only registry of named grind presets."""


def preset_names() -> tuple[str, ...]:
    """Every preset name, sorted."""
    return tuple(sorted(GRIND_PRESETS))


def get_preset(name: str) -> GrindPreset:
    """Look up a preset by name.

    Raises:
        KeyError: If no such preset exists; the message lists the ones
            that do.
    """
    try:
        return GRIND_PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown grind preset {name!r}; available: {', '.join(preset_names())}"
        ) from None
