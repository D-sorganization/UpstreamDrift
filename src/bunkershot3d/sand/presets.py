"""USGA bunker-sand presets and playing conditions (issue #8610).

Two gradations, both taken from published sieve tables:

``USGA_LAB_MIDBAND_PSD``
    Mid-band of the Turf & Soil Diagnostics bunker-sand table: 81.5 % coarse
    plus medium, 1.5 % silt and clay, Cu = 2.7.
``USGA_GSR_2020_WINDY_PSD``
    The windy-site gradation of USGA Green Section Record 58(11) 2020: > 80 %
    between 0.25 and 1 mm with 13 % in the 1-2 mm fraction to resist erosion.
    It deliberately breaches that same record's <= 7 % very-coarse cap, which
    the compliance report surfaces rather than hides.

Four playing conditions are built on the mid-band gradation and swept by
penetrometer firmness.

**Honesty.** The friction angle, packing fraction and grain density are
borrowed from Quikrete medium sand (0.3-0.8 mm) as an analogue; no published
values specific to golf bunker sand were found. Every preset records that in
its provenance, and the tests fail if one stops doing so. See
:mod:`bunkershot3d.sand.provenance` and issue #7999.
"""

from __future__ import annotations

from enum import StrEnum

from .bed import BedZone, BunkerBedGeometry
from .exceptions import SandModelError
from .firmness import (
    FIRMNESS_SWEEP_KG_PER_CM2,
    firmness_pa_from_kg_per_cm2,
    relative_density_from_firmness,
)
from .moisture import MoistureState
from .packing import Angularity, PackingState
from .provenance import (
    QUIKRETE_FRICTION_ANGLE_DEG,
    QUIKRETE_PARTICLE_DENSITY_KG_M3,
    PropertyProvenance,
    ProvenanceBasis,
    SandProvenance,
    borrowed_from_quikrete,
    published_specification,
)
from .psd import ParticleSizeDistribution
from .state import SandState

__all__ = [
    "DEFAULT_BED",
    "MENISCUS_RADIUS_TO_D10",
    "USGA_GSR_2020_WINDY_PSD",
    "USGA_LAB_MIDBAND_PSD",
    "PlayingCondition",
    "all_presets",
    "firmness_sweep",
    "playing_condition",
    "usga_reference_sand",
]

# USDA size-class boundaries used by both published tables, in metres.
_USDA_EDGES_M = (2e-6, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)

USGA_LAB_MIDBAND_PSD = ParticleSizeDistribution.from_bins(
    bin_edges_m=_USDA_EDGES_M,
    # silt+clay, very fine, fine, medium, coarse, very coarse, gravel
    bin_fractions=(0.015, 0.025, 0.080, 0.435, 0.380, 0.055, 0.010),
    name="usga-lab-midband",
)

USGA_GSR_2020_WINDY_PSD = ParticleSizeDistribution.from_bins(
    bin_edges_m=_USDA_EDGES_M,
    bin_fractions=(0.010, 0.010, 0.030, 0.350, 0.460, 0.130, 0.010),
    name="usga-gsr-2020-windy",
)

DEFAULT_BED = BunkerBedGeometry(
    depth_m=0.125,
    plan_length_m=0.40,
    plan_width_m=0.30,
    zone=BedZone.FLOOR,
)
"""Mid-band USGA bunker floor: 125 mm of sand over a 0.4 x 0.3 m patch."""

MENISCUS_RADIUS_TO_D10 = 0.06
"""Meniscus neck radius as a fraction of d10.

Chosen so the resulting apparent cohesion lands inside the 1-10 kPa band
reported for damp sand. It reproduces a published *band*; it is not a measured
pore geometry, and the presets record it as an estimate.
"""


class PlayingCondition(StrEnum):
    """The four bunker conditions a wedge has to cope with."""

    FIRM = "firm"
    FLUFFY = "fluffy"
    WET = "wet"
    PLUGGED = "plugged"


# firmness (kg/cm^2), gravimetric water content, grain shape
_CONDITIONS: dict[PlayingCondition, tuple[float, float, Angularity]] = {
    PlayingCondition.FIRM: (2.8, 0.050, Angularity.ANGULAR),
    PlayingCondition.FLUFFY: (1.8, 0.002, Angularity.SUBANGULAR),
    PlayingCondition.WET: (2.0, 0.260, Angularity.ANGULAR),
    PlayingCondition.PLUGGED: (1.6, 0.005, Angularity.SUBROUNDED),
}

_MOISTURE_PROVENANCE = PropertyProvenance(
    basis=ProvenanceBasis.ESTIMATED,
    source=(
        "regime thresholds from pendular/funicular/capillary saturation "
        "conventions; apparent cohesion band 1-10 kPa per the BunkerShot3D "
        "research digest section 3"
    ),
    note=(
        "Water contents are representative of the named playing condition, "
        "not measurements on a specific bunker. The meniscus radius is tuned "
        "to reproduce the published cohesion band."
    ),
)

_FIRMNESS_PROVENANCE = PropertyProvenance(
    basis=ProvenanceBasis.CONVENTION,
    source="USGA / Turf & Soil Diagnostics penetrometer rating scale",
    note=(
        "The rating bands are published. The linear map from firmness to "
        "relative density is a modelling convention chosen so the 1.6-2.8 "
        "kg/cm^2 sweep spans loose to dense; no published correlation exists."
    ),
)


def _provenance(gradation_note: str) -> SandProvenance:
    return SandProvenance(
        entries={
            "particle_size_distribution": published_specification(gradation_note),
            "particle_density_kg_m3": borrowed_from_quikrete(
                "Grain density 2600 kg/m^3."
            ),
            "packing": borrowed_from_quikrete(
                "Packing fraction 0.60; the e_min/e_max limits are the "
                "random-close and random-loose packings of equal spheres."
            ),
            "friction_angle_deg": borrowed_from_quikrete(
                f"Internal friction angle {QUIKRETE_FRICTION_ANGLE_DEG} deg."
            ),
            "moisture": _MOISTURE_PROVENANCE,
            "penetrometer_firmness": _FIRMNESS_PROVENANCE,
        }
    )


def usga_reference_sand(
    name: str,
    firmness_kg_per_cm2: float,
    gravimetric_water_content: float,
    angularity: Angularity = Angularity.ANGULAR,
    psd: ParticleSizeDistribution = USGA_LAB_MIDBAND_PSD,
    bed: BunkerBedGeometry = DEFAULT_BED,
    gradation_note: str = "USGA / Turf & Soil Diagnostics bunker-sand table",
) -> SandState:
    """Build a sand state from the published USGA reference values.

    Args:
        name: Label for the state.
        firmness_kg_per_cm2: Penetrometer reading, published unit.
        gravimetric_water_content: Mass of water per mass of dry solids.
        angularity: Grain shape class.
        psd: Sieve analysis; defaults to the mid-band USGA gradation.
        bed: Bed geometry; defaults to a 125 mm USGA floor.
        gradation_note: Provenance note for the gradation.

    Returns:
        A fully populated, provenance-carrying :class:`SandState`.
    """
    packing = PackingState.from_relative_density(
        particle_density_kg_m3=QUIKRETE_PARTICLE_DENSITY_KG_M3,
        relative_density=relative_density_from_firmness(firmness_kg_per_cm2),
    )
    moisture = MoistureState.from_water_content(
        gravimetric_water_content=gravimetric_water_content,
        void_ratio=packing.void_ratio,
        particle_density_kg_m3=packing.particle_density_kg_m3,
        meniscus_radius_m=MENISCUS_RADIUS_TO_D10 * psd.d10_m,
    )
    return SandState(
        name=name,
        psd=psd,
        packing=packing,
        moisture=moisture,
        bed=bed,
        angularity=angularity,
        friction_angle_deg=QUIKRETE_FRICTION_ANGLE_DEG,
        penetrometer_firmness_pa=firmness_pa_from_kg_per_cm2(firmness_kg_per_cm2),
        provenance=_provenance(gradation_note),
    )


def playing_condition(
    condition: PlayingCondition,
    psd: ParticleSizeDistribution = USGA_LAB_MIDBAND_PSD,
    bed: BunkerBedGeometry = DEFAULT_BED,
) -> SandState:
    """Return the sand state for a named playing condition.

    Args:
        condition: One of firm, fluffy, wet or plugged.
        psd: Gradation to use; defaults to the mid-band USGA table.
        bed: Bed geometry; defaults to a 125 mm USGA floor.

    Returns:
        The corresponding :class:`SandState`.

    Raises:
        SandModelError: if ``condition`` is not a known playing condition.
    """
    try:
        key = PlayingCondition(condition)
    except ValueError:
        raise SandModelError(
            f"unknown playing condition {condition!r}; expected one of "
            + ", ".join(sorted(c.value for c in PlayingCondition))
        ) from None
    firmness, water_content, angularity = _CONDITIONS[key]
    return usga_reference_sand(
        name=f"usga-{key.value}",
        firmness_kg_per_cm2=firmness,
        gravimetric_water_content=water_content,
        angularity=angularity,
        psd=psd,
        bed=bed,
    )


def firmness_sweep(
    gravimetric_water_content: float = 0.050,
    psd: ParticleSizeDistribution = USGA_LAB_MIDBAND_PSD,
    bed: BunkerBedGeometry = DEFAULT_BED,
) -> tuple[SandState, ...]:
    """Return one sand state per published penetrometer sweep point.

    The sweep isolates firmness: gradation, moisture and grain shape are held
    fixed while the compaction state moves from loose to dense across
    1.6 / 2.0 / 2.4 / 2.8 kg/cm^2.
    """
    return tuple(
        usga_reference_sand(
            name=f"usga-sweep-{value:g}",
            firmness_kg_per_cm2=value,
            gravimetric_water_content=gravimetric_water_content,
            psd=psd,
            bed=bed,
        )
        for value in FIRMNESS_SWEEP_KG_PER_CM2
    )


def all_presets() -> dict[str, SandState]:
    """Return every named preset, keyed by name."""
    presets = {
        state.name: state for state in (playing_condition(c) for c in PlayingCondition)
    }
    windy = usga_reference_sand(
        name="usga-windy-firm",
        firmness_kg_per_cm2=2.8,
        gravimetric_water_content=0.050,
        psd=USGA_GSR_2020_WINDY_PSD,
        gradation_note=(
            "USGA Green Section Record 58(11) 2020 windy-site gradation; "
            "deliberately above that record's very-coarse cap"
        ),
    )
    presets[windy.name] = windy
    return presets
