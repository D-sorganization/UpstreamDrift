"""Sand state model for BunkerShot3D (issue #8610, epic #8607).

The package answers three questions the previous code could not:

1. **What is the sand?** :class:`~bunkershot3d.sand.state.SandState` carries a
   sieve analysis, a compaction state, a moisture state and a bed geometry as
   one frozen value object, with USGA-spec presets built from published
   tables.
2. **How wet is it, and which physics does that imply?** Moisture is two
   regimes, not a scalar: damp/capillary and saturated/cavitating. Any suction
   term is hard-clamped at the cavitation limit, about -100 kPa gauge.
3. **Can the requested simulation exist?** Given a bed depth, a gradation and
   a packing fraction, :mod:`~bunkershot3d.sand.feasibility` reports the grain
   count actually needed and refuses configurations that cannot fill the bed
   (defect B29: 50,000 grains of d = 0.4 mm in a 0.4 x 0.3 x 0.1 m domain is a
   solid fraction of 1.4e-4, a settled bed 0.023 mm deep).

Every preset carries a provenance record naming the basis of each
honesty-critical value. The friction angle, packing fraction and grain density
are borrowed from a Quikrete medium-sand analogue, because no published values
specific to golf bunker sand were found; see
:mod:`bunkershot3d.sand.provenance`.
"""

from __future__ import annotations

from .bed import (
    MAX_STABLE_SLOPE_RAD,
    USGA_FACE_DEPTH_RANGE_M,
    USGA_FLOOR_DEPTH_RANGE_M,
    BedZone,
    BunkerBedGeometry,
)
from .exceptions import (
    BedGeometryError,
    InfeasibleBedError,
    MoistureRegimeError,
    PackingStateError,
    ParticleSizeDistributionError,
    ProvenanceError,
    SandModelError,
)
from .feasibility import (
    MAX_PHYSICAL_SOLID_FRACTION,
    BedFeasibilityReport,
    achieved_solid_fraction,
    evaluate_bed_feasibility,
    grain_volume_m3,
    require_feasible_bed,
    required_grain_count,
    settled_bed_depth_m,
)
from .firmness import (
    FIRMNESS_SWEEP_KG_PER_CM2,
    FirmnessRating,
    firmness_kg_per_cm2_from_pa,
    firmness_pa_from_kg_per_cm2,
    firmness_rating,
    relative_density_from_firmness,
)
from .moisture import (
    CAVITATION_PORE_PRESSURE_PA,
    CAVITATION_SUCTION_LIMIT_PA,
    MoistureRegime,
    MoistureState,
    capillary_apparent_cohesion_pa,
    capillary_suction_pa,
    cavitation_limited_strength_gain_pa,
    clamp_pore_pressure_pa,
    clamp_suction_pa,
    classify_regime,
    degree_of_saturation,
)
from .packing import (
    SAND_VOID_RATIO_MAX,
    SAND_VOID_RATIO_MIN,
    Angularity,
    PackingState,
    solid_fraction_from_void_ratio,
    void_ratio_from_solid_fraction,
)
from .presets import (
    USGA_GSR_2020_WINDY_PSD,
    USGA_LAB_MIDBAND_PSD,
    PlayingCondition,
    all_presets,
    firmness_sweep,
    playing_condition,
    usga_reference_sand,
)
from .provenance import (
    REQUIRED_PROVENANCE_KEYS,
    PropertyProvenance,
    ProvenanceBasis,
    SandProvenance,
)
from .psd import ParticleSizeDistribution
from .specification import (
    USGA_GSR_2020_SPECIFICATION,
    USGA_LAB_SPECIFICATION,
    ComplianceReport,
    SieveBand,
    Specification,
    evaluate_compliance,
)
from .state import SandState

__all__ = [
    "CAVITATION_PORE_PRESSURE_PA",
    "CAVITATION_SUCTION_LIMIT_PA",
    "FIRMNESS_SWEEP_KG_PER_CM2",
    "MAX_PHYSICAL_SOLID_FRACTION",
    "MAX_STABLE_SLOPE_RAD",
    "REQUIRED_PROVENANCE_KEYS",
    "SAND_VOID_RATIO_MAX",
    "SAND_VOID_RATIO_MIN",
    "USGA_FACE_DEPTH_RANGE_M",
    "USGA_FLOOR_DEPTH_RANGE_M",
    "USGA_GSR_2020_SPECIFICATION",
    "USGA_GSR_2020_WINDY_PSD",
    "USGA_LAB_MIDBAND_PSD",
    "USGA_LAB_SPECIFICATION",
    "Angularity",
    "BedFeasibilityReport",
    "BedGeometryError",
    "BedZone",
    "BunkerBedGeometry",
    "ComplianceReport",
    "FirmnessRating",
    "InfeasibleBedError",
    "MoistureRegime",
    "MoistureRegimeError",
    "MoistureState",
    "PackingState",
    "PackingStateError",
    "ParticleSizeDistribution",
    "ParticleSizeDistributionError",
    "PlayingCondition",
    "PropertyProvenance",
    "ProvenanceBasis",
    "ProvenanceError",
    "SandModelError",
    "SandProvenance",
    "SandState",
    "SieveBand",
    "Specification",
    "achieved_solid_fraction",
    "all_presets",
    "capillary_apparent_cohesion_pa",
    "capillary_suction_pa",
    "cavitation_limited_strength_gain_pa",
    "clamp_pore_pressure_pa",
    "clamp_suction_pa",
    "classify_regime",
    "degree_of_saturation",
    "evaluate_bed_feasibility",
    "evaluate_compliance",
    "firmness_kg_per_cm2_from_pa",
    "firmness_pa_from_kg_per_cm2",
    "firmness_rating",
    "firmness_sweep",
    "grain_volume_m3",
    "playing_condition",
    "relative_density_from_firmness",
    "require_feasible_bed",
    "required_grain_count",
    "settled_bed_depth_m",
    "solid_fraction_from_void_ratio",
    "usga_reference_sand",
    "void_ratio_from_solid_fraction",
]
