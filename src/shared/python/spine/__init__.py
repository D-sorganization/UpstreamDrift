"""Engine-independent segmental spine contracts (epic #12735, ADR-0055).

Public API re-exports only; see the submodules for definitions. The golf-term
``definition`` and ``evaluate`` functions stay in ``spine.golf_terms`` because
their names are too generic for the package namespace.
"""

from src.shared.python.spine.context_bands import (
    ContextBand,
    ContextComparison,
    annotate,
    context_bands,
    load_context_bands,
)
from src.shared.python.spine.coupling import (
    CouplingLaw,
    LinearCouplingLaw,
    ParityCouplingLaw,
    PolynomialCouplingLaw,
    StateDependentCouplingLaw,
    from_joint_coupling,
    level_rotations,
    regional_fraction,
    restrict_to_levels,
)
from src.shared.python.spine.coupling_data import (
    available_laws,
    coupling_law,
    load_coupling_laws,
)
from src.shared.python.spine.distribution import (
    ChainOffsets,
    DistributionResult,
    DistributionSeriesResult,
    ObservabilityReport,
    distribute,
    distribute_series,
    distribute_series_detailed,
)
from src.shared.python.spine.golf_terms import (
    GolfTermContributions,
    GolfTermDefinition,
    contribution_remainder,
    registered_terms,
    segmental_contribution_breakdown,
    segmental_contributions,
)
from src.shared.python.spine.levels import (
    SpinalLevel,
    SpinalRegion,
    levels_between,
    parse_level,
)
from src.shared.python.spine.loads import (
    REQUIRED_PROVENANCE_KEYS,
    LoadObservable,
    SpinalAxialLoadProvider,
    SpinalLoadSeries,
    from_wrench_in_frame,
)
from src.shared.python.spine.movement_variants import (
    MovementVariant,
    VariantResult,
    apply_variant,
    load_movement_variants,
    movement_variant,
)
from src.shared.python.spine.segmental import (
    LevelTransform,
    SegmentalKinematicsSeries,
    SegmentalPose,
)
from src.shared.python.spine.viz_data import (
    HeatmapData,
    StackData,
    level_angle_stack,
    load_heatmap,
    to_tension_positive,
)

__all__ = [
    "REQUIRED_PROVENANCE_KEYS",
    "ChainOffsets",
    "ContextBand",
    "ContextComparison",
    "CouplingLaw",
    "DistributionResult",
    "DistributionSeriesResult",
    "GolfTermContributions",
    "GolfTermDefinition",
    "HeatmapData",
    "LevelTransform",
    "LinearCouplingLaw",
    "ParityCouplingLaw",
    "LoadObservable",
    "MovementVariant",
    "ObservabilityReport",
    "PolynomialCouplingLaw",
    "SegmentalKinematicsSeries",
    "SegmentalPose",
    "SpinalAxialLoadProvider",
    "SpinalLevel",
    "SpinalLoadSeries",
    "SpinalRegion",
    "StackData",
    "StateDependentCouplingLaw",
    "VariantResult",
    "annotate",
    "apply_variant",
    "available_laws",
    "context_bands",
    "contribution_remainder",
    "coupling_law",
    "distribute",
    "distribute_series",
    "distribute_series_detailed",
    "from_joint_coupling",
    "from_wrench_in_frame",
    "level_angle_stack",
    "level_rotations",
    "levels_between",
    "movement_variant",
    "load_context_bands",
    "load_coupling_laws",
    "load_heatmap",
    "load_movement_variants",
    "parse_level",
    "regional_fraction",
    "registered_terms",
    "restrict_to_levels",
    "segmental_contribution_breakdown",
    "segmental_contributions",
    "to_tension_positive",
]
