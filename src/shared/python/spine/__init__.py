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
    PolynomialCouplingLaw,
    StateDependentCouplingLaw,
    from_joint_coupling,
    level_rotations,
    regional_fraction,
)
from src.shared.python.spine.golf_terms import (
    GolfTermDefinition,
    contribution_remainder,
    registered_terms,
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
from src.shared.python.spine.segmental import (
    LevelTransform,
    SegmentalKinematicsSeries,
    SegmentalPose,
)

__all__ = [
    "REQUIRED_PROVENANCE_KEYS",
    "ContextBand",
    "ContextComparison",
    "CouplingLaw",
    "GolfTermDefinition",
    "LevelTransform",
    "LinearCouplingLaw",
    "LoadObservable",
    "PolynomialCouplingLaw",
    "SegmentalKinematicsSeries",
    "SegmentalPose",
    "SpinalAxialLoadProvider",
    "SpinalLevel",
    "SpinalLoadSeries",
    "SpinalRegion",
    "StateDependentCouplingLaw",
    "annotate",
    "context_bands",
    "contribution_remainder",
    "from_joint_coupling",
    "from_wrench_in_frame",
    "level_rotations",
    "levels_between",
    "load_context_bands",
    "parse_level",
    "regional_fraction",
    "registered_terms",
    "segmental_contributions",
]
