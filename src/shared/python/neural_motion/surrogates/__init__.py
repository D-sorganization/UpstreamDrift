"""NM-07 forward surrogates and physics-structured alternatives (issue #10622)."""

from .comparison import (
    DEFAULT_EVALUATORS,
    compare_surrogates_and_alternatives,
    surrogate_passes_gates,
    surrogate_selection_key,
)
from .physics_structured import PhysicsStructuredSurrogate
from .sparse_residual import (
    CandidateLibrary,
    SparseResidualFit,
    fit_sparse_residual,
)
from .types import (
    SURROGATE_COMPARISON_SCHEMA,
    SurrogateAblationResult,
    SurrogateCandidateKind,
    SurrogateComparisonConfig,
    SurrogateComparisonReport,
)

__all__ = [
    "CandidateLibrary",
    "DEFAULT_EVALUATORS",
    "SURROGATE_COMPARISON_SCHEMA",
    "PhysicsStructuredSurrogate",
    "SparseResidualFit",
    "SurrogateAblationResult",
    "SurrogateCandidateKind",
    "SurrogateComparisonConfig",
    "SurrogateComparisonReport",
    "compare_surrogates_and_alternatives",
    "fit_sparse_residual",
    "surrogate_passes_gates",
    "surrogate_selection_key",
]
