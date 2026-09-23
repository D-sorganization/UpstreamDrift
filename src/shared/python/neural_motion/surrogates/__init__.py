"""NM-07 forward surrogates and physics-structured alternatives (issue #10622)."""

from .comparison import compare_surrogates_and_alternatives
from .physics_structured import PhysicsStructuredSurrogate
from .types import (
    SURROGATE_COMPARISON_SCHEMA,
    SurrogateAblationResult,
    SurrogateCandidateKind,
    SurrogateComparisonConfig,
    SurrogateComparisonReport,
)

__all__ = [
    "SURROGATE_COMPARISON_SCHEMA",
    "PhysicsStructuredSurrogate",
    "SurrogateAblationResult",
    "SurrogateCandidateKind",
    "SurrogateComparisonConfig",
    "SurrogateComparisonReport",
    "compare_surrogates_and_alternatives",
]
