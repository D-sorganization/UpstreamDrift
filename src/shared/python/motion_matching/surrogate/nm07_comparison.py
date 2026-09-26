"""Discovery pointer: NM-07 forward surrogate comparison lives in ``neural_motion``.

Issue #10622 compares forward surrogates and physics-structured alternatives.
This module re-exports the public comparison surface so surrogate callers
and path-membership checks resolve honestly without duplicating contracts (DRY),
matching the NM-00 ``nm00_audit`` and NM-01 ``nm01_freeze`` pattern.
"""

from __future__ import annotations

from src.shared.python.neural_motion.surrogates import (
    DEFAULT_EVALUATORS,
    SURROGATE_COMPARISON_SCHEMA,
    PhysicsStructuredSurrogate,
    SurrogateAblationResult,
    SurrogateCandidateKind,
    SurrogateComparisonConfig,
    SurrogateComparisonReport,
    compare_surrogates_and_alternatives,
    surrogate_passes_gates,
    surrogate_selection_key,
)

__all__ = [
    "DEFAULT_EVALUATORS",
    "SURROGATE_COMPARISON_SCHEMA",
    "PhysicsStructuredSurrogate",
    "SurrogateAblationResult",
    "SurrogateCandidateKind",
    "SurrogateComparisonConfig",
    "SurrogateComparisonReport",
    "compare_surrogates_and_alternatives",
    "surrogate_passes_gates",
    "surrogate_selection_key",
]
