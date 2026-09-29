"""NM-07 (#10622): surrogate discovery pointer re-exports neural_motion surrogates."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_surrogate_nm07_comparison_reexports_neural_motion_surface() -> None:
    from src.shared.python.motion_matching.surrogate import nm07_comparison
    from src.shared.python.neural_motion import surrogates

    assert (
        nm07_comparison.SURROGATE_COMPARISON_SCHEMA
        == surrogates.SURROGATE_COMPARISON_SCHEMA
    )
    assert (
        nm07_comparison.compare_surrogates_and_alternatives
        is surrogates.compare_surrogates_and_alternatives
    )
    assert (
        nm07_comparison.PhysicsStructuredSurrogate
        is surrogates.PhysicsStructuredSurrogate
    )
    assert nm07_comparison.DEFAULT_EVALUATORS is surrogates.DEFAULT_EVALUATORS
    assert nm07_comparison.surrogate_passes_gates is surrogates.surrogate_passes_gates
    assert nm07_comparison.surrogate_selection_key is surrogates.surrogate_selection_key
