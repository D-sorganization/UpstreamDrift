"""Tests for Modular Impact Model.

Guideline K3 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    ImpactSolverAPI,
)

# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================


class TestCORValidation:
    """Tests for COR validation accuracy (Issue #758)."""

    @pytest.mark.parametrize("cor", [0.6, 0.7, 0.78, 0.85])
    def test_cor_matches_parameter(self, cor: float) -> None:
        """Measured COR should approximately match parameter."""
        solver = ImpactSolverAPI(params=ImpactParameters(cor=cor))

        for _ in range(3):
            solver.solve_impact(
                timestamp=0.0,
                clubhead_velocity=np.array([40.0, 0.0, 0.0]),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            )

        result = solver.validate_cor_behavior(tolerance=0.15)

        # Measured COR should be within tolerance of expected
        assert result["deviation"] < 0.15
