"""Tests for muscle synergy analysis module.

Tests the Non-negative Matrix Factorization (NMF) based muscle synergy
extraction and analysis functionality.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.biomechanics.muscle_analysis import (
    SKLEARN_AVAILABLE,
    MuscleSynergyAnalyzer,
    SynergyResult,
)
from src.shared.python.core.contracts import PreconditionError
from src.shared.python.engine_core.engine_availability import skip_if_unavailable


@pytest.mark.skipif(SKLEARN_AVAILABLE, reason="Test for sklearn not available")
class TestSklearnNotAvailable:
    """Test behavior when sklearn is not installed."""

    def test_extract_synergies_raises_import_error(self) -> None:
        """Test that extract_synergies raises ImportError without sklearn."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(ImportError, match="sklearn is required"):
            analyzer.extract_synergies(n_synergies=2)

    def test_find_optimal_synergies_raises_import_error(self) -> None:
        """Test that find_optimal_synergies raises ImportError without sklearn."""
        data = np.random.rand(100, 5)
        analyzer = MuscleSynergyAnalyzer(data)

        with pytest.raises(ImportError, match="sklearn is required"):
            analyzer.find_optimal_synergies(max_synergies=5, vaf_threshold=0.90)
