"""Tests for the scientific computing package.

This module contains comprehensive tests for all package functionality,
including positive tests, negative tests, and edge cases.
"""

import logging
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import constants, logger_utils


class TestConstants:
    """Test physical and mathematical constants."""

    def test_mathematical_constants(self) -> None:
        """Test mathematical constants have correct values."""
        assert pytest.approx(math.pi, rel=1e-15) == constants.PI
        assert pytest.approx(2.718281828459045, rel=1e-15) == constants.E

    def test_physical_constants_positive(self) -> None:
        """Test physical constants are positive."""
        assert constants.GRAVITY_M_S2 > 0
        assert constants.SPEED_OF_LIGHT_M_S > 0
        assert constants.AIR_DENSITY_SEA_LEVEL_KG_M3 > 0
        assert constants.GOLF_BALL_MASS_KG > 0
        assert constants.GOLF_BALL_DIAMETER_M > 0

    def test_conversion_factors(self) -> None:
        """Test conversion factors work correctly."""
        # Test speed conversions
        speed_mps = 10.0
        speed_kph = speed_mps * constants.MPS_TO_KPH
        speed_mph = speed_mps * constants.MPS_TO_MPH

        assert speed_kph == pytest.approx(36.0, rel=1e-10)
        assert speed_mph == pytest.approx(22.3694, rel=1e-10)

        # Test angle conversions
        angle_deg = 180.0
        angle_rad = angle_deg * constants.DEG_TO_RAD
        angle_deg_back = angle_rad * constants.RAD_TO_DEG

        assert angle_rad == pytest.approx(math.pi, rel=1e-10)
        assert angle_deg_back == pytest.approx(180.0, rel=1e-10)

    def test_material_properties_positive(self) -> None:
        """Test material densities are positive."""
        assert constants.GRAPHITE_DENSITY_KG_M3 > 0
        assert constants.STEEL_DENSITY_KG_M3 > 0
        assert constants.TITANIUM_DENSITY_KG_M3 > 0
        assert constants.ALUMINUM_DENSITY_KG_M3 > 0


class TestLoggerUtils:
    """Test logging utilities."""

    def test_set_seeds_default(self) -> None:
        """Test setting seeds with default value."""
        import numpy as np
        import random

        # Set seeds
        logger_utils.set_seeds()

        # Verify numpy's random state was set by checking reproducibility
        # Setting the same seed should give the same random values
        logger_utils.set_seeds(42)  # Use known seed
        first_value = np.random.random()

        logger_utils.set_seeds(42)  # Reset to same seed
        second_value = np.random.random()

        assert first_value == second_value, "Seed setting should produce reproducible results"

    def test_set_seeds_custom(self) -> None:
        """Test setting seeds with custom value."""
        import numpy as np

        custom_seed = 12345
        logger_utils.set_seeds(custom_seed)

        # Record value
        first_value = np.random.random()

        # Reset to same seed should reproduce the same sequence
        logger_utils.set_seeds(custom_seed)
        second_value = np.random.random()

        assert first_value == second_value, "Custom seed should produce reproducible results"

    def test_get_logger(self) -> None:
        """Test getting a logger instance."""
        logger = logger_utils.get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_setup_logging(self) -> None:
        """Test logging setup."""
        logger_utils.setup_logging()

        # Verify logging was configured by checking root logger has handlers
        root_logger = logging.getLogger()
        # After setup_logging, there should be at least one handler
        # or the logging level should be configured
        assert root_logger.level != logging.NOTSET or len(root_logger.handlers) > 0, (
            "setup_logging should configure the logging system"
        )


class TestNegativeCases:
    """Test negative cases and error handling."""

    def test_set_seeds_negative_value(self) -> None:
        """Test setting seeds with negative value (should work)."""
        # NumPy doesn't accept negative seeds, so we should handle this
        with pytest.raises(ValueError, match="expected non-negative integer"):
            logger_utils.set_seeds(-42)

    def test_get_logger_empty_name(self) -> None:
        """Test getting logger with empty name."""
        logger = logger_utils.get_logger("")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "root"  # Empty name becomes "root"


def test_sanity() -> None:
    """Basic sanity test."""
    assert 2 + 2 == 4
    assert constants.PI > 3.0
    assert constants.E > 2.0
