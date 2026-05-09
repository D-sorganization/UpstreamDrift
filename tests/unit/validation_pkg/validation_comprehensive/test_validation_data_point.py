"""Comprehensive unit tests for the validation_pkg package.

Tests cover:
- validation.py: PhysicalValidationError, validate_mass, validate_timestep,
  validate_inertia_matrix, validate_joint_limits, validate_friction_coefficient,
  validate_physical_bounds decorator
- validation_utils.py: validate_array_shape, validate_array_dimensions,
  validate_array_length, validate_positive, validate_range, validate_file_exists,
  validate_directory_exists, validate_extension, validate_not_none,
  validate_type, validate_dict_keys, validate_numeric, validate_all
- validation_helpers.py: ValidationLevel, PhysicsValidationError, validate_finite,
  validate_magnitude, validate_joint_state, validate_cartesian_state,
  validate_model_parameters, physical constants
- validation_data.py: DataSource, ValidationDataPoint, PGA_TOUR_2024,
  AMATEUR_AVERAGES, ALL_VALIDATION_DATA, get_validation_data_for_club
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

# --- validation.py ---
from src.shared.python.validation_pkg.validation import (
    PhysicalValidationError,
    validate_friction_coefficient,
    validate_inertia_matrix,
    validate_joint_limits,
    validate_mass,
    validate_physical_bounds,
    validate_timestep,
)

# --- validation_data.py ---
from src.shared.python.validation_pkg.validation_data import (
    ALL_VALIDATION_DATA,
    AMATEUR_AVERAGES,
    PGA_TOUR_2024,
    DataSource,
    ValidationDataPoint,
    get_validation_data_for_club,
)

# --- validation_helpers.py ---
from src.shared.python.validation_pkg.validation_helpers import (
    MAX_CARTESIAN_ACCELERATION_M_S2,
    MAX_CARTESIAN_VELOCITY_M_S,
    MAX_JOINT_ACCELERATION_RAD_S2,
    MAX_JOINT_POSITION_RAD,
    MAX_JOINT_VELOCITY_RAD_S,
    PhysicsValidationError,
    ValidationLevel,
    validate_cartesian_state,
    validate_finite,
    validate_joint_state,
    validate_magnitude,
    validate_model_parameters,
)

# --- validation_utils.py ---
from src.shared.python.validation_pkg.validation_utils import (
    validate_all,
    validate_array_dimensions,
    validate_array_length,
    validate_array_shape,
    validate_dict_keys,
    validate_directory_exists,
    validate_extension,
    validate_file_exists,
    validate_not_none,
    validate_numeric,
    validate_positive,
    validate_range,
    validate_type,
)

# ============================================================================
# validation.py -- PhysicalValidationError
# ============================================================================


# ============================================================================
# validation.py -- validate_mass
# ============================================================================


# ============================================================================
# validation.py -- validate_timestep
# ============================================================================


# ============================================================================
# validation.py -- validate_inertia_matrix
# ============================================================================


# ============================================================================
# validation.py -- validate_joint_limits
# ============================================================================


# ============================================================================
# validation.py -- validate_friction_coefficient
# ============================================================================


# ============================================================================
# validation.py -- validate_physical_bounds decorator
# ============================================================================


# ============================================================================
# validation_utils.py -- Array Validators
# ============================================================================


# ============================================================================
# validation_utils.py -- Scalar Validators
# ============================================================================


# ============================================================================
# validation_utils.py -- Path/File Validators
# ============================================================================


# ============================================================================
# validation_utils.py -- Type/Dict Validators
# ============================================================================


# ============================================================================
# validation_utils.py -- validate_all
# ============================================================================


# ============================================================================
# validation_helpers.py -- Constants
# ============================================================================


# ============================================================================
# validation_helpers.py -- ValidationLevel
# ============================================================================


# ============================================================================
# validation_helpers.py -- validate_finite
# ============================================================================


# ============================================================================
# validation_helpers.py -- validate_magnitude
# ============================================================================


# ============================================================================
# validation_helpers.py -- validate_joint_state
# ============================================================================


# ============================================================================
# validation_helpers.py -- validate_cartesian_state
# ============================================================================


# ============================================================================
# validation_helpers.py -- validate_model_parameters
# ============================================================================


# ============================================================================
# validation_data.py -- DataSource Enum
# ============================================================================


# ============================================================================
# validation_data.py -- ValidationDataPoint
# ============================================================================


class TestValidationDataPoint:
    """Tests for ValidationDataPoint dataclass."""

    @pytest.fixture()
    def driver_data(self) -> ValidationDataPoint:
        return PGA_TOUR_2024[0]  # Driver

    def test_is_frozen(self, driver_data: ValidationDataPoint) -> None:
        with pytest.raises(AttributeError):
            driver_data.club = "modified"  # type: ignore[misc]

    def test_ball_speed_mph_conversion(self, driver_data: ValidationDataPoint) -> None:
        mph = driver_data.ball_speed_mph
        assert mph > 0
        # PGA Tour driver ~ 174 mph
        assert 100 < mph < 200

    def test_carry_distance_yards(self, driver_data: ValidationDataPoint) -> None:
        yards = driver_data.carry_distance_yards
        assert yards > 0
        # PGA Tour driver ~ 282 yards
        assert 200 < yards < 350

    @pytest.mark.parametrize(
        "multiplier, expected_valid",
        [
            (1.0, True),
            (2.0, False),
        ],
        ids=["within-tolerance", "outside-tolerance"],
    )
    def test_is_valid_carry(
        self,
        driver_data: ValidationDataPoint,
        multiplier: float,
        expected_valid: bool,
    ) -> None:
        assert (
            driver_data.is_valid_carry(driver_data.carry_distance_m * multiplier)
            == expected_valid
        )

    def test_is_valid_carry_at_boundary(self, driver_data: ValidationDataPoint) -> None:
        tol = driver_data.carry_tolerance_pct / 100
        edge = driver_data.carry_distance_m * (1 + tol)
        assert driver_data.is_valid_carry(edge)


# ============================================================================
# validation_data.py -- Reference Data Collections
# ============================================================================


# ============================================================================
# validation_data.py -- get_validation_data_for_club
# ============================================================================
