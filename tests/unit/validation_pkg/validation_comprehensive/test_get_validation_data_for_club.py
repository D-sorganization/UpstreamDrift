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


# ============================================================================
# validation_data.py -- Reference Data Collections
# ============================================================================


# ============================================================================
# validation_data.py -- get_validation_data_for_club
# ============================================================================


class TestGetValidationDataForClub:
    """Tests for get_validation_data_for_club function."""

    @pytest.mark.parametrize(
        "club_query, min_results",
        [
            ("Driver", 1),
            ("Iron", 1),
        ],
        ids=["driver", "iron"],
    )
    def test_club_match(self, club_query: str, min_results: int) -> None:
        results = get_validation_data_for_club(club_query)
        assert len(results) >= min_results

    def test_validation_comprehensive_case_insensitive(self) -> None:
        results_lower = get_validation_data_for_club("driver")
        results_upper = get_validation_data_for_club("DRIVER")
        assert len(results_lower) == len(results_upper)

    def test_no_match_returns_empty(self) -> None:
        results = get_validation_data_for_club("Nonexistent Club XYZ")
        assert results == []
