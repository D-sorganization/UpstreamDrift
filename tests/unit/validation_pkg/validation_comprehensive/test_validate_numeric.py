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


class TestValidateNumeric:
    """Tests for validate_numeric."""

    @pytest.mark.parametrize(
        "value, name",
        [
            (42, "count"),
            (3.14, "pi"),
            (np.float64(1.5), "val"),
        ],
        ids=["int", "float", "numpy-float"],
    )
    def test_valid_numeric(self, value: object, name: str) -> None:
        validate_numeric(value, name)

    def test_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            validate_numeric("hello", "val")

    @pytest.mark.parametrize(
        "value, kwargs, error_type, match",
        [
            (float("nan"), {}, ValueError, "NaN"),
            (float("inf"), {}, ValueError, "infinite"),
        ],
        ids=["nan-default", "inf-default"],
    )
    def test_special_values_raise(
        self,
        value: float,
        kwargs: dict,
        error_type: type,
        match: str,
    ) -> None:
        with pytest.raises(error_type, match=match):
            validate_numeric(value, "val", **kwargs)

    @pytest.mark.parametrize(
        "value, kwargs",
        [
            (float("nan"), {"allow_nan": True}),
            (float("inf"), {"allow_inf": True}),
        ],
        ids=["nan-allowed", "inf-allowed"],
    )
    def test_special_values_allowed(self, value: float, kwargs: dict) -> None:
        validate_numeric(value, "val", **kwargs)


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
