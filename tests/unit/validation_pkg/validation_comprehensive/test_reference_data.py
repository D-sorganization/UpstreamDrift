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


class TestReferenceData:
    """Tests for the reference data collections."""

    @pytest.mark.parametrize(
        "collection, expected_nonempty",
        [
            (PGA_TOUR_2024, True),
            (AMATEUR_AVERAGES, True),
        ],
        ids=["pga-tour", "amateur"],
    )
    def test_collections_not_empty(
        self, collection: list, expected_nonempty: bool
    ) -> None:
        assert (len(collection) > 0) == expected_nonempty

    def test_all_data_is_union(self) -> None:
        assert len(ALL_VALIDATION_DATA) == len(PGA_TOUR_2024) + len(AMATEUR_AVERAGES)

    @pytest.mark.parametrize(
        "club_name",
        ["Driver", "7-Iron"],
        ids=["driver", "7-iron"],
    )
    def test_pga_club_exists(self, club_name: str) -> None:
        clubs = [d.club for d in PGA_TOUR_2024]
        assert club_name in clubs

    @pytest.mark.parametrize(
        "attr, assertion",
        [
            ("ball_speed_mps", "positive"),
            ("carry_distance_m", "positive"),
        ],
        ids=["ball-speed", "carry-distance"],
    )
    def test_all_data_positive_values(self, attr: str, assertion: str) -> None:
        for d in ALL_VALIDATION_DATA:
            value = getattr(d, attr)
            assert value > 0, f"{d.club} has non-positive {attr}"

    def test_driver_carries_further_than_pw(self) -> None:
        driver = [d for d in PGA_TOUR_2024 if d.club == "Driver"][0]
        pw = [d for d in PGA_TOUR_2024 if d.club == "PW"][0]
        assert driver.carry_distance_m > pw.carry_distance_m

    def test_data_sources_valid(self) -> None:
        for d in ALL_VALIDATION_DATA:
            assert isinstance(d.source, DataSource)


# ============================================================================
# validation_data.py -- get_validation_data_for_club
# ============================================================================
