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


class TestPhysicalValidationError:
    """Tests for PhysicalValidationError class."""

    def test_old_style_message(self) -> None:
        err = PhysicalValidationError("mass is bad")
        assert "mass is bad" in str(err)

    def test_new_style_structured(self) -> None:
        err = PhysicalValidationError("mass", value=-1.0, physical_constraint="mass>0")
        assert err.physical_constraint == "mass>0"

    def test_is_exception(self) -> None:
        with pytest.raises(PhysicalValidationError):
            raise PhysicalValidationError("test error")


# ============================================================================
# validation.py -- validate_mass
# ============================================================================


class TestValidateMass:
    """Tests for validate_mass function."""

    @pytest.mark.parametrize(
        "value",
        [1.5, 1e-10, 100.0, 0.001],
        ids=["normal", "tiny-positive", "large", "small"],
    )
    def test_valid_mass(self, value: float) -> None:
        validate_mass(value)  # Should not raise

    @pytest.mark.parametrize(
        "value",
        [0.0, -1.0, -0.001],
        ids=["zero", "negative", "small-negative"],
    )
    def test_invalid_mass_raises(self, value: float) -> None:
        with pytest.raises(PhysicalValidationError):
            validate_mass(value)

    def test_custom_param_name(self) -> None:
        with pytest.raises(PhysicalValidationError, match="head_mass"):
            validate_mass(-0.5, param_name="head_mass")


# ============================================================================
# validation.py -- validate_timestep
# ============================================================================


class TestValidateTimestep:
    """Tests for validate_timestep function."""

    @pytest.mark.parametrize(
        "value",
        [0.001, 1.0, 1.5],
        ids=["typical", "boundary", "large"],
    )
    def test_valid_timestep(self, value: float) -> None:
        validate_timestep(value)  # Should not raise

    @pytest.mark.parametrize(
        "value",
        [0.0, -0.01],
        ids=["zero", "negative"],
    )
    def test_invalid_timestep_raises(self, value: float) -> None:
        with pytest.raises(PhysicalValidationError):
            validate_timestep(value)


# ============================================================================
# validation.py -- validate_inertia_matrix
# ============================================================================


class TestValidateInertiaMatrix:
    """Tests for validate_inertia_matrix function."""

    @pytest.mark.parametrize(
        "inertia",
        [
            np.diag([1.0, 2.0, 3.0]),
            np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.1], [0.0, 0.1, 1.5]]),
        ],
        ids=["diagonal", "symmetric"],
    )
    def test_valid_inertia(self, inertia: np.ndarray) -> None:
        validate_inertia_matrix(inertia)  # Should not raise

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(PhysicalValidationError, match="shape"):
            validate_inertia_matrix(np.eye(4))

    def test_asymmetric_raises(self) -> None:
        inertia = np.array([[1.0, 0.5, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(PhysicalValidationError, match="symmetric"):
            validate_inertia_matrix(inertia)

    @pytest.mark.parametrize(
        "inertia, description",
        [
            (
                np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
                "negative-eigenvalue",
            ),
            (
                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                "zero-eigenvalue",
            ),
        ],
        ids=["negative-eigenvalue", "zero-eigenvalue"],
    )
    def test_non_positive_definite_raises(
        self, inertia: np.ndarray, description: str
    ) -> None:
        with pytest.raises(PhysicalValidationError):
            validate_inertia_matrix(inertia)


# ============================================================================
# validation.py -- validate_joint_limits
# ============================================================================


class TestValidateJointLimits:
    """Tests for validate_joint_limits function."""

    def test_valid_limits(self) -> None:
        q_min = np.array([0.0, -np.pi])
        q_max = np.array([np.pi, np.pi])
        validate_joint_limits(q_min, q_max)  # Should not raise

    def test_reversed_raises(self) -> None:
        q_min = np.array([1.0, 1.0])
        q_max = np.array([0.0, 2.0])
        with pytest.raises(PhysicalValidationError, match="q_min"):
            validate_joint_limits(q_min, q_max)

    def test_equal_raises(self) -> None:
        q = np.array([1.0, 2.0])
        with pytest.raises(PhysicalValidationError):
            validate_joint_limits(q, q)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(PhysicalValidationError, match="shape"):
            validate_joint_limits(np.zeros(2), np.ones(3))


# ============================================================================
# validation.py -- validate_friction_coefficient
# ============================================================================


class TestValidateFrictionCoefficient:
    """Tests for validate_friction_coefficient function."""

    @pytest.mark.parametrize(
        "value",
        [0.5, 0.0, 1.0],
        ids=["typical", "frictionless", "high"],
    )
    def test_valid_friction(self, value: float) -> None:
        validate_friction_coefficient(value)

    def test_negative_raises(self) -> None:
        with pytest.raises(PhysicalValidationError):
            validate_friction_coefficient(-0.1)


# ============================================================================
# validation.py -- validate_physical_bounds decorator
# ============================================================================


class TestValidatePhysicalBounds:
    """Tests for the validate_physical_bounds decorator."""

    def test_decorator_passes_valid(self) -> None:
        @validate_physical_bounds
        def set_mass(mass: float) -> float:
            return mass

        assert set_mass(mass=1.5) == 1.5

    @pytest.mark.parametrize(
        "param_name, invalid_value",
        [
            ("mass", -1.0),
            ("dt", 0.0),
            ("friction", -0.5),
        ],
        ids=["negative-mass", "zero-dt", "negative-friction"],
    )
    def test_decorator_catches_invalid(
        self, param_name: str, invalid_value: float
    ) -> None:
        @validate_physical_bounds
        def func(**kwargs: float) -> float:
            return list(kwargs.values())[0]

        with pytest.raises(PhysicalValidationError):
            func(**{param_name: invalid_value})


# ============================================================================
# validation_utils.py -- Array Validators
# ============================================================================


class TestValidateArrayShape:
    """Tests for validate_array_shape."""

    def test_correct_shape(self) -> None:
        validate_array_shape(np.eye(3), (3, 3))

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_array_shape(np.eye(3), (2, 2))


class TestValidateArrayDimensions:
    """Tests for validate_array_dimensions."""

    def test_correct_ndim(self) -> None:
        validate_array_dimensions(np.zeros(5), 1)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            validate_array_dimensions(np.zeros((3, 3)), 1)


class TestValidateArrayLength:
    """Tests for validate_array_length."""

    def test_correct_length(self) -> None:
        validate_array_length(np.zeros(5), 5)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            validate_array_length(np.zeros(5), 3)


# ============================================================================
# validation_utils.py -- Scalar Validators
# ============================================================================


class TestValidatePositive:
    """Tests for validate_positive."""

    @pytest.mark.parametrize(
        "value, name, strict, should_raise",
        [
            (1.0, "mass", True, False),
            (0.0, "mass", True, True),
            (0.0, "distance", False, False),
            (-1.0, "distance", False, True),
        ],
        ids=[
            "positive-strict",
            "zero-strict",
            "zero-non-strict",
            "negative-non-strict",
        ],
    )
    def test_validate_positive(
        self, value: float, name: str, strict: bool, should_raise: bool
    ) -> None:
        if should_raise:
            with pytest.raises(ValueError):
                validate_positive(value, name, strict=strict)
        else:
            validate_positive(value, name, strict=strict)


class TestValidateRange:
    """Tests for validate_range."""

    @pytest.mark.parametrize(
        "value, lo, hi, name, inclusive, should_raise",
        [
            (0.0, -1.0, 1.0, "angle", True, False),
            (1.0, 0.0, 1.0, "probability", True, False),
            (1.0, 0.0, 1.0, "val", False, True),
            (2.0, 0.0, 1.0, "probability", True, True),
        ],
        ids=[
            "within-range-inclusive",
            "at-boundary-inclusive",
            "at-boundary-exclusive",
            "out-of-range",
        ],
    )
    def test_validate_range(
        self,
        value: float,
        lo: float,
        hi: float,
        name: str,
        inclusive: bool,
        should_raise: bool,
    ) -> None:
        if should_raise:
            with pytest.raises(ValueError):
                validate_range(value, lo, hi, name, inclusive=inclusive)
        else:
            validate_range(value, lo, hi, name, inclusive=inclusive)


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


class TestValidateFileExists:
    """Tests for validate_file_exists."""

    def test_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = validate_file_exists(str(f))
        assert result == f

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_file_exists("/nonexistent/file.txt")

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            validate_file_exists(str(tmp_path))


class TestValidateDirectoryExists:
    """Tests for validate_directory_exists."""

    def test_existing_dir(self, tmp_path: Path) -> None:
        result = validate_directory_exists(str(tmp_path))
        assert result == tmp_path

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_directory_exists("/nonexistent/dir")

    def test_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            validate_directory_exists(str(f))


class TestValidateExtension:
    """Tests for validate_extension."""

    @pytest.mark.parametrize(
        "filename, allowed",
        [
            ("model.xml", [".xml", ".urdf"]),
            ("model.XML", [".xml"]),
        ],
        ids=["exact-match", "case-insensitive"],
    )
    def test_valid_extension(self, filename: str, allowed: list[str]) -> None:
        validate_extension(filename, allowed)

    def test_invalid_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid extension"):
            validate_extension("model.json", [".xml", ".urdf"])


# ============================================================================
# validation_utils.py -- Type/Dict Validators
# ============================================================================


class TestValidateNotNone:
    """Tests for validate_not_none."""

    def test_not_none(self) -> None:
        validate_not_none(42, "val")

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            validate_not_none(None, "model")


class TestValidateType:
    """Tests for validate_type."""

    @pytest.mark.parametrize(
        "value, expected_type, name",
        [
            (42, int, "count"),
            (3.14, (int, float), "number"),
        ],
        ids=["exact-type", "tuple-of-types"],
    )
    def test_correct_type(
        self, value: object, expected_type: type | tuple, name: str
    ) -> None:
        validate_type(value, expected_type, name)

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be"):
            validate_type("hello", int, "count")


class TestValidateDictKeys:
    """Tests for validate_dict_keys."""

    def test_all_required_present(self) -> None:
        data = {"engine": "mujoco", "model": "arm.xml"}
        validate_dict_keys(data, required_keys=["engine", "model"])

    def test_missing_required_raises(self) -> None:
        data = {"engine": "mujoco"}
        with pytest.raises(ValueError, match="missing required"):
            validate_dict_keys(data, required_keys=["engine", "model"])

    def test_unknown_keys_warn(self) -> None:
        data = {"engine": "mujoco", "model": "arm.xml", "extra": True}
        # Should log a warning for unknown keys
        validate_dict_keys(
            data,
            required_keys=["engine", "model"],
            optional_keys=["timestep"],
        )


# ============================================================================
# validation_utils.py -- validate_all
# ============================================================================


class TestValidateAll:
    """Tests for validate_all function."""

    def test_all_pass(self) -> None:
        validate_all(
            (validate_positive, (1.0, "mass"), {}),
            (validate_range, (0.5, 0.0, 1.0, "prob"), {}),
        )

    def test_collects_errors(self) -> None:
        from src.shared.python.core.error_utils import ValidationError

        with pytest.raises(ValidationError, match="2 error"):
            validate_all(
                (validate_positive, (-1.0, "mass"), {}),
                (validate_range, (2.0, 0.0, 1.0, "prob"), {}),
            )


# ============================================================================
# validation_helpers.py -- Constants
# ============================================================================


class TestPhysicsConstants:
    """Tests for physics validation constants."""

    @pytest.mark.parametrize(
        "constant_value, name",
        [
            (MAX_JOINT_VELOCITY_RAD_S, "MAX_JOINT_VELOCITY_RAD_S"),
            (MAX_JOINT_ACCELERATION_RAD_S2, "MAX_JOINT_ACCELERATION_RAD_S2"),
            (MAX_CARTESIAN_VELOCITY_M_S, "MAX_CARTESIAN_VELOCITY_M_S"),
            (MAX_CARTESIAN_ACCELERATION_M_S2, "MAX_CARTESIAN_ACCELERATION_M_S2"),
        ],
        ids=[
            "joint-velocity",
            "joint-acceleration",
            "cartesian-velocity",
            "cartesian-acceleration",
        ],
    )
    def test_constant_positive(self, constant_value: float, name: str) -> None:
        assert constant_value > 0, f"{name} must be positive"

    def test_max_joint_position_is_2pi(self) -> None:
        assert pytest.approx(2 * math.pi) == MAX_JOINT_POSITION_RAD

    def test_hierarchy(self) -> None:
        """Position limit in rad should be much smaller than velocity limit (rad/s)."""
        assert MAX_JOINT_VELOCITY_RAD_S > MAX_JOINT_POSITION_RAD


# ============================================================================
# validation_helpers.py -- ValidationLevel
# ============================================================================


class TestValidationLevel:
    """Tests for ValidationLevel enum."""

    def test_all_levels(self) -> None:
        levels = {v.value for v in ValidationLevel}
        assert levels == {"permissive", "standard", "strict"}


# ============================================================================
# validation_helpers.py -- validate_finite
# ============================================================================


class TestValidateFinite:
    """Tests for validate_finite function."""

    def test_finite_array_passes(self) -> None:
        validate_finite(np.array([1.0, 2.0, 3.0]), "data")

    def test_nan_strict_raises(self) -> None:
        with pytest.raises(PhysicsValidationError, match="NaN"):
            validate_finite(
                np.array([1.0, np.nan, 3.0]), "data", ValidationLevel.STRICT
            )

    def test_inf_standard_raises(self) -> None:
        with pytest.raises(PhysicsValidationError, match="NaN or Inf"):
            validate_finite(np.array([np.inf, 1.0]), "data", ValidationLevel.STANDARD)

    def test_nan_permissive_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_finite(np.array([np.nan]), "data", ValidationLevel.PERMISSIVE)
            assert len(w) >= 1
            assert "NaN" in str(w[0].message)


# ============================================================================
# validation_helpers.py -- validate_magnitude
# ============================================================================


class TestValidateMagnitude:
    """Tests for validate_magnitude function."""

    def test_within_bounds_passes(self) -> None:
        validate_magnitude(np.array([1.0, 2.0]), "vel", 10.0, "m/s")

    def test_exceeds_bound_strict_raises(self) -> None:
        with pytest.raises(PhysicsValidationError, match="implausibly"):
            validate_magnitude(
                np.array([200.0]), "vel", 100.0, "m/s", ValidationLevel.STRICT
            )

    def test_exceeds_bound_standard_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_magnitude(
                np.array([200.0]), "vel", 100.0, "m/s", ValidationLevel.STANDARD
            )
            assert len(w) >= 1


# ============================================================================
# validation_helpers.py -- validate_joint_state
# ============================================================================


class TestValidateJointState:
    """Tests for validate_joint_state function."""

    def test_valid_state(self) -> None:
        qpos = np.array([0.5, -0.3])
        qvel = np.array([1.0, 2.0])
        validate_joint_state(qpos, qvel)

    def test_nan_position_raises(self) -> None:
        with pytest.raises(PhysicsValidationError):
            validate_joint_state(np.array([np.nan, 0.0]))

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(PhysicsValidationError, match="Dimension"):
            validate_joint_state(np.array([0.5, 0.3]), qvel=np.array([1.0]))

    def test_position_only(self) -> None:
        validate_joint_state(np.array([0.5, -0.3]))

    def test_all_three(self) -> None:
        qpos = np.array([0.1, 0.2])
        qvel = np.array([1.0, 2.0])
        qacc = np.array([10.0, 20.0])
        validate_joint_state(qpos, qvel, qacc)


# ============================================================================
# validation_helpers.py -- validate_cartesian_state
# ============================================================================


class TestValidateCartesianState:
    """Tests for validate_cartesian_state function."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"position": np.array([1.0, 2.0, 3.0])},
            {"velocity": np.array([5.0, 10.0, 15.0])},
            {},
        ],
        ids=["position-only", "velocity-only", "all-none"],
    )
    def test_valid_cartesian_state(self, kwargs: dict) -> None:
        validate_cartesian_state(**kwargs)

    def test_nan_velocity_raises(self) -> None:
        with pytest.raises(PhysicsValidationError):
            validate_cartesian_state(
                velocity=np.array([np.nan, 1.0, 2.0]),
                level=ValidationLevel.STRICT,
            )


# ============================================================================
# validation_helpers.py -- validate_model_parameters
# ============================================================================


class TestValidateModelParameters:
    """Tests for validate_model_parameters function."""

    @pytest.mark.parametrize(
        "masses",
        [
            np.array([70.0, 5.0, 2.0]),
            np.array([500.0, 500.0]),
        ],
        ids=["valid-masses", "few-bodies-skip-total"],
    )
    def test_valid_masses(self, masses: np.ndarray) -> None:
        validate_model_parameters(masses)

    @pytest.mark.parametrize(
        "masses, match",
        [
            (np.array([70.0, 0.0, 2.0]), "positive"),
            (np.array([70.0, -1.0, 2.0]), "positive"),
            (np.array([np.nan, 5.0]), "NaN"),
        ],
        ids=["zero-mass", "negative-mass", "nan-mass"],
    )
    def test_invalid_masses_raise(self, masses: np.ndarray, match: str) -> None:
        with pytest.raises(PhysicsValidationError, match=match):
            validate_model_parameters(masses)

    def test_implausible_total_warns(self) -> None:
        # Many bodies with huge total mass
        masses = np.full(15, 50.0)  # 750 kg total > 500
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_model_parameters(masses)
            assert any("mass" in str(x.message).lower() for x in w)


# ============================================================================
# validation_data.py -- DataSource Enum
# ============================================================================


class TestDataSource:
    """Tests for DataSource enum."""

    def test_all_sources(self) -> None:
        expected = {
            "trackman_pga_tour",
            "golf_monthly",
            "usga_research",
            "kaggle_garmin",
        }
        assert {s.value for s in DataSource} == expected


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

    def test_case_insensitive(self) -> None:
        results_lower = get_validation_data_for_club("driver")
        results_upper = get_validation_data_for_club("DRIVER")
        assert len(results_lower) == len(results_upper)

    def test_no_match_returns_empty(self) -> None:
        results = get_validation_data_for_club("Nonexistent Club XYZ")
        assert results == []
