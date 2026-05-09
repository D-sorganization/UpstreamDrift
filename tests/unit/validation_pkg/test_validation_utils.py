"""Tests for src.shared.python.validation_pkg.validation_utils (Issues #1949, #1744)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from src.shared.python.validation_pkg.validation_utils import (
    validate_array_dimensions,
    validate_array_length,
    validate_array_shape,
    validate_directory_exists,
    validate_extension,
    validate_file_exists,
    validate_not_none,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# validate_array_shape
# ---------------------------------------------------------------------------


class TestValidateArrayShape:
    def test_correct_shape_passes(self) -> None:
        arr = np.zeros((3, 3))
        validate_array_shape(arr, (3, 3))  # no exception

    def test_validation_utils_wrong_shape_raises(self) -> None:
        arr = np.zeros((2, 3))
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_array_shape(arr, (3, 3), "rotation")

    def test_name_in_error(self) -> None:
        arr = np.zeros((2,))
        with pytest.raises(ValueError, match="my_array"):
            validate_array_shape(arr, (3,), "my_array")

    def test_1d_shape(self) -> None:
        arr = np.ones(6)
        validate_array_shape(arr, (6,))


# ---------------------------------------------------------------------------
# validate_array_dimensions
# ---------------------------------------------------------------------------


class TestValidateArrayDimensions:
    def test_correct_ndim_passes(self) -> None:
        arr = np.zeros((3, 4))
        validate_array_dimensions(arr, 2)

    def test_wrong_ndim_raises(self) -> None:
        arr = np.zeros((3,))
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_array_dimensions(arr, 2, "position")

    def test_3d_accepted(self) -> None:
        arr = np.zeros((2, 3, 4))
        validate_array_dimensions(arr, 3)


# ---------------------------------------------------------------------------
# validate_array_length
# ---------------------------------------------------------------------------


class TestValidateArrayLength:
    def test_correct_length_passes(self) -> None:
        arr = np.zeros(5)
        validate_array_length(arr, 5)

    def test_validation_utils_wrong_length_raises(self) -> None:
        arr = np.zeros(3)
        with pytest.raises(ValueError, match="length mismatch"):
            validate_array_length(arr, 5, "q")

    def test_name_in_error(self) -> None:
        arr = np.zeros(2)
        with pytest.raises(ValueError, match="joints"):
            validate_array_length(arr, 3, "joints")


# ---------------------------------------------------------------------------
# validate_positive
# ---------------------------------------------------------------------------


class TestValidatePositive:
    def test_positive_value_passes(self) -> None:
        validate_positive(1.0)

    def test_zero_strict_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            validate_positive(0.0, "mass")

    def test_negative_strict_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            validate_positive(-1.0, "mass")

    def test_zero_non_strict_passes(self) -> None:
        validate_positive(0.0, strict=False)

    def test_negative_non_strict_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            validate_positive(-0.1, "distance", strict=False)

    def test_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="stiffness"):
            validate_positive(-5, "stiffness")


# ---------------------------------------------------------------------------
# validate_range
# ---------------------------------------------------------------------------


class TestValidateRange:
    def test_in_range_passes(self) -> None:
        validate_range(0.5, 0.0, 1.0)

    def test_at_min_inclusive(self) -> None:
        validate_range(0.0, 0.0, 1.0, inclusive=True)

    def test_at_max_inclusive(self) -> None:
        validate_range(1.0, 0.0, 1.0, inclusive=True)

    def test_below_min_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            validate_range(-0.1, 0.0, 1.0, "probability")

    def test_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            validate_range(1.1, 0.0, 1.0, "probability")

    def test_exclusive_boundary_raises_at_endpoint(self) -> None:
        with pytest.raises(ValueError):
            validate_range(0.0, 0.0, 1.0, inclusive=False)

    def test_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="angle"):
            validate_range(4.0, 0.0, 3.14, "angle")


# ---------------------------------------------------------------------------
# validate_file_exists
# ---------------------------------------------------------------------------


class TestValidateFileExists:
    def test_existing_file_returns_path(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            result = validate_file_exists(f.name)
            assert isinstance(result, Path)
            assert result == Path(f.name)

    def test_validation_utils_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_file_exists("/nonexistent/path/file.xml")

    def test_directory_raises_value_error(self) -> None:
        with (
            tempfile.TemporaryDirectory() as d,
            pytest.raises(ValueError, match="not a file"),
        ):
            validate_file_exists(d)


# ---------------------------------------------------------------------------
# validate_directory_exists
# ---------------------------------------------------------------------------


class TestValidateDirectoryExists:
    def test_existing_dir_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            result = validate_directory_exists(d)
            assert isinstance(result, Path)

    def test_missing_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_directory_exists("/nonexistent/dir/path")

    def test_file_raises_value_error(self) -> None:
        with (
            tempfile.NamedTemporaryFile() as f,
            pytest.raises(ValueError, match="not a directory"),
        ):
            validate_directory_exists(f.name)


# ---------------------------------------------------------------------------
# validate_extension
# ---------------------------------------------------------------------------


class TestValidateExtension:
    def test_allowed_extension_passes(self) -> None:
        validate_extension("model.urdf", [".urdf", ".xml"])

    def test_disallowed_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid extension"):
            validate_extension("model.txt", [".urdf", ".xml"])

    def test_validation_utils_case_insensitive(self) -> None:
        validate_extension("model.URDF", [".urdf"])

    def test_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="robot_model"):
            validate_extension("file.csv", [".urdf"], "robot_model")


# ---------------------------------------------------------------------------
# validate_not_none
# ---------------------------------------------------------------------------


class TestValidateNotNone:
    def test_non_none_passes(self) -> None:
        validate_not_none(0)
        validate_not_none("")
        validate_not_none(False)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="None"):
            validate_not_none(None)

    def test_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="model"):
            validate_not_none(None, "model")
