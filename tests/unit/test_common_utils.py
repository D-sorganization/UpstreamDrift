"""Unit tests for shared/python/common_utils.py hardening."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shared.python.common_utils import (
    convert_units,
    ensure_output_dir,
    standardize_joint_angles,
)
from shared.python.exceptions import ArrayDimensionError, ValidationConstraintError


class TestCommonUtilsHardening(unittest.TestCase):
    """Test suite for hardened utility functions."""

    @patch("shared.python.OUTPUT_ROOT", Path("/tmp/output"))
    def test_ensure_output_dir_traversal(self):
        """Test path traversal prevention in ensure_output_dir."""
        # Test engine name traversal
        with self.assertRaises(ValidationConstraintError):
            ensure_output_dir("../malicious")

        with self.assertRaises(ValidationConstraintError):
            ensure_output_dir("engine/sub")

        with self.assertRaises(ValidationConstraintError):
            ensure_output_dir("valid", subdir="../hidden")

    @patch("shared.python.OUTPUT_ROOT", Path("/tmp/output"))
    def test_ensure_output_dir_valid(self):
        """Test valid output dir creation."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            path = ensure_output_dir("engine_a")
            self.assertEqual(path, Path("/tmp/output/engine_a"))
            mock_mkdir.assert_called_once()

    def test_standardize_joint_angles_validation(self):
        """Test validation in standardize_joint_angles."""
        # 1D array
        bad_arr_1d = np.array([1, 2, 3])
        with self.assertRaises(ArrayDimensionError):
            standardize_joint_angles(bad_arr_1d)

        # 3D array
        bad_arr_3d = np.zeros((10, 3, 3))
        with self.assertRaises(ArrayDimensionError):
            standardize_joint_angles(bad_arr_3d)

        # Mismatched names
        good_arr = np.zeros((10, 3))
        bad_names = ["j1", "j2"]
        with self.assertRaises(ArrayDimensionError):
            standardize_joint_angles(good_arr, angle_names=bad_names)

        # NaN values
        nan_arr = np.array([[1.0, np.nan], [0.0, 1.0]])
        with self.assertRaises(ValidationConstraintError):
            standardize_joint_angles(nan_arr)

        # Valid case
        df = standardize_joint_angles(good_arr)
        self.assertEqual(len(df.columns), 4)  # 3 joints + time
        self.assertIn("time", df.columns)

    def test_convert_units_validation(self):
        """Test validation in convert_units."""
        with self.assertRaises(ValidationConstraintError):
            convert_units(10.0, "furlongs", "fortnights")

        # Valid case
        val = convert_units(180, "deg", "rad")
        self.assertAlmostEqual(val, np.pi)


if __name__ == "__main__":
    unittest.main()
