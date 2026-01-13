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


class TestCommonUtilsHardening(unittest.TestCase):
    """Test suite for hardened utility functions."""

    @patch("shared.python.OUTPUT_ROOT", Path("/tmp/output"))
    def test_ensure_output_dir_traversal(self) -> None:
        """Test path traversal prevention in ensure_output_dir."""
        # Test that the function works with valid engine names
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            path = ensure_output_dir("valid_engine")
            self.assertEqual(path, Path("/tmp/output/valid_engine"))
            mock_mkdir.assert_called_once()

        # Test with potentially problematic names (should still work)
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            path = ensure_output_dir("engine_with_underscore")
            self.assertEqual(path, Path("/tmp/output/engine_with_underscore"))
            mock_mkdir.assert_called_once()

    @patch("shared.python.OUTPUT_ROOT", Path("/tmp/output"))
    def test_ensure_output_dir_valid(self) -> None:
        """Test valid output dir creation."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            path = ensure_output_dir("engine_a")
            self.assertEqual(path, Path("/tmp/output/engine_a"))
            mock_mkdir.assert_called_once()

    def test_standardize_joint_angles_validation(self) -> None:
        """Test validation in standardize_joint_angles."""
        # 1D array - should raise IndexError when trying to access shape[1]
        bad_arr_1d = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            standardize_joint_angles(bad_arr_1d)

        # Valid 2D array
        good_arr = np.zeros((10, 3))
        df = standardize_joint_angles(good_arr)
        self.assertEqual(len(df.columns), 4)  # 3 joints + time
        self.assertIn("time", df.columns)

        # Test with custom names
        custom_names = ["joint_1", "joint_2", "joint_3"]
        df_custom = standardize_joint_angles(good_arr, angle_names=custom_names)
        self.assertEqual(len(df_custom.columns), 4)  # 3 joints + time
        for name in custom_names:
            self.assertIn(name, df_custom.columns)

    def test_convert_units_validation(self) -> None:
        """Test validation in convert_units."""
        with self.assertRaises(ValueError):
            convert_units(10.0, "furlongs", "fortnights")

        # Valid case
        val = convert_units(180, "deg", "rad")
        self.assertAlmostEqual(val, np.pi)


if __name__ == "__main__":
    unittest.main()
