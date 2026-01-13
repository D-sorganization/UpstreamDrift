"""Unit tests for shared/python/cross_engine_validator.py."""

import unittest

import numpy as np

from shared.python.cross_engine_validator import (
    CrossEngineValidator,
)


class TestCrossEngineValidator(unittest.TestCase):
    """Test suite for cross-engine validator."""

    def setUp(self):
        """Set up validator."""
        self.validator = CrossEngineValidator()

    def test_compare_states_exact_match(self):
        """Test validation with exact match."""
        res = self.validator.compare_states(
            "engine1",
            np.array([1.0, 2.0]),
            "engine2",
            np.array([1.0, 2.0]),
            metric="position",
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.severity, "PASSED")
        self.assertEqual(res.max_deviation, 0.0)

    def test_compare_states_within_tolerance(self):
        """Test validation within tolerance."""
        tol = self.validator.TOLERANCES["position"]
        dev = tol * 0.5
        res = self.validator.compare_states(
            "engine1",
            np.array([1.0, 2.0]),
            "engine2",
            np.array([1.0 + dev, 2.0]),
            metric="position",
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.severity, "PASSED")

    def test_compare_states_warning(self):
        """Test validation in warning range."""
        tol = self.validator.TOLERANCES["position"]
        dev = tol * 1.5  # < 2.0 (WARNING_THRESHOLD)
        res = self.validator.compare_states(
            "engine1",
            np.array([1.0]),
            "engine2",
            np.array([1.0 + dev]),
            metric="position",
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.severity, "WARNING")

    def test_compare_states_error(self):
        """Test validation in error range."""
        tol = self.validator.TOLERANCES["position"]
        dev = tol * 5.0  # > 2.0, < 10.0 (ERROR_THRESHOLD)
        res = self.validator.compare_states(
            "engine1",
            np.array([1.0]),
            "engine2",
            np.array([1.0 + dev]),
            metric="position",
        )
        self.assertFalse(res.passed)
        self.assertEqual(res.severity, "ERROR")

    def test_compare_states_blocker(self):
        """Test validation in blocker range."""
        tol = self.validator.TOLERANCES["position"]
        dev = tol * 200.0  # > 100.0 (BLOCKER_THRESHOLD)
        res = self.validator.compare_states(
            "engine1",
            np.array([1.0]),
            "engine2",
            np.array([1.0 + dev]),
            metric="position",
        )
        self.assertFalse(res.passed)
        self.assertEqual(res.severity, "BLOCKER")

    def test_invalid_metric(self):
        """Test unknown metric raises ValueError."""
        with self.assertRaises(ValueError):
            self.validator.compare_states(
                "e1",
                np.zeros(1),
                "e2",
                np.zeros(1),
                metric="unknown",  # type: ignore[arg-type]
            )

    def test_shape_mismatch(self):
        """Test handling of shape mismatch."""
        res = self.validator.compare_states(
            "e1", np.zeros(1), "e2", np.zeros(2), metric="position"
        )
        self.assertFalse(res.passed)
        self.assertIn("Shape mismatch", res.message)

    def test_compare_torques_with_rms(self):
        """Test torque RMS comparison."""
        # 0% difference
        res = self.validator.compare_torques_with_rms(
            "e1", np.array([10.0]), "e2", np.array([10.0])
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.max_deviation, 0.0)

        # 10% difference
        # RMS(10) = 10
        # RMS(11) = 11
        # RMS(diff) = 1
        # Pct = 1/10 * 100 = 10%
        # Threshold default is 10% -> this is exactly on edge, usually float issues
        # Let's try explicit pass
        res = self.validator.compare_torques_with_rms(
            "e1", np.array([10.0]), "e2", np.array([10.5])
        )  # 5%
        self.assertTrue(res.passed)

        # Fail case
        res = self.validator.compare_torques_with_rms(
            "e1", np.array([10.0]), "e2", np.array([12.0])
        )  # 20%
        self.assertFalse(res.passed)
        self.assertGreater(res.max_deviation, 10.0)


if __name__ == "__main__":
    unittest.main()
