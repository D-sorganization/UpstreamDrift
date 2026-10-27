"""Unit tests for shared/python/cross_engine_validator.py."""

import unittest

import numpy as np
import pytest
from src.shared.python.engine_core.cross_engine_validator import (
    CrossEngineValidator,
)


class TestCrossEngineValidator(unittest.TestCase):
    """Test suite for cross-engine validator."""

    def setUp(self) -> None:
        """Set up validator."""
        self.validator = CrossEngineValidator()

    def test_compare_states_exact_match(self) -> None:
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

    def test_compare_states_within_tolerance(self) -> None:
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

    def test_compare_states_warning(self) -> None:
        """Test validation in warning range."""
        tol = self.validator.TOLERANCES["position"]
        dev = tol * 0.9  # > 0.8 (WARNING_THRESHOLD) and <= 1.0
        res = self.validator.compare_states(
            "engine1",
            np.array([1.0]),
            "engine2",
            np.array([1.0 + dev]),
            metric="position",
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.severity, "WARNING")

    def test_compare_states_error(self) -> None:
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

    def test_compare_states_blocker(self) -> None:
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

    @pytest.mark.unit
    def test_classify_severity_fails_beyond_tolerance(self) -> None:
        """UD-18: deviation > tol => passed=False."""
        tol = 1e-6
        # Just beyond tolerance
        passed, severity = self.validator._classify_severity(tol * 1.0001, tol)
        self.assertFalse(passed)
        self.assertEqual(severity, "ERROR")

        # 1.5x tolerance (previously returned passed=True with WARNING)
        passed, severity = self.validator._classify_severity(tol * 1.5, tol)
        self.assertFalse(passed)
        self.assertEqual(severity, "ERROR")

        # Via compare_states: deviation > tolerance must fail validation
        res = self.validator.compare_states(
            "engine1",
            np.array([1.0]),
            "engine2",
            np.array([1.0 + tol * 1.5]),
            metric="position",
        )
        self.assertFalse(res.passed)
        self.assertEqual(res.severity, "ERROR")
        self.assertEqual(res.tolerance, tol)

    def test_invalid_metric(self) -> None:
        """Test unknown metric raises ValueError."""
        with self.assertRaises(ValueError):
            self.validator.compare_states(
                "e1",
                np.zeros(1),
                "e2",
                np.zeros(1),
                metric="unknown",  # type: ignore[arg-type]
            )

    def test_shape_mismatch(self) -> None:
        """Test handling of shape mismatch."""
        res = self.validator.compare_states(
            "e1", np.zeros(1), "e2", np.zeros(2), metric="position"
        )
        self.assertFalse(res.passed)
        self.assertIn("Shape mismatch", res.message)

    def test_compare_torques_with_rms(self) -> None:
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
