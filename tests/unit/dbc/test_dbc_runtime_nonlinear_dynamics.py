"""Runtime DbC tests for nonlinear dynamics contracts.

Tests the require()/ensure() contracts added to:
- estimate_lyapunov_exponent (preconditions: tau>=1, dim>=1, window>=1; postcondition: finite)
- compute_permutation_entropy (preconditions: order>=2, delay>=1; postcondition: >=0)
- compute_sample_entropy (preconditions: m>=1, r>0; postcondition: >=0)
- compute_fractal_dimension (precondition: k_max>=1; postcondition: finite)
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np


def _make_mixin(n: int = 500, n_joints: int = 3, dt: float = 0.01) -> object:
    """Create a mock NonlinearDynamicsMixin with realistic data."""
    from src.shared.python.analysis.nonlinear_dynamics import NonlinearDynamicsMixin

    rng = np.random.default_rng(42)
    obj = MagicMock(spec=NonlinearDynamicsMixin)
    t = np.arange(n) * dt
    obj.times = t
    obj.dt = dt
    obj.joint_positions = rng.standard_normal((n, n_joints)).cumsum(axis=0)
    obj.joint_velocities = rng.standard_normal((n, n_joints))

    # Bind real methods
    obj.compute_permutation_entropy = (
        NonlinearDynamicsMixin.compute_permutation_entropy.__get__(obj)
    )
    obj.compute_sample_entropy = NonlinearDynamicsMixin.compute_sample_entropy.__get__(
        obj
    )
    obj.compute_fractal_dimension = (
        NonlinearDynamicsMixin.compute_fractal_dimension.__get__(obj)
    )
    obj.estimate_lyapunov_exponent = (
        NonlinearDynamicsMixin.estimate_lyapunov_exponent.__get__(obj)
    )
    return obj


class TestPermutationEntropyContracts(unittest.TestCase):
    """Verify require()/ensure() on compute_permutation_entropy."""

    def test_valid_permutation_entropy(self) -> None:
        obj = _make_mixin()
        data = np.sin(np.linspace(0, 10 * np.pi, 200))
        result = obj.compute_permutation_entropy(data, order=3, delay=1)
        self.assertGreaterEqual(result, 0.0)

    def test_order_less_than_2_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_permutation_entropy(np.ones(50), order=1, delay=1)

    def test_order_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_permutation_entropy(np.ones(50), order=0, delay=1)

    def test_delay_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_permutation_entropy(np.ones(50), order=3, delay=0)

    def test_short_data_returns_zero(self) -> None:
        obj = _make_mixin()
        result = obj.compute_permutation_entropy(np.array([1.0, 2.0]), order=5, delay=1)
        self.assertEqual(result, 0.0)

    def test_constant_signal_low_entropy(self) -> None:
        obj = _make_mixin()
        result = obj.compute_permutation_entropy(np.ones(100), order=3, delay=1)
        self.assertEqual(result, 0.0)

    def test_random_signal_higher_entropy(self) -> None:
        obj = _make_mixin()
        rng = np.random.default_rng(42)
        result = obj.compute_permutation_entropy(
            rng.standard_normal(500), order=3, delay=1
        )
        self.assertGreater(result, 0.0)


class TestSampleEntropyContracts(unittest.TestCase):
    """Verify require()/ensure() on compute_sample_entropy."""

    def test_valid_sample_entropy(self) -> None:
        obj = _make_mixin()
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        result = obj.compute_sample_entropy(data, m=2, r=0.2)
        self.assertGreaterEqual(result, 0.0)

    def test_m_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_sample_entropy(np.ones(50), m=0, r=0.2)

    def test_r_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_sample_entropy(np.ones(50), m=2, r=0.0)

    def test_r_negative_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_sample_entropy(np.ones(50), m=2, r=-0.1)

    def test_short_data_returns_zero(self) -> None:
        obj = _make_mixin()
        result = obj.compute_sample_entropy(np.array([1.0]), m=2, r=0.2)
        self.assertEqual(result, 0.0)


class TestFractalDimensionContracts(unittest.TestCase):
    """Verify require()/ensure() on compute_fractal_dimension."""

    def test_valid_fractal_dimension(self) -> None:
        obj = _make_mixin()
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        result = obj.compute_fractal_dimension(data, k_max=10)
        self.assertTrue(np.isfinite(result))

    def test_k_max_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_fractal_dimension(np.ones(50), k_max=0)

    def test_k_max_negative_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.compute_fractal_dimension(np.ones(50), k_max=-5)

    def test_short_data_returns_default(self) -> None:
        obj = _make_mixin()
        result = obj.compute_fractal_dimension(np.array([1.0, 2.0]), k_max=10)
        self.assertEqual(result, 1.0)

    def test_result_is_finite(self) -> None:
        obj = _make_mixin()
        rng = np.random.default_rng(123)
        data = rng.standard_normal(300)
        result = obj.compute_fractal_dimension(data, k_max=5)
        self.assertTrue(np.isfinite(result))


class TestLyapunovExponentContracts(unittest.TestCase):
    """Verify require()/ensure() on estimate_lyapunov_exponent."""

    def test_valid_lyapunov(self) -> None:
        obj = _make_mixin(n=300)
        rng = np.random.default_rng(42)
        data = rng.standard_normal(300)
        result = obj.estimate_lyapunov_exponent(data, tau=1, dim=3, window=10)
        self.assertTrue(np.isfinite(result))

    def test_tau_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.estimate_lyapunov_exponent(np.ones(100), tau=0, dim=3, window=10)

    def test_dim_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.estimate_lyapunov_exponent(np.ones(100), tau=1, dim=0, window=10)

    def test_window_zero_raises(self) -> None:
        obj = _make_mixin()
        with self.assertRaises((ValueError, Exception)):
            obj.estimate_lyapunov_exponent(np.ones(100), tau=1, dim=3, window=0)

    def test_short_data_returns_zero(self) -> None:
        obj = _make_mixin()
        result = obj.estimate_lyapunov_exponent(
            np.array([1.0, 2.0]), tau=1, dim=3, window=10
        )
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
