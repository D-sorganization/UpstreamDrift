"""Runtime DbC tests for ActivationDynamics.

Tests that the require()/ensure() contracts added to the source module
fire correctly at runtime — these are NOT just unit tests, they validate
the *contract enforcement mechanism* itself.
"""

from __future__ import annotations

import unittest

import numpy as np


class TestActivationDynamicsConstructorContracts(unittest.TestCase):
    """Verify constructor preconditions fire."""

    def _make(
        self,
        tau_act: float = 0.01,
        tau_deact: float = 0.04,
        min_act: float = 0.001,
    ) -> object:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        return ActivationDynamics(tau_act, tau_deact, min_act)

    def test_valid_construction(self) -> None:
        dyn = self._make()
        self.assertIsNotNone(dyn)

    def test_zero_tau_act_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(tau_act=0.0)

    def test_negative_tau_act_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(tau_act=-0.01)

    def test_zero_tau_deact_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(tau_deact=0.0)

    def test_negative_tau_deact_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(tau_deact=-0.04)

    def test_zero_min_activation_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(min_act=0.0)

    def test_negative_min_activation_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(min_act=-0.1)

    def test_min_activation_one_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(min_act=1.0)

    def test_min_activation_above_one_raises(self) -> None:
        with self.assertRaises((ValueError, Exception)):
            self._make(min_act=1.5)


class TestActivationDynamicsUpdateContracts(unittest.TestCase):
    """Verify update() preconditions and postconditions fire."""

    def _make(self) -> object:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        return ActivationDynamics(tau_act=0.01, tau_deact=0.04)

    def test_zero_dt_raises(self) -> None:
        dyn = self._make()
        with self.assertRaises((ValueError, Exception)):
            dyn.update(u=0.5, a=0.5, dt=0.0)  # type: ignore[attr-defined]

    def test_negative_dt_raises(self) -> None:
        dyn = self._make()
        with self.assertRaises((ValueError, Exception)):
            dyn.update(u=0.5, a=0.5, dt=-0.001)  # type: ignore[attr-defined]

    def test_positive_dt_ok(self) -> None:
        dyn = self._make()
        result = dyn.update(u=0.5, a=0.5, dt=0.001)  # type: ignore[attr-defined]
        self.assertIsInstance(result, float)

    def test_output_within_bounds(self) -> None:
        dyn = self._make()
        # Extreme inputs: high excitation, low activation, large dt
        result = dyn.update(u=1.0, a=0.0, dt=0.1)  # type: ignore[attr-defined]
        self.assertGreaterEqual(result, 0.001)  # min_activation
        self.assertLessEqual(result, 1.0)

    def test_output_within_bounds_deactivation(self) -> None:
        dyn = self._make()
        # Extreme inputs: low excitation, high activation
        result = dyn.update(u=0.0, a=1.0, dt=0.1)  # type: ignore[attr-defined]
        self.assertGreaterEqual(result, 0.001)
        self.assertLessEqual(result, 1.0)


class TestActivationDynamicsDerivativeContracts(unittest.TestCase):
    """Verify derivative postconditions."""

    def _make(self) -> object:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        return ActivationDynamics()

    def test_derivative_is_finite(self) -> None:
        dyn = self._make()
        result = dyn.compute_derivative(u=0.5, a=0.3)  # type: ignore[attr-defined]
        self.assertTrue(np.isfinite(result))

    def test_derivative_positive_for_activation(self) -> None:
        """When u > a, derivative should be positive (activation rising)."""
        dyn = self._make()
        result = dyn.compute_derivative(u=0.8, a=0.2)  # type: ignore[attr-defined]
        self.assertGreater(result, 0)

    def test_derivative_negative_for_deactivation(self) -> None:
        """When u < a, derivative should be negative (activation falling)."""
        dyn = self._make()
        result = dyn.compute_derivative(u=0.2, a=0.8)  # type: ignore[attr-defined]
        self.assertLess(result, 0)

    def test_derivative_near_zero_at_equilibrium(self) -> None:
        """When u ≈ a, derivative should be near zero."""
        dyn = self._make()
        result = dyn.compute_derivative(u=0.5, a=0.5)  # type: ignore[attr-defined]
        self.assertAlmostEqual(result, 0.0, places=5)


class TestActivationDynamicsPhysiologicalBehavior(unittest.TestCase):
    """Verify physiological invariants hold under contract enforcement."""

    def _make(self) -> object:
        from src.shared.python.biomechanics.activation_dynamics import (
            ActivationDynamics,
        )

        return ActivationDynamics(tau_act=0.01, tau_deact=0.04)

    def test_activation_faster_than_deactivation(self) -> None:
        """Activation should reach 90% faster than deactivation reaches 10%."""
        dyn = self._make()
        dt = 0.001

        # Activation phase: u=1, a starts at 0
        a = 0.0
        steps_to_90 = 0
        for _ in range(1000):
            a = dyn.update(u=1.0, a=a, dt=dt)  # type: ignore[attr-defined]
            steps_to_90 += 1
            if a >= 0.9:
                break

        # Deactivation phase: u=0, a starts at 1
        a = 1.0
        steps_to_10 = 0
        for _ in range(1000):
            a = dyn.update(u=0.0, a=a, dt=dt)  # type: ignore[attr-defined]
            steps_to_10 += 1
            if a <= 0.1:
                break

        self.assertLess(steps_to_90, steps_to_10)

    def test_steady_state_converges(self) -> None:
        """Constant excitation should converge to that excitation value."""
        dyn = self._make()
        a = 0.0
        target = 0.7
        dt = 0.001
        for _ in range(5000):
            a = dyn.update(u=target, a=a, dt=dt)  # type: ignore[attr-defined]
        self.assertAlmostEqual(a, target, places=1)

    def test_all_updates_in_bounds(self) -> None:
        """Every update must respect [min_activation, 1.0] bounds."""
        dyn = self._make()
        a = 0.5
        rng = np.random.default_rng(42)
        for _ in range(200):
            u = rng.uniform(0, 1)
            a = dyn.update(u=u, a=a, dt=0.001)  # type: ignore[attr-defined]
            self.assertGreaterEqual(a, 0.001)
            self.assertLessEqual(a, 1.0)


if __name__ == "__main__":
    unittest.main()
