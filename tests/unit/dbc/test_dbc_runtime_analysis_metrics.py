"""Runtime DbC tests for analysis metrics modules.

Tests the ensure()/require() contracts added to:
- energy_metrics.py (EnergyMetricsMixin)
- stability_metrics.py (StabilityMetricsMixin)
- angular_momentum.py (AngularMomentumMetricsMixin)
"""

from __future__ import annotations

import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Energy Metrics
# ---------------------------------------------------------------------------


class _EnergyHost:
    """Minimal host class that mixes in EnergyMetricsMixin."""

    club_head_speed: np.ndarray | None = None

    def __init__(self, club_head_speed: np.ndarray | None = None) -> None:
        from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin

        # Dynamically mix in
        self.__class__ = type(
            "_EnergyHostMixed",
            (EnergyMetricsMixin,),
            {},
        )
        if club_head_speed is not None:
            self.club_head_speed = club_head_speed


class TestEnergyMetricsPreconditions(unittest.TestCase):
    """Verify require() contracts fire on bad inputs."""

    def _host(self, chs: np.ndarray | None = None) -> object:
        from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin

        class Host(EnergyMetricsMixin):
            pass

        h = Host()
        if chs is not None:
            h.club_head_speed = chs  # type: ignore[attr-defined]
        return h

    def test_empty_kinetic_raises(self) -> None:
        h = self._host()
        with self.assertRaises((ValueError, Exception)):
            h.compute_energy_metrics(np.array([]), np.array([1.0]))  # type: ignore[attr-defined]

    def test_empty_potential_raises(self) -> None:
        h = self._host()
        with self.assertRaises((ValueError, Exception)):
            h.compute_energy_metrics(np.array([1.0]), np.array([]))  # type: ignore[attr-defined]

    def test_mismatched_lengths_raises(self) -> None:
        h = self._host()
        with self.assertRaises((ValueError, Exception)):
            h.compute_energy_metrics(  # type: ignore[attr-defined]
                np.array([1.0, 2.0]), np.array([1.0])
            )

    def test_negative_kinetic_energy_raises(self) -> None:
        h = self._host()
        with self.assertRaises((ValueError, Exception)):
            h.compute_energy_metrics(  # type: ignore[attr-defined]
                np.array([-1.0, 2.0]), np.array([1.0, 2.0])
            )

    def test_inf_kinetic_raises(self) -> None:
        h = self._host()
        with self.assertRaises((ValueError, Exception)):
            h.compute_energy_metrics(  # type: ignore[attr-defined]
                np.array([np.inf, 2.0]), np.array([1.0, 2.0])
            )

    def test_nan_potential_raises(self) -> None:
        h = self._host()
        with self.assertRaises((ValueError, Exception)):
            h.compute_energy_metrics(  # type: ignore[attr-defined]
                np.array([1.0, 2.0]), np.array([np.nan, 2.0])
            )


class TestEnergyMetricsPostconditions(unittest.TestCase):
    """Verify ensure() contracts in output."""

    def _host(self, chs: np.ndarray | None = None) -> object:
        from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin

        class Host(EnergyMetricsMixin):
            pass

        h = Host()
        if chs is not None:
            h.club_head_speed = chs  # type: ignore[attr-defined]
        return h

    def test_all_values_finite(self) -> None:
        h = self._host(np.array([0.0, 5.0, 10.0, 8.0]))
        result = h.compute_energy_metrics(  # type: ignore[attr-defined]
            np.array([0.0, 10.0, 50.0, 30.0]),
            np.array([100.0, 80.0, 40.0, 60.0]),
        )
        for key, val in result.items():
            self.assertTrue(np.isfinite(val), f"{key} is not finite: {val}")

    def test_efficiency_non_negative(self) -> None:
        h = self._host(np.array([0.0, 5.0, 10.0]))
        result = h.compute_energy_metrics(  # type: ignore[attr-defined]
            np.array([0.0, 50.0, 100.0]),
            np.array([100.0, 50.0, 0.0]),
        )
        self.assertGreaterEqual(result["energy_efficiency"], 0.0)

    def test_without_club_head_speed(self) -> None:
        h = self._host()
        result = h.compute_energy_metrics(  # type: ignore[attr-defined]
            np.array([10.0, 20.0, 30.0]),
            np.array([30.0, 20.0, 10.0]),
        )
        self.assertEqual(result["energy_efficiency"], 0.0)

    def test_expected_keys_present(self) -> None:
        h = self._host()
        result = h.compute_energy_metrics(  # type: ignore[attr-defined]
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 1.0]),
        )
        expected = {
            "max_kinetic_energy",
            "max_potential_energy",
            "max_total_energy",
            "energy_efficiency",
            "energy_variation",
            "energy_drift",
        }
        self.assertEqual(set(result.keys()), expected)


# ---------------------------------------------------------------------------
# Angular Momentum Metrics
# ---------------------------------------------------------------------------


class TestAngularMomentumPostconditions(unittest.TestCase):
    """Verify ensure() contracts on AngularMomentumMetrics output."""

    def _host(
        self,
        am: np.ndarray | None = None,
        times: np.ndarray | None = None,
    ) -> object:
        from src.shared.python.analysis.angular_momentum import (
            AngularMomentumMetricsMixin,
        )

        class Host(AngularMomentumMetricsMixin):
            pass

        h = Host()
        h.angular_momentum = am  # type: ignore[attr-defined]
        h.times = times  # type: ignore[attr-defined]
        return h

    def test_returns_none_for_empty(self) -> None:
        h = self._host(am=np.array([]))
        result = h.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        self.assertIsNone(result)

    def test_returns_none_for_none(self) -> None:
        h = self._host(am=None)
        result = h.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        self.assertIsNone(result)

    def test_peak_magnitude_non_negative(self) -> None:
        am = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        h = self._host(am=am, times=np.array([0.0, 0.1, 0.2]))
        result = h.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.peak_magnitude, 0)

    def test_mean_magnitude_non_negative(self) -> None:
        am = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        h = self._host(am=am, times=np.array([0.0, 0.1]))
        result = h.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.mean_magnitude, 0)

    def test_variability_non_negative(self) -> None:
        am = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        h = self._host(am=am, times=np.array([0.0, 0.1, 0.2]))
        result = h.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.variability, 0)

    def test_component_peaks_non_negative(self) -> None:
        am = np.array([[-5.0, 3.0, -1.0], [2.0, -7.0, 4.0]])
        h = self._host(am=am, times=np.array([0.0, 0.1]))
        result = h.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.peak_lx, 0)
        self.assertGreaterEqual(result.peak_ly, 0)
        self.assertGreaterEqual(result.peak_lz, 0)

    def test_peak_time_exists(self) -> None:
        am = np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        h = self._host(am=am, times=np.array([0.0, 0.5, 1.0]))
        result = h.compute_angular_momentum_metrics()  # type: ignore[attr-defined]
        self.assertAlmostEqual(result.peak_time, 0.5)


# ---------------------------------------------------------------------------
# Stability Metrics
# ---------------------------------------------------------------------------


class TestStabilityMetricsPostconditions(unittest.TestCase):
    """Verify ensure() contracts on StabilityMetrics output."""

    def _host(
        self,
        cop: np.ndarray | None = None,
        com: np.ndarray | None = None,
    ) -> object:
        from src.shared.python.analysis.stability_metrics import StabilityMetricsMixin

        class Host(StabilityMetricsMixin):
            pass

        h = Host()
        h.cop_position = cop  # type: ignore[attr-defined]
        h.com_position = com  # type: ignore[attr-defined]
        return h

    def test_returns_none_missing_cop(self) -> None:
        h = self._host(cop=None, com=np.zeros((10, 3)))
        result = h.compute_stability_metrics()  # type: ignore[attr-defined]
        self.assertIsNone(result)

    def test_returns_none_missing_com(self) -> None:
        h = self._host(cop=np.zeros((10, 3)), com=None)
        result = h.compute_stability_metrics()  # type: ignore[attr-defined]
        self.assertIsNone(result)

    def test_returns_none_length_mismatch(self) -> None:
        h = self._host(cop=np.zeros((10, 3)), com=np.zeros((5, 3)))
        result = h.compute_stability_metrics()  # type: ignore[attr-defined]
        self.assertIsNone(result)

    def test_distances_non_negative(self) -> None:
        cop = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        com = np.array([[0.0, 0.0, 1.0], [0.1, 0.05, 1.0], [0.15, 0.1, 1.0]])
        h = self._host(cop=cop, com=com)
        result = h.compute_stability_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.min_com_cop_distance, 0)
        self.assertGreaterEqual(result.max_com_cop_distance, 0)
        self.assertGreaterEqual(result.mean_com_cop_distance, 0)

    def test_angles_in_valid_range(self) -> None:
        cop = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        com = np.array([[0.0, 0.0, 1.0], [0.3, 0.2, 0.8]])
        h = self._host(cop=cop, com=com)
        result = h.compute_stability_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.peak_inclination_angle, 0)
        self.assertLessEqual(result.peak_inclination_angle, 180.0)
        self.assertGreaterEqual(result.mean_inclination_angle, 0)
        self.assertLessEqual(result.mean_inclination_angle, 180.0)

    def test_min_leq_max_distance(self) -> None:
        rng = np.random.default_rng(42)
        cop = rng.uniform(0, 1, (50, 3))
        com = rng.uniform(0, 1, (50, 3))
        h = self._host(cop=cop, com=com)
        result = h.compute_stability_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertLessEqual(result.min_com_cop_distance, result.max_com_cop_distance)

    def test_2d_cop_supported(self) -> None:
        """CoP can be 2D (floor plane)."""
        cop = np.array([[0.0, 0.0], [0.1, 0.0]])
        com = np.array([[0.0, 0.0, 1.0], [0.1, 0.05, 0.95]])
        h = self._host(cop=cop, com=com)
        result = h.compute_stability_metrics()  # type: ignore[attr-defined]
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.min_com_cop_distance, 0)


if __name__ == "__main__":
    unittest.main()
