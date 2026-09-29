"""Tests for analysis.angular_momentum module.

Validates AngularMomentumMetricsMixin.compute_angular_momentum_metrics()
using a minimal stub object that provides the required attributes.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub helper
# ---------------------------------------------------------------------------


class _Stub(AngularMomentumMetricsMixin):
    """Minimal host for the mixin — provides angular_momentum and times."""

    def __init__(
        self,
        angular_momentum: np.ndarray | None,
        times: np.ndarray | None = None,
    ) -> None:
        self.angular_momentum = angular_momentum
        self.times = times


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAngularMomentumMetricsMixin:
    def test_returns_none_when_no_attribute(self) -> None:
        stub = _Stub(angular_momentum=None)
        assert stub.compute_angular_momentum_metrics() is None

    def test_returns_none_when_empty_array(self) -> None:
        stub = _Stub(angular_momentum=np.empty((0, 3)))
        assert stub.compute_angular_momentum_metrics() is None

    def test_peak_magnitude_correct(self) -> None:
        # Three frames: magnitudes 1, 2, 3
        am = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        times = np.array([0.0, 0.5, 1.0])
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        assert result.peak_magnitude == pytest.approx(3.0)

    def test_mean_magnitude_correct(self) -> None:
        am = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        times = np.array([0.0, 0.5, 1.0])
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        expected_mean = (1.0 + 2.0 + 3.0) / 3
        assert result.mean_magnitude == pytest.approx(expected_mean)

    def test_component_peaks(self) -> None:
        am = np.array(
            [
                [3.0, -1.0, 2.0],
                [-2.0, 4.0, -3.0],
                [1.0, 0.0, 1.0],
            ]
        )
        times = np.array([0.0, 0.5, 1.0])
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        # peak_lx = max(|3|, |-2|, |1|) = 3
        assert result.peak_lx == pytest.approx(3.0)
        # peak_ly = max(|-1|, |4|, |0|) = 4
        assert result.peak_ly == pytest.approx(4.0)
        # peak_lz = max(|2|, |-3|, |1|) = 3
        assert result.peak_lz == pytest.approx(3.0)

    def test_variability_zero_for_constant_angular_momentum(self) -> None:
        am = np.tile([1.0, 0.0, 0.0], (5, 1))
        times = np.linspace(0.0, 1.0, 5)
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        assert result.variability == pytest.approx(0.0, abs=1e-10)

    def test_angular_momentum_variability_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        am = rng.standard_normal((20, 3))
        times = np.linspace(0.0, 1.0, 20)
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        assert result.variability >= 0.0

    def test_postconditions_hold(self) -> None:
        am = np.array([[2.0, 3.0, 6.0]])  # magnitude = 7
        times = np.array([0.5])
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        assert result.peak_magnitude >= 0
        assert result.mean_magnitude >= 0
        assert result.variability >= 0
        assert result.peak_lx >= 0
        assert result.peak_ly >= 0
        assert result.peak_lz >= 0

    def test_peak_time_from_times_array(self) -> None:
        am = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        times = np.array([0.1, 0.7])
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        # Peak magnitude is at index 1 (magnitude 5), time should be 0.7
        assert result.peak_time == pytest.approx(0.7)

    def test_peak_time_defaults_zero_when_times_none(self) -> None:
        am = np.array([[0.0, 0.0, 3.0]])
        stub = _Stub(am, times=None)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        assert result.peak_time == pytest.approx(0.0)

    def test_single_frame(self) -> None:
        am = np.array([[1.0, 2.0, 2.0]])  # magnitude = 3
        times = np.array([1.0])
        stub = _Stub(am, times)
        result = stub.compute_angular_momentum_metrics()
        assert result is not None
        assert result.peak_magnitude == pytest.approx(3.0)
        assert result.mean_magnitude == pytest.approx(3.0)
