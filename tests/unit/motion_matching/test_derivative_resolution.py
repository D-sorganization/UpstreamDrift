"""TDD unit tests for numerical derivative resolution and measured derivative floors.

Validates:
1. Measurement of derivative floors on analytic test functions (linear, quadratic, sinusoidal).
2. Detection of truncation vs. roundoff noise floors.
3. Optimal finite difference step selection.
4. Scale-aware step vector generation.
5. DbC contracts: strictly positive steps, dimension bounds, finite values.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.motion_matching.derivative_resolution import (
    DerivativeResolutionResult,
    compute_finite_difference_step_vector,
    measure_derivative_floor,
)

pytestmark = pytest.mark.unit


def test_measure_derivative_floor_quadratic() -> None:
    """f(x) = x^2, df/dx at x=2 is 4.0.

    Check that measure_derivative_floor finds the optimal step and accurate slope.
    """

    def f(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2])

    x0 = np.array([2.0])
    result = measure_derivative_floor(f, x0, component_idx=0)

    assert isinstance(result, DerivativeResolutionResult)
    assert result.is_resolved
    assert result.optimal_step > 0.0
    best_slope = result.measured_slopes[int(np.argmin(result.relative_errors))]
    assert math.isclose(best_slope, 4.0, rel_tol=1e-4)
    assert result.noise_floor >= 0.0


def test_measure_derivative_floor_sinusoid() -> None:
    """f(x) = sin(x), df/dx at x=0 is cos(0) = 1.0."""

    def f(x: np.ndarray) -> np.ndarray:
        return np.array([np.sin(x[0])])

    x0 = np.array([0.0])
    steps = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    result = measure_derivative_floor(f, x0, steps=steps, component_idx=0)

    assert result.is_resolved
    best_idx = int(np.argmin(result.relative_errors))
    assert math.isclose(result.measured_slopes[best_idx], 1.0, rel_tol=1e-4)


def test_measure_derivative_floor_validation() -> None:
    """DbC validation for invalid inputs."""

    def f(x: np.ndarray) -> np.ndarray:
        return x

    # Non-finite x0
    with pytest.raises(ValueError, match="finite"):
        measure_derivative_floor(f, np.array([np.nan]))

    # Invalid component_idx
    with pytest.raises(ValueError, match="component_idx"):
        measure_derivative_floor(f, np.array([1.0]), component_idx=5)

    # Invalid steps (negative or not strictly decreasing)
    with pytest.raises(ValueError, match="steps"):
        measure_derivative_floor(f, np.array([1.0]), steps=[1e-4, 1e-2])


def test_compute_finite_difference_step_vector() -> None:
    """Scale-aware step vector generation."""
    x0 = np.array([0.0, 100.0, -50.0])
    h = compute_finite_difference_step_vector(x0, default_step=1e-4, floor=1e-8)

    assert len(h) == 3
    assert h[0] == pytest.approx(1e-4 * (1.0 + 0.0))
    assert h[1] == pytest.approx(1e-4 * (1.0 + 100.0))
    assert h[2] == pytest.approx(1e-4 * (1.0 + 50.0))
    assert np.all(h >= 1e-8)
