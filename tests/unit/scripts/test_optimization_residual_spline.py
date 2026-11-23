"""Analytic and contract tests for the Slice 3 residual parameterization."""

import numpy as np
import pytest

from scripts.research.optimization_residual_spline import sample_residual

pytestmark = pytest.mark.unit


def _sample(**overrides):
    inputs = {
        "times": np.linspace(0, 1, 11),
        "breakpoints": np.array([0.0, 1.0]),
        "coefficients": np.zeros((4, 2)),
        "torque_radius": np.array([2.0, 3.0]),
        "minimum_spacing_s": 0.1,
    }
    inputs.update(overrides)
    return sample_residual(**inputs)


def test_single_span_matches_cubic_bernstein_polynomial_and_derivative():
    # A clamped cubic with controls [0, 0, 0, 2] is exactly 2*t**3.
    t = np.linspace(0, 1, 31)
    c = np.zeros((4, 2))
    c[-1] = [2.0, -3.0]
    value, slew = _sample(times=t, coefficients=c)
    np.testing.assert_allclose(value, t[:, None] ** 3 * [2.0, -3.0])
    np.testing.assert_allclose(slew, 3 * t[:, None] ** 2 * [2.0, -3.0])


def test_physical_time_scaling_and_endpoint_values():
    c = np.array([[0.0, 0], [0, 0], [0, 0], [2, -3]])
    value, slew = _sample(
        times=np.array([2.0, 3.0, 4.0]),
        breakpoints=np.array([2.0, 4.0]),
        coefficients=c,
    )
    np.testing.assert_allclose(value, [[0, 0], [0.25, -0.375], [2, -3]])
    np.testing.assert_allclose(slew[-1], [3, -4.5])


def test_multispan_convex_hull_bound_and_zero_radius_joint():
    rng = np.random.default_rng(15493)
    c = rng.uniform(-1, 1, (6, 3)) * [2, 0, 3]
    value, slew = _sample(
        times=np.linspace(0, 1, 1001),
        breakpoints=np.array([0, 0.2, 0.7, 1]),
        coefficients=c,
        torque_radius=np.array([2, 0, 3]),
    )
    assert value.shape == slew.shape == (1001, 3)
    assert np.all(np.abs(value) <= [2, 0, 3])
    assert np.isfinite(slew).all()


def test_analytic_slew_agrees_with_finite_difference_across_interior_knot():
    c = np.array([[1], [-1], [2], [-2], [0.5]])
    t = np.array([0.2, 0.5, 0.8])
    args = {
        "breakpoints": np.array([0, 0.5, 1]),
        "coefficients": c,
        "torque_radius": np.array([2.0]),
    }
    _, slew = _sample(times=t, **args)
    plus, _ = _sample(times=t + 1e-6, **args)
    minus, _ = _sample(times=t - 1e-6, **args)
    np.testing.assert_allclose(slew, (plus - minus) / 2e-6, atol=1e-8)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"breakpoints": [0]}, "breakpoints"),
        ({"breakpoints": [[0, 1]]}, "breakpoints"),
        ({"breakpoints": [0, 0, 1]}, "spacing"),
        ({"breakpoints": [0, 0.01, 1]}, "spacing"),
        ({"breakpoints": [1, 0]}, "spacing"),
        ({"breakpoints": [0, np.inf]}, "finite"),
        ({"minimum_spacing_s": 0}, "minimum_spacing"),
        ({"minimum_spacing_s": np.nan}, "minimum_spacing"),
        ({"torque_radius": [-1, 3]}, "torque_radius"),
        ({"torque_radius": []}, "torque_radius"),
        ({"torque_radius": [[2, 3]]}, "torque_radius"),
        ({"torque_radius": [np.inf, 3]}, "finite"),
        ({"coefficients": np.zeros((5, 2))}, "coefficients"),
        ({"coefficients": np.zeros((4, 1))}, "coefficients"),
        ({"coefficients": np.full((4, 2), np.nan)}, "finite"),
        ({"coefficients": np.full((4, 2), 2.1)}, "trust"),
        ({"coefficients": np.full((4, 2), -2.1)}, "trust"),
        ({"times": []}, "times"),
        ({"times": [[0.5]]}, "times"),
        ({"times": [np.nan]}, "finite"),
        ({"times": [-0.01]}, "support"),
        ({"times": [1.01]}, "support"),
    ],
)
def test_invalid_inputs_are_rejected_without_clipping(overrides, message):
    with pytest.raises(ValueError, match=message):
        _sample(**overrides)


def test_unordered_query_times_preserve_caller_order_and_inputs():
    times = np.array([1, 0, 0.5, 0.5])
    c = np.tile([2, -3], (4, 1))
    original = c.copy()
    value, slew = _sample(times=times, coefficients=c)
    np.testing.assert_allclose(value, original)
    np.testing.assert_allclose(slew, 0)
    np.testing.assert_array_equal(c, original)
    np.testing.assert_array_equal(times, [1, 0, 0.5, 0.5])


def test_nonrepresentable_slew_is_rejected():
    with pytest.raises(ValueError, match="remain finite"):
        _sample(
            coefficients=np.array([[-1e308], [1e308], [-1e308], [1e308]]),
            torque_radius=np.array([1e308]),
        )


def test_exact_minimum_spacing_is_admitted():
    value, slew = _sample(
        times=np.array([0, 0.125, 0.25]),
        breakpoints=np.array([0, 0.125, 0.25]),
        minimum_spacing_s=0.125,
        coefficients=np.zeros((5, 2)),
    )
    np.testing.assert_array_equal(value, np.zeros((3, 2)))
    np.testing.assert_array_equal(slew, np.zeros((3, 2)))
