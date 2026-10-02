"""Independent bounds and differential checks for the Hermite domain."""

import numpy as np
import pytest

from src.shared.python.estimation.hermite_bounds import HermiteBoundsDomain
from src.shared.python.estimation.map_estimator import CubicHermiteSplineTrajectory


def _controls(times, q, v):
    h = np.diff(times)[:, None]
    return np.stack(
        (q[:-1], q[:-1] + h * v[:-1] / 3, q[1:] - h * v[1:] / 3, q[1:]), axis=1
    )


def test_nonuniform_domain_bounds_whole_spline_and_preserves_packing():
    times = np.array([0.0, 0.4, 2.0])
    domain = HermiteBoundsDomain(times, ((-1.0, 2.0), None, (4.0, 4.0)))
    trajectory = CubicHermiteSplineTrajectory(times, 3)
    lower, upper = domain.decision_bounds()
    assert domain.physical_size == trajectory.coefficient_size
    assert domain.decision_size == 12
    for seed in range(15):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-0.9, 0.9, domain.decision_size)
        x[:6:2] = rng.uniform(-1.0, 2.0, 3)
        coefficients = domain.decode(x)
        q, v = trajectory.unpack(coefficients)
        controls = _controls(times, q, v)
        assert np.all(controls[:, :, 0] >= -1.0 - 1e-14)
        assert np.all(controls[:, :, 0] <= 2.0 + 1e-14)
        assert np.array_equal(q[:, 2], np.full(3, 4.0))
        assert np.array_equal(v[:, 2], np.zeros(3))
        evaluated = trajectory.evaluate(coefficients, np.linspace(0, 2, 501)).q
        assert evaluated[:, 0].min() >= -1.0 - 1e-14
        assert evaluated[:, 0].max() <= 2.0 + 1e-14
        np.testing.assert_allclose(domain.encode(coefficients), x, atol=1e-14)
    assert np.isneginf(lower[1]) and np.isposinf(upper[1])


def test_bounded_knots_do_not_make_infeasible_slopes_acceptable():
    domain = HermiteBoundsDomain((0.0, 1.0), ((0.0, 1.0),))
    with pytest.raises(ValueError, match="velocity|control"):
        domain.encode(np.array([0.5, 0.5, 8.0, -8.0]))
    with pytest.raises(ValueError, match="position"):
        domain.encode(np.array([-0.1, 0.5, 0.0, 0.0]))


def test_decoder_jacobian_matches_independent_central_differences():
    domain = HermiteBoundsDomain((0.0, 0.7, 2.0), ((-1.0, 2.0), None))
    x = np.array([-0.3, 0.2, 0.4, 0.5, 1.1, -0.4, 0.3, -0.7, -0.2, 0.6, 0.8, 0.9])
    numerical = np.column_stack(
        [
            (
                domain.decode(x + np.eye(x.size)[i] * 1e-6)
                - domain.decode(x - np.eye(x.size)[i] * 1e-6)
            )
            / 2e-6
            for i in range(x.size)
        ]
    )
    np.testing.assert_allclose(domain.decode_jacobian(x), numerical, atol=2e-9)


def test_tie_uses_symmetric_generalized_derivative_with_distinct_sides():
    domain = HermiteBoundsDomain((0.0, 1.0, 2.0), ((0.0, 1.0),))
    x = np.array([0.2, 0.5, 0.8, 0.1, 0.4, -0.3])
    direction = np.eye(6)[1] * 1e-7
    left = (domain.decode(x) - domain.decode(x - direction)) / 1e-7
    right = (domain.decode(x + direction) - domain.decode(x)) / 1e-7
    symmetric = (left + right) / 2
    np.testing.assert_allclose(domain.decode_jacobian(x)[:, 1], symmetric, atol=2e-8)
    assert abs(left[4] - right[4]) > 1.0


def test_fixed_all_and_collapsed_interior_velocity():
    fixed = HermiteBoundsDomain((0.0, 1.0), ((3.0, 3.0),))
    assert fixed.decision_size == 0
    np.testing.assert_array_equal(fixed.decode(np.array([])), [3.0, 3.0, 0.0, 0.0])
    assert fixed.decode_jacobian(np.array([])).shape == (4, 0)
    assert fixed.encode(np.array([3.0, 3.0, 0.0, 0.0])).size == 0
    with pytest.raises(ValueError, match="fixed"):
        fixed.encode(np.array([3.0, 3.0, 0.1, 0.0]))
    domain = HermiteBoundsDomain((0.0, 1.0, 2.0), ((0.0, 1.0),))
    encoded = domain.encode(np.array([0.2, 0.0, 0.8, 0.0, 0.0, 0.0]))
    assert encoded[4] == 0.0
    assert domain.decode(encoded)[4] == 0.0


def test_inputs_are_copied_and_result_arrays_do_not_mutate_domain():
    times = np.array([0.0, 1.0])
    bounds = [[0.0, 1.0]]
    domain = HermiteBoundsDomain(times, bounds)
    times[1] = 9.0
    bounds[0][1] = 9.0
    assert domain.knot_times == (0.0, 1.0)
    assert domain.coordinate_bounds == ((0.0, 1.0),)
    lower, _ = domain.decision_bounds()
    lower[0] = -9.0
    assert domain.decision_bounds()[0][0] == 0.0


def test_bernstein_polynomial_stationary_extrema_and_c1_velocities():
    times = np.array([0.0, 0.25, 2.5, 3.0])
    domain = HermiteBoundsDomain(times, ((-2.0, 3.0),))
    trajectory = CubicHermiteSplineTrajectory(times, 1)
    x = np.array([-1.8, 2.2, -0.5, 2.8, -1.0, 1.0, -1.0, 1.0])
    coefficients = domain.decode(x)
    q, v = trajectory.unpack(coefficients)
    for controls in _controls(times, q, v)[:, :, 0]:
        b0, b1, b2, b3 = controls
        polynomial = np.array(
            [-b0 + 3 * b1 - 3 * b2 + b3, 3 * b0 - 6 * b1 + 3 * b2, -3 * b0 + 3 * b1, b0]
        )
        roots = np.roots(np.trim_zeros(np.polyder(polynomial), "f"))
        probes = [0.0, 1.0] + [
            float(root.real)
            for root in roots
            if abs(root.imag) < 1e-12 and 0 < root.real < 1
        ]
        values = np.polyval(polynomial, probes)
        assert values.min() >= -2.0 - 1e-12
        assert values.max() <= 3.0 + 1e-12
    np.testing.assert_allclose(
        trajectory.evaluate(coefficients, times).v, v, atol=1e-13
    )


@pytest.mark.parametrize(
    "times,bounds",
    [
        ((-1e308, 1e308), ((0.0, 1.0),)),
        ((0.0, 1.0), ((-1e308, 1e308),)),
    ],
)
def test_unrepresentable_interval_width_is_rejected(times, bounds):
    with pytest.raises(ValueError, match="width|duration"):
        HermiteBoundsDomain(times, bounds)


def test_feasible_large_finite_seed_does_not_overflow_normalization():
    domain = HermiteBoundsDomain((0.0, 3.0), ((0.0, 1e308),))
    physical = np.array([1e308, 0.0, 0.0, 0.0])
    encoded = domain.encode(physical)
    assert np.all(np.isfinite(encoded))
    np.testing.assert_array_equal(domain.decode(encoded), physical)


@pytest.mark.parametrize(
    "times,bounds",
    [
        ((0.0, 0.0), ((0.0, 1.0),)),
        ((0.0, float("inf")), ((0.0, 1.0),)),
        ((0.0, 1.0), ()),
        ((0.0, 1.0), ((1.0, 0.0),)),
        ((0.0, 1.0), ((0.0, float("inf")),)),
        ((0.0, 1.0), ((False, 1.0),)),
        ((False, 1.0), ((0.0, 1.0),)),
        ((0.0, 1.0), ((0.0,),)),
    ],
)
def test_invalid_constructor_contracts(times, bounds):
    with pytest.raises(ValueError):
        HermiteBoundsDomain(times, bounds)


@pytest.mark.parametrize(
    "method,values",
    [
        ("encode", [0.0, 0.0, 0.0]),
        ("encode", [0.0, 0.0, np.nan, 0.0]),
        ("decode", [0.0, 0.0, 2.0, 0.0]),
        ("decode", [-0.1, 0.0, 0.0, 0.0]),
        ("decode_jacobian", [0.0, 0.0, np.inf, 0.0]),
    ],
)
def test_invalid_vectors_fail(method, values):
    domain = HermiteBoundsDomain((0.0, 1.0), ((0.0, 1.0),))
    with pytest.raises(ValueError):
        getattr(domain, method)(np.asarray(values))
