"""Analytic control penalties must preserve marker and shooting derivatives."""

import numpy as np
import pytest

from src.shared.python.motion_matching.multi_shooting_fit import (
    MultipleShootingOptions,
    fit_multiple_shooting,
)
from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    PrefixFitOptions,
    fit_prefixes,
)

pytestmark = pytest.mark.unit


def _target(time: np.ndarray) -> MarkerTarget:
    points = np.zeros((len(time), 1, 3))
    points[:, 0, 0] = 2 * time
    return MarkerTarget(time, points, np.ones(1))


def _forward(p: np.ndarray, time: np.ndarray) -> np.ndarray:
    points = np.zeros((len(time), 1, 3))
    points[:, 0, 0] = p[0] * time
    return points


def _marker_jacobian(p: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.zeros((len(time), 1, 3, 1))
    result[:, 0, 0, 0] = time
    return result


def _prefix(derivative, residual=lambda p: 2 * p):
    return fit_prefixes(
        _target(np.array([0.0, 1.0])),
        _forward,
        initial=np.array([1.0]),
        lower=np.array([-5.0]),
        upper=np.array([5.0]),
        prefix_end_s=[1.0],
        acceptance_rmse_m=10,
        options=PrefixFitOptions(
            marker_jacobian=_marker_jacobian,
            regularization=residual,
            regularization_jacobian=derivative,
        ),
    )


def test_prefix_analytic_penalty_has_known_optimum():
    # (p-2)^2 + (2p)^2 has its unique minimum at p=0.4.
    result = _prefix(lambda p: np.array([[2.0]]))
    assert result.parameters[0] == pytest.approx(0.4, abs=1e-7)


@pytest.mark.parametrize(
    "derivative", [lambda p: np.zeros((2, 1)), lambda p: np.array([[np.nan]])]
)
def test_prefix_rejects_invalid_penalty_derivative(derivative):
    with pytest.raises(ValueError, match="regularization"):
        _prefix(derivative)


def test_prefix_requires_derivative_for_analytic_penalty():
    with pytest.raises(ValueError, match="regularization"):
        _prefix(None)


def test_prefix_rejects_nonfinite_penalty_even_with_derivative():
    with pytest.raises(ValueError, match="regularization"):
        _prefix(lambda p: np.ones((1, 1)), lambda p: np.array([np.inf]))


def _segmented(p, time, state):
    start = 0.0 if state is None else state[0]
    points = np.zeros((len(time), 1, 3))
    points[:, 0, 0] = start + p[0] * (time - time[0])
    return points, np.array([points[-1, 0, 0]])


def _window_jacobian(p, time, state):
    size = 1 if state is None else 2
    markers = np.zeros((len(time), 1, 3, size))
    markers[:, 0, 0, 0] = time - time[0]
    if state is not None:
        markers[:, 0, 0, 1] = 1
    return markers, markers[-1, 0, :1].copy()


def test_constrained_shooting_penalty_excludes_node_columns():
    # Exact continuity forces z=p/2. The existing fitter includes t=.5 in
    # both windows, giving 1.5(p-2)^2 + 4p^2: p=6/11 and z=3/11.
    result = fit_multiple_shooting(
        _target(np.array([0.0, 0.5, 1.0])),
        _segmented,
        _forward,
        initial_theta=np.array([1.0]),
        lower_theta=np.array([-5.0]),
        upper_theta=np.array([5.0]),
        initial_states={0.5: np.array([0.5])},
        state_bounds={0.5: (np.array([-5.0]), np.array([5.0]))},
        options=MultipleShootingOptions(
            shooting_nodes=(0.5, 1.0),
            solver="slsqp",
            window_jacobian=_window_jacobian,
            regularization=lambda p: 2 * p,
            regularization_jacobian=lambda p: np.array([[2.0]]),
            max_iterations=50,
            max_nfev=100,
        ),
    )
    assert result.theta[0] == pytest.approx(6 / 11, abs=1e-6)
    assert result.intermediate_states[0.5][0] == pytest.approx(3 / 11, abs=1e-6)
    assert result.max_defect_norm < 1e-7
