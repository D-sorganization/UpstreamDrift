"""Adaptive physical error must not be diluted by sensitivity padding."""

import numpy as np
import pytest
from src.shared.python.motion_matching.continuous_forward import integrate_forward
from src.shared.python.motion_matching.forward_sensitivity import (
    integrate_sensitivities,
)

pytestmark = pytest.mark.unit


def test_zero_sensitivity_padding_preserves_physical_error() -> None:
    clock = np.linspace(0, 1, 1001)
    reference = integrate_forward(
        np.zeros(1),
        clock,
        lambda t, x: np.array([200 * np.cos(200 * t)]),
        rtol=1e-7,
        atol=1e-9,
        max_step=0.1,
    )
    errors = []
    for count in (1, 27, 81):
        result = integrate_sensitivities(
            np.zeros(1),
            clock,
            lambda t, x, count=count: (
                np.array([200 * np.cos(200 * t)]),
                np.zeros((1, 1)),
                np.zeros((1, count)),
            ),
            count,
            separate_error_control=True,
            rtol=1e-7,
            atol=1e-9,
            max_step=0.1,
        )
        np.testing.assert_allclose(
            result.integration.state, reference.state, rtol=0, atol=1e-10
        )
        np.testing.assert_array_equal(result.state_parameter_jacobian, 0)
        errors.append(
            np.max(np.abs(result.integration.state[:, 0] - np.sin(200 * clock)))
        )
    assert max(errors) < 1e-7
    assert max(errors) - min(errors) < 1e-10


def test_sensitivity_error_also_controls_adaptive_steps() -> None:
    clock = np.linspace(0, 1, 1001)
    result = integrate_sensitivities(
        np.zeros(1),
        clock,
        lambda t, x: (
            np.zeros(1),
            np.zeros((1, 1)),
            np.array([[200 * np.cos(200 * t)]]),
        ),
        1,
        separate_error_control=True,
        rtol=1e-7,
        atol=1e-9,
        max_step=0.1,
    )
    np.testing.assert_allclose(
        result.state_parameter_jacobian[:, 0, 0], np.sin(200 * clock), rtol=0, atol=1e-7
    )


def test_grouped_exponential_sensitivity_and_initial_derivative() -> None:
    clock = np.linspace(0, 1, 31)
    result = integrate_sensitivities(
        np.ones(1),
        clock,
        lambda t, x: (2 * x, np.array([[2.0]]), x[:, None]),
        1,
        initial_sensitivity=np.array([[3.0]]),
        separate_error_control=True,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.3,
    )
    np.testing.assert_allclose(
        result.integration.state[:, 0], np.exp(2 * clock), rtol=1e-9
    )
    np.testing.assert_allclose(
        result.state_parameter_jacobian[:, 0, 0],
        np.exp(2 * clock) * (3 + clock),
        rtol=1e-9,
    )


def test_grouped_constant_force_sensitivity() -> None:
    clock = np.array([0.0, 0.3, 1.0])
    result = integrate_sensitivities(
        np.zeros(2),
        clock,
        lambda t, x: (
            np.array([x[1], 2.0]),
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.array([[0.0], [1.0]]),
        ),
        1,
        separate_error_control=True,
        max_step=0.3,
    )
    np.testing.assert_allclose(result.integration.state[:, 0], clock**2, atol=1e-12)
    np.testing.assert_allclose(
        result.state_parameter_jacobian[:, 0, 0], clock**2 / 2, atol=1e-12
    )
    np.testing.assert_allclose(
        result.state_parameter_jacobian[:, 1, 0], clock, atol=1e-12
    )


@pytest.mark.parametrize(
    "blocks", [(), (0, 2), (-1, 3), (True, 1), (1.5, 0.5), (1,), [1, 1]]
)
def test_invalid_blocks_fail_before_callbacks(blocks: object) -> None:
    calls = []
    with pytest.raises(ValueError, match="error_block_sizes"):
        integrate_forward(
            np.zeros(2),
            np.array([0.0, 1.0]),
            lambda t, x: calls.append(t),
            error_block_sizes=blocks,
        )
    assert calls == []


@pytest.mark.parametrize("flag", [1, "yes", None])
def test_separate_control_flag_requires_bool(flag: object) -> None:
    with pytest.raises(ValueError, match="separate_error_control"):
        integrate_sensitivities(
            np.zeros(1),
            np.array([0.0, 1.0]),
            lambda t, x: None,
            1,
            separate_error_control=flag,
        )


def test_grouped_budget_stops_before_extra_callback() -> None:
    calls = []

    def linearize(t: float, x: np.ndarray) -> tuple:
        calls.append(t)
        return x, np.eye(1), np.zeros((1, 27))

    with pytest.raises(RuntimeError, match="budget"):
        integrate_sensitivities(
            np.ones(1),
            np.array([0.0, 1.0]),
            linearize,
            27,
            separate_error_control=True,
            max_evaluations=3,
        )
    assert len(calls) == 3


def test_grouped_norm_reuses_scipy_norm_for_each_block() -> None:
    from scipy.integrate import DOP853
    from src.shared.python.motion_matching.grouped_dop853 import grouped_dop853

    solver = grouped_dop853((1, 2), 3)(lambda t, x: np.zeros(3), 0.0, np.ones(3), 1.0)
    stages = np.random.default_rng(7).normal(size=(solver.n_stages + 1, 3))
    scale = np.array([0.1, 0.2, 0.3])
    expected = max(
        DOP853._estimate_error_norm(solver, stages[:, :1], 0.1, scale[:1]),
        DOP853._estimate_error_norm(solver, stages[:, 1:], 0.1, scale[1:]),
    )
    assert solver._estimate_error_norm(stages, 0.1, scale) == expected


def test_false_option_preserves_default_results() -> None:
    def linearize(t: float, x: np.ndarray) -> tuple:
        return x, np.ones((1, 1)), np.ones((1, 3))

    args = (np.ones(1), np.array([0.0, 0.1]), linearize, 3)
    default = integrate_sensitivities(*args, max_step=0.03)
    explicit = integrate_sensitivities(
        *args, max_step=0.03, separate_error_control=False
    )
    np.testing.assert_array_equal(default.integration.state, explicit.integration.state)
    np.testing.assert_array_equal(
        default.state_parameter_jacobian, explicit.state_parameter_jacobian
    )
    assert default.integration.evaluations == explicit.integration.evaluations
