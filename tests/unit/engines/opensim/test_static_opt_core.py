"""Shared static-optimisation numerics (SPN-10b, #12761).

The leg solver (``musculoskeletal_static_opt.solve_frame``) moved into
``static_opt_core``; the golden values below were recorded from the leg
implementation BEFORE the move and must not change.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import musculoskeletal_static_opt as leg
from src.engines.physics_engines.opensim.python.static_opt_core import (
    SUPPORTED_EXPONENTS,
    ActivationSolution,
    activation_cost,
    participation_ratio,
    solve_activations,
    solve_frame,
)

pytestmark = pytest.mark.unit

_ACTIVE = np.array([800.0, 1200.0, 500.0, 950.0, 300.0])
_PASSIVE = np.array([10.0, 0.0, 25.0, 5.0, 0.0])
_MOMENT = np.array(
    [
        [0.04, -0.01, 0.0],
        [0.03, 0.02, -0.01],
        [-0.05, 0.0, 0.02],
        [0.0, 0.03, 0.01],
        [0.01, -0.02, 0.03],
    ]
)


# ------------------------------------------------------------ leg golden


@pytest.mark.parametrize(
    ("tau", "weight", "activation", "reserve", "cost"),
    [
        (
            [30.0, 10.0, 5.0],
            1.0,
            [0.63450204033691, 0.2578512566469009, 0.0, 0.39731616465452163]
            + [0.4144316474535459],
            [0.019994527569810572, 0.0006653552370678995, 0.03982668846294235],
            0.8006802237971825,
        ),
        (
            [300.0, -80.0, 40.0],
            2.5,
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [229.85000000000002, -90.05, 42.45],
            156858.06875000003,
        ),
    ],
)
def test_leg_solve_frame_golden_unchanged(tau, weight, activation, reserve, cost):
    for solver in (leg.solve_frame, solve_frame):
        sol = solver(_ACTIVE, _PASSIVE, _MOMENT, np.array(tau), reserve_weight=weight)
        np.testing.assert_allclose(sol.activation, activation, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sol.reserve, reserve, rtol=1e-13, atol=1e-12)
        assert sol.cost == pytest.approx(cost, rel=1e-13)
        assert sol.success


def test_leg_module_reexports_the_shared_core() -> None:
    assert leg.solve_frame is solve_frame
    assert leg.FrameSolution is solve_frame.__globals__["FrameSolution"]


# ------------------------------------------------------- cost and gradient


@pytest.mark.parametrize("exponent", SUPPORTED_EXPONENTS)
def test_activation_cost_gradient_matches_finite_differences(exponent) -> None:
    rng = np.random.default_rng(7)
    a = rng.uniform(0.05, 0.95, 9)
    value, grad = activation_cost(a, exponent)
    assert value == pytest.approx(float(np.sum(a**exponent)), rel=1e-14)
    step = 1e-6
    fd = np.array(
        [
            (activation_cost(a + step * e, exponent)[0])
            - activation_cost(a - step * e, exponent)[0]
            for e in np.eye(a.size)
        ]
    ) / (2 * step)
    np.testing.assert_allclose(grad, fd, rtol=1e-7, atol=1e-9)


def test_activation_cost_rejects_unsupported_exponent() -> None:
    with pytest.raises(ValueError, match="exponent"):
        activation_cost(np.ones(3), 4)
    with pytest.raises(TypeError, match="exponent"):
        activation_cost(np.ones(3), 2.0)  # type: ignore[arg-type]


def test_participation_ratio_bounds() -> None:
    assert participation_ratio(np.ones(6)) == pytest.approx(6.0)
    assert participation_ratio(np.array([0.0, 0.7, 0.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-zero"):
        participation_ratio(np.zeros(3))


# ---------------------------------------------- exact (lexicographic) solve


def _redundant_problem() -> tuple[np.ndarray, ...]:
    """Six muscles, two DOFs: a reachable target from a known activation."""
    active = np.array([900.0, 900.0, 700.0, 700.0, 400.0, 400.0])
    passive = np.array([5.0, 5.0, 2.0, 2.0, 0.0, 0.0])
    moment = np.array(
        [
            [0.05, 0.03],
            [0.05, -0.03],
            [0.04, 0.05],
            [0.04, -0.05],
            [0.02, 0.01],
            [0.02, -0.01],
        ]
    )
    a_true = np.array([0.3, 0.2, 0.25, 0.1, 0.4, 0.35])
    tau = moment.T @ (passive + a_true * active)
    return active, passive, moment, tau


@pytest.mark.parametrize("exponent", SUPPORTED_EXPONENTS)
def test_reachable_target_is_met_with_zero_reserve(exponent) -> None:
    active, passive, moment, tau = _redundant_problem()
    sol = solve_activations(active, passive, moment, tau, exponent=exponent)
    assert isinstance(sol, ActivationSolution)
    assert sol.success
    assert np.all(sol.activation >= 0.0) and np.all(sol.activation <= 1.0)
    np.testing.assert_allclose(sol.muscle_moment, tau, atol=1e-6)
    np.testing.assert_allclose(sol.reserve, 0.0, atol=1e-6)
    assert sol.cost == pytest.approx(float(np.sum(sol.activation**exponent)))


def test_p2_matches_the_closed_form_minimum_norm_activation() -> None:
    active, passive, moment, tau = _redundant_problem()
    gain = moment.T * active[None, :]
    expected = np.linalg.pinv(gain) @ (tau - moment.T @ passive)
    assert np.all((expected > 0) & (expected < 1))
    sol = solve_activations(active, passive, moment, tau, exponent=2)
    np.testing.assert_allclose(sol.activation, expected, atol=1e-8)


def test_p3_spreads_load_more_than_p2() -> None:
    active, passive, moment, tau = _redundant_problem()
    two = solve_activations(active, passive, moment, tau, exponent=2)
    three = solve_activations(active, passive, moment, tau, exponent=3)
    assert participation_ratio(three.activation) > participation_ratio(two.activation)
    assert float(np.max(three.activation)) < float(np.max(two.activation))


def test_unreachable_target_puts_the_weighted_remainder_on_reserves() -> None:
    active, passive, moment, _ = _redundant_problem()
    tau = np.array([500.0, 3.0])  # flexion far beyond capacity
    scale = np.array([2.0, 0.5])
    sol = solve_activations(
        active, passive, moment, tau, exponent=3, reserve_scale=scale
    )
    np.testing.assert_allclose(sol.muscle_moment + sol.reserve, tau, atol=1e-9)
    # KKT of min ||(tau - M(a)) / scale||^2 on the box: muscles that could
    # still reduce the weighted remainder are saturated.
    gain = moment.T * active[None, :]
    slope = gain.T @ (sol.reserve / scale**2)
    for a, g in zip(sol.activation, slope, strict=True):
        if a < 1.0 - 1e-9:
            assert g <= 1e-6
        if a > 1e-9:
            assert g >= -1e-6
    assert sol.reserve[0] > 0.0


def test_solve_activations_validates_inputs() -> None:
    active, passive, moment, tau = _redundant_problem()
    with pytest.raises(ValueError, match="exponent"):
        solve_activations(active, passive, moment, tau, exponent=5)
    with pytest.raises(ValueError, match="tau"):
        solve_activations(active, passive, moment, tau[:1])
    with pytest.raises(ValueError, match="reserve_scale"):
        solve_activations(
            active, passive, moment, tau, reserve_scale=np.array([1.0, 0.0])
        )
    with pytest.raises(ValueError, match="finite"):
        solve_activations(active, passive, moment, np.array([np.nan, 0.0]))
    with pytest.raises(TypeError, match="ndarray"):
        solve_activations(list(active), passive, moment, tau)  # type: ignore[arg-type]
