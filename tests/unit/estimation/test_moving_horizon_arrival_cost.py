"""Behavioural tests: the MHE applies and propagates its arrival factor (#11545).

Analytic truth is a linear-Gaussian least-squares problem on the same cubic
Hermite spline the estimator uses. Each window has ``2 * N`` knot unknowns
``c = [q_0..q_{N-1}, v_0..v_{N-1}]``. Per sample ``k`` the residual rows are

* position observation ``(q(t_k) - y_k) / sigma_y``,
* velocity observation ``(v(t_k) - w_k) / sigma_v`` and
* white-noise-acceleration process model ``a(t_k) / sigma_a``,

and the Gaussian prior on the first knot state ``x_0 = (q_0, v_0)`` with mean
``m`` and information ``P^{-1}`` is the arrival cost ``|R (x_0 - m)|^2`` with
``R^T R = P^{-1}``. Every row is linear in ``c``, so the MAP estimate is the
solution of one linear least-squares problem, which is the Kalman/RTS
estimate of the same linear-Gaussian model. A moving-horizon estimator whose
arrival factor is applied and propagated by exact marginalisation must
reproduce, for every window, the batch estimate over all samples seen so far
restricted to that window's knots.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation import (
    MapEstimatorOptions,
    MovingHorizonEstimator,
    MovingHorizonOptions,
    MovingHorizonProblem,
)
from src.shared.python.estimation.map_estimator import CubicHermiteSplineTrajectory
from src.shared.python.estimation.moving_horizon import ArrivalFactor

pytestmark = pytest.mark.unit

SIGMA_Y = 0.05
SIGMA_V = 0.2
SIGMA_A = 2.0
DT = 0.1
PRIOR_MEAN = np.array([0.5, -3.0])
PRIOR_INFORMATION = np.diag([1.0 / 0.01**2, 1.0 / 0.1**2])
# Linear problem solved by least_squares to xtol/ftol/gtol = 1e-10; the batch
# reference is a direct lstsq. Agreement is limited only by solver round-off.
ATOL = 1e-7


def _samples(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return times and (n, 2) noisy position/velocity observations."""
    rng = np.random.default_rng(11545)
    times = DT * np.arange(n, dtype=float)
    y = np.sin(3.0 * times) + SIGMA_Y * rng.standard_normal(n)
    w = 3.0 * np.cos(3.0 * times) + SIGMA_V * rng.standard_normal(n)
    return times, np.column_stack([y, w])


def _prior() -> ArrivalFactor:
    sqrt_info = np.linalg.cholesky(PRIOR_INFORMATION).T
    return ArrivalFactor(
        reference_state=PRIOR_MEAN,
        sqrt_information=sqrt_info,
        residual_offset=np.zeros(2),
        rank=2,
    )


def _problem(
    obs_lookup: dict[float, np.ndarray],
    window_size: int,
    arrival: ArrivalFactor | None,
) -> MovingHorizonProblem:
    def residual(evaluation, _parameters):
        obs = np.array([obs_lookup[round(float(t), 9)] for t in evaluation.times])
        return np.concatenate(
            [
                (evaluation.q[:, 0] - obs[:, 0]) / SIGMA_Y,
                (evaluation.v[:, 0] - obs[:, 1]) / SIGMA_V,
                evaluation.a[:, 0] / SIGMA_A,
            ]
        )

    def jacobian(evaluation, _parameters, layout):
        rows = [
            evaluation.q_basis[:, 0, :] / SIGMA_Y,
            evaluation.v_basis[:, 0, :] / SIGMA_V,
            evaluation.a_basis[:, 0, :] / SIGMA_A,
        ]
        jac = np.zeros((3 * evaluation.times.size, layout.size))
        jac[:, : layout.trajectory_size] = np.vstack(rows)
        return jac

    return MovingHorizonProblem(
        n_dof=1,
        fixed_parameters={},
        residual=residual,
        jacobian=jacobian,
        options=MovingHorizonOptions(
            window_size=window_size,
            step_size=1,
            latency_budget_ms=1e4,
            solver_options=MapEstimatorOptions(max_iterations=50),
        ),
        arrival_factor=arrival,
    )


def _batch_solution(
    times: np.ndarray, obs: np.ndarray, *, with_prior: bool = True
) -> np.ndarray:
    """Analytic linear least-squares MAP over every sample (the batch truth)."""
    spline = CubicHermiteSplineTrajectory(times, 1)
    basis = spline.evaluate(np.zeros(spline.coefficient_size), times)
    blocks = [
        basis.q_basis[:, 0, :] / SIGMA_Y,
        basis.v_basis[:, 0, :] / SIGMA_V,
        basis.a_basis[:, 0, :] / SIGMA_A,
    ]
    targets = [obs[:, 0] / SIGMA_Y, obs[:, 1] / SIGMA_V, np.zeros(times.size)]
    if with_prior:
        sqrt_info = np.linalg.cholesky(PRIOR_INFORMATION).T
        first_state = np.vstack([basis.q_basis[0], basis.v_basis[0]])
        blocks.append(sqrt_info @ first_state)
        targets.append(sqrt_info @ PRIOR_MEAN)
    solution, *_ = np.linalg.lstsq(np.vstack(blocks), np.concatenate(targets))
    return solution


def _knot_states(coefficients: np.ndarray, n_knots: int) -> np.ndarray:
    """Return (n_knots, 2) knot states (q, v) from 1-DOF spline coefficients."""
    return np.column_stack([coefficients[:n_knots], coefficients[n_knots:]])


def _lookup(times: np.ndarray, obs: np.ndarray) -> dict[float, np.ndarray]:
    return {round(float(t), 9): row for t, row in zip(times, obs, strict=True)}


def test_single_window_applies_arrival_prior_like_batch_least_squares() -> None:
    """One window with a prior equals the analytic prior-weighted LS solution."""
    times, y = _samples(4)
    estimator = MovingHorizonEstimator(_problem(_lookup(times, y), 4, _prior()))
    estimator.append_samples(times, y[:, :1])

    result = estimator.solve_next()

    assert result is not None and result.success
    with_prior = _batch_solution(times, y)
    without_prior = _batch_solution(times, y, with_prior=False)
    # The prior must visibly move the answer, or this test proves nothing.
    assert np.max(np.abs(with_prior - without_prior)) > 1e-2
    np.testing.assert_allclose(result.coefficients, with_prior, atol=ATOL)
    np.testing.assert_allclose(
        result.objective,
        0.5 * float(np.sum(result.residual**2)),
        rtol=1e-12,
    )


def test_sliding_windows_reproduce_batch_kalman_estimate() -> None:
    """Every window equals the batch MAP over all samples seen so far."""
    n_samples, window = 10, 4
    times, y = _samples(n_samples)
    estimator = MovingHorizonEstimator(_problem(_lookup(times, y), window, _prior()))
    estimator.append_samples(times[:window], y[:window, :1])

    for stop in range(window, n_samples + 1):
        if stop > window:
            estimator.append_samples(times[stop - 1 : stop], y[stop - 1 : stop, :1])
        result = estimator.solve_next()
        assert result is not None and result.success

        batch = _knot_states(_batch_solution(times[:stop], y[:stop]), stop)
        mhe = _knot_states(result.coefficients, window)
        # Filtering estimate (last knot) is the Kalman filter estimate; the
        # rest of the window are fixed-lag smoothed states.
        np.testing.assert_allclose(mhe, batch[stop - window : stop], atol=ATOL)

    assert estimator.arrival_factor is not None
    assert estimator.arrival_sample_index == n_samples - window
    for index in range(n_samples - window):
        assert estimator.accumulation_guard.is_marginalized(index)


def test_window_without_arrival_still_marginalises_dropped_samples() -> None:
    """With no user prior, dropped samples still inform later windows."""
    n_samples, window = 8, 4
    times, y = _samples(n_samples)
    estimator = MovingHorizonEstimator(_problem(_lookup(times, y), window, None))
    estimator.append_samples(times[:window], y[:window, :1])
    estimator.solve_next()
    for stop in range(window + 1, n_samples + 1):
        estimator.append_samples(times[stop - 1 : stop], y[stop - 1 : stop, :1])
        result = estimator.solve_next()
        assert result is not None and result.success

    batch = _knot_states(_batch_solution(times, y, with_prior=False), n_samples)
    np.testing.assert_allclose(
        _knot_states(result.coefficients, window), batch[-window:], atol=ATOL
    )


def test_from_gaussian_prior_builds_square_root_information() -> None:
    information = np.array([[4.0, 1.0], [1.0, 3.0]])
    factor = ArrivalFactor.from_gaussian_prior(PRIOR_MEAN, information)

    np.testing.assert_allclose(
        factor.sqrt_information.T @ factor.sqrt_information, information, atol=1e-12
    )
    assert factor.rank == 2
    state = PRIOR_MEAN + np.array([0.3, -0.2])
    delta = state - PRIOR_MEAN
    np.testing.assert_allclose(
        factor.evaluate_cost(state), 0.5 * delta @ information @ delta, rtol=1e-12
    )


@pytest.mark.parametrize(
    "information",
    [
        np.array([[1.0, 0.0], [0.0, 0.0]]),  # singular
        np.array([[1.0, 2.0], [2.0, 1.0]]),  # indefinite
        np.array([[1.0, 0.5], [0.0, 1.0]]),  # asymmetric
        np.eye(3),  # wrong shape
    ],
)
def test_from_gaussian_prior_rejects_invalid_information(
    information: np.ndarray,
) -> None:
    with pytest.raises(PreconditionError):
        ArrivalFactor.from_gaussian_prior(PRIOR_MEAN, information)


def test_problem_rejects_arrival_of_wrong_state_dimension() -> None:
    wrong = ArrivalFactor.from_gaussian_prior(np.zeros(3), np.eye(3))
    with pytest.raises(PreconditionError, match="2 \\* n_dof"):
        _problem({}, 4, wrong)


def test_arrival_is_reset_when_window_skips_past_unsolved_samples(caplog) -> None:
    """Samples never seen by a solve cannot be marginalised: fail visibly."""
    times, y = _samples(12)
    estimator = MovingHorizonEstimator(_problem(_lookup(times, y), 4, _prior()))
    estimator.append_samples(times[:4], y[:4, :1])
    estimator.solve_next()
    estimator.append_samples(times[4:12], y[4:12, :1])

    with caplog.at_level("WARNING"):
        result = estimator.solve_next()

    assert result is not None
    assert result.arrival_factor is None
    assert estimator.arrival_factor is None
    assert "arrival factor discarded" in caplog.text
