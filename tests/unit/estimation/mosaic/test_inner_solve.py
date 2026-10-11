"""Tests for the variable-projection inner solve (parameters, inputs, template)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.inner_solve import (
    InnerProblem,
    LinearAnchor,
    reduced_gauss_newton,
    solve_inner,
)
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

pytestmark = pytest.mark.unit


def _underactuated_chain() -> tuple[PlanarChain, np.ndarray]:
    chain = PlanarChain(
        link_lengths=np.array([0.9, 0.6]), actuated=np.array([False, True])
    )
    pi = np.array([1.3, 0.585, 0.0, 0.343, 0.7, 0.175, 0.0, 0.064])
    return chain, pi


def _true_nodes(
    chain: PlanarChain, pi: np.ndarray, n_trials: int, n_nodes: int, seed: int
):
    rng = np.random.default_rng(seed)
    dt = 0.01
    times = np.arange(n_nodes) * dt
    amplitude = rng.uniform(0.5, 2.0, size=n_trials)
    u = (amplitude[:, None] * np.sin(2 * np.pi * 1.5 * times)[None, :])[..., None]
    x0 = np.tile(np.array([0.4, 0.3, 0.0, 0.0]), (n_trials, 1))
    q, v = chain.rollout(x0, u[:, :-1], pi, dt)
    q_flat, v_flat, u_flat = q.reshape(-1, 2), v.reshape(-1, 2), u.reshape(-1, 1)
    a_flat = chain.forward_dynamics(q_flat, v_flat, u_flat, pi)
    trial_index = np.repeat(np.arange(n_trials), n_nodes)
    phase_index = np.tile(np.arange(n_nodes), n_trials)
    return chain.regressor(q_flat, v_flat, a_flat), u_flat, trial_index, phase_index


def _problem(
    chain, pi_prior, pi_prior_weight, regressors, trial_index, phase_index, **kw
):
    defaults = {
        "input_effort_weight": 1e-6,
        "input_smoothness_weight": 0.0,
        "template_weight": 0.0,
        "anchors": (),
    }
    defaults.update(kw)
    return InnerProblem(
        regressors=regressors,
        input_matrix=chain.input_matrix,
        trial_index=trial_index,
        phase_index=phase_index,
        dynamics_weight=np.full(chain.n_links, 10.0),
        parameter_prior_mean=pi_prior,
        parameter_prior_weight=pi_prior_weight,
        **defaults,
    )


def test_locked_parameters_recover_inputs_exactly() -> None:
    chain, pi = _underactuated_chain()
    regressors, u_true, trials, phases = _true_nodes(chain, pi, 2, 30, seed=3)
    problem = _problem(chain, pi, np.full(pi.size, 1e8), regressors, trials, phases)
    solution = solve_inner(problem)
    np.testing.assert_allclose(solution.parameters, pi, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(solution.inputs, u_true, atol=1e-6)
    assert solution.cost < 1e-8


def test_solution_matches_dense_joint_least_squares() -> None:
    chain, pi = _underactuated_chain()
    regressors, _, trials, phases = _true_nodes(chain, pi, 1, 6, seed=5)
    prior_mean = pi * 1.1
    prior_weight = np.full(pi.size, 0.3)
    problem = _problem(
        chain,
        prior_mean,
        prior_weight,
        regressors,
        trials,
        phases,
        input_effort_weight=0.05,
        input_smoothness_weight=0.2,
        template_weight=0.7,
        anchors=(LinearAnchor(np.array([1, 0, 0, 0, 1, 0, 0, 0.0]), 2.0, 4.0),),
    )
    solution = solve_inner(problem)
    design = solution.design.toarray()
    dense_solution, *_ = np.linalg.lstsq(design, solution.rhs, rcond=None)
    np.testing.assert_allclose(solution.stacked, dense_solution, atol=1e-7)
    np.testing.assert_allclose(
        solution.residual, design @ dense_solution - solution.rhs, atol=1e-9
    )


def test_template_coupling_drives_trials_to_shared_pattern() -> None:
    chain, pi = _underactuated_chain()
    regressors, u_true, trials, phases = _true_nodes(chain, pi, 3, 20, seed=7)
    locked = np.full(pi.size, 1e8)
    free = solve_inner(_problem(chain, pi, locked, regressors, trials, phases))
    assert np.ptp(free.inputs.reshape(3, 20, 1), axis=0).max() > 0.1
    coupled = solve_inner(
        _problem(chain, pi, locked, regressors, trials, phases, template_weight=1e4)
    )
    assert coupled.template is not None and coupled.template.shape == (20, 1)
    spread = np.ptp(coupled.inputs.reshape(3, 20, 1), axis=0).max()
    assert spread < 1e-2
    np.testing.assert_allclose(
        coupled.template[:, 0], u_true.reshape(3, 20).mean(axis=0), atol=5e-2
    )


def test_reduced_gauss_newton_matches_explicit_projector() -> None:
    chain, pi = _underactuated_chain()
    regressors, _, trials, phases = _true_nodes(chain, pi, 1, 8, seed=11)
    problem = _problem(
        chain,
        pi * 0.9,
        np.full(pi.size, 0.5),
        regressors,
        trials,
        phases,
        input_effort_weight=0.1,
        input_smoothness_weight=0.3,
    )
    solution = solve_inner(problem)
    rng = np.random.default_rng(2)
    n_dyn = regressors.shape[0] * regressors.shape[1]
    jacobian_xi = rng.normal(size=(n_dyn, 5))
    hessian, gradient = reduced_gauss_newton(solution, jacobian_xi)
    design = solution.design.toarray()
    padded = np.zeros((design.shape[0], 5))
    padded[:n_dyn] = jacobian_xi
    projector = np.eye(design.shape[0]) - design @ np.linalg.pinv(design)
    np.testing.assert_allclose(hessian, padded.T @ projector @ padded, atol=1e-8)
    np.testing.assert_allclose(gradient, padded.T @ solution.residual, atol=1e-9)


def test_rejects_shape_mismatch() -> None:
    chain, pi = _underactuated_chain()
    regressors, _, trials, phases = _true_nodes(chain, pi, 1, 4, seed=1)
    with pytest.raises(ContractViolationError):
        _problem(chain, pi, np.ones(3), regressors, trials, phases)
