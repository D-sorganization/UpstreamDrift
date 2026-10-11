"""End-to-end tests for the MOSAIC outer solver on the planar analytic fixture."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.ik_init import (
    initialize_joint_angles,
    initialize_trajectory,
)
from src.shared.python.estimation.mosaic.inertial import PlanarParameterization
from src.shared.python.estimation.mosaic.inner_solve import LinearAnchor
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis
from src.shared.python.estimation.mosaic.outer_solve import (
    OuterOptions,
    OuterPriors,
    TrialData,
    dynamics_jacobian_by_finite_differences,
    fit_trials,
)
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)

pytestmark = pytest.mark.unit

ACTUATED = np.array([False, True, True])
TRUE_LENGTHS = np.array([0.9, 0.6, 0.4])
TRUE_PI = np.array(
    [1.3, 0.585, 0.0, 0.343, 0.7, 0.175, 0.0, 0.064, 0.3, 0.09, 0.0, 0.03]
)
MASS_ROW = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0.0])
# the last link length hangs nothing off its end: unidentifiable, held by prior
IDENTIFIABLE_LENGTHS = slice(0, 2)
MARKERS = PlanarMarkerSet(
    link_index=np.array([0, 0, 1, 1, 2, 2]),
    offsets=np.array(
        [[0.3, 0.02], [0.8, -0.03], [0.2, 0.0], [0.55, 0.04], [0.1, 0.0], [0.35, -0.02]]
    ),
)


def _factory(lengths: np.ndarray) -> PlanarChain:
    return PlanarChain(link_lengths=lengths, actuated=ACTUATED)


def _synthetic_trials(n_trials: int, n_nodes: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    chain = _factory(TRUE_LENGTHS)
    dt = 1.0 / 200.0
    times = np.arange(n_nodes) * dt
    amps = rng.uniform(0.6, 1.2, size=n_trials)
    freq = rng.uniform(0.9, 1.3, size=n_trials)
    phase = 2 * np.pi * freq[:, None] * times[None, :]
    u = np.stack([amps[:, None] * np.sin(phase), 0.3 * np.cos(phase)], axis=-1)
    hanging = np.array([-np.pi / 2 + 0.4, 0.3, -0.2, 0.0, 0.0, 0.0])
    x0 = np.tile(hanging, (n_trials, 1)) + rng.normal(scale=0.05, size=(n_trials, 6))
    q, v = chain.rollout(x0, u[:, :-1], TRUE_PI, dt)
    trials, truth = [], []
    for k in range(n_trials):
        clean = chain.marker_positions(q[k], MARKERS)
        observed = clean + rng.normal(scale=noise, size=clean.shape)
        basis = BSplineBasis.uniform(times, n_coefficients=n_nodes // 4, degree=5)
        trials.append(
            TrialData(
                times, observed, MARKERS, max(noise, 1e-4), basis, np.arange(n_nodes)
            )
        )
        truth.append((q[k], v[k], u[k]))
    return trials, truth


def _priors(
    pi_prior: np.ndarray, anchor_mass: bool, club_known: bool = True
) -> OuterPriors:
    """Realistic priors: 15 % anthropometric sigma, total mass and known club inertia."""
    anchors: list[LinearAnchor] = []
    if anchor_mass:
        anchors.append(LinearAnchor(MASS_ROW, float(MASS_ROW @ TRUE_PI), 100.0))
    if club_known:  # the club is a measured body: it acts as a force sensor
        anchors.extend(
            LinearAnchor(np.eye(12)[i], float(TRUE_PI[i]), 1e3) for i in (8, 9, 11)
        )
    return OuterPriors(
        geometry_mean=TRUE_LENGTHS * 1.05,
        geometry_weight=np.full(3, 1.0 / 0.05),
        parameter_prior_mean=pi_prior,
        parameter_prior_weight=1.0 / (0.15 * np.abs(TRUE_PI) + 0.02),
        dynamics_sigma=np.full(3, 0.2),
        input_effort_weight=0.3,
        input_smoothness_weight=10.0,
        template_weight=0.0,
        anchors=tuple(anchors),
    )


def _lumped_base_parameters(pi: np.ndarray) -> np.ndarray:
    """Gautier-Khalil base combinations observable from the pivot row."""
    l0, l1 = TRUE_LENGTHS[0], TRUE_LENGTHS[1]
    return np.array(
        [
            pi[1] + pi[4] * l0,
            pi[3] + pi[4] * l0**2,
            pi[5] + pi[8] * l1,
            pi[7] + pi[8] * l1**2,
        ]
    )


def test_finite_difference_dynamics_jacobian_matches_direct_perturbation() -> None:
    chain = _factory(TRUE_LENGTHS)
    rng = np.random.default_rng(4)
    q, v, a = (rng.normal(size=(5, 3)) for _ in range(3))
    dq, dv, da, dl = dynamics_jacobian_by_finite_differences(
        _factory, TRUE_LENGTHS, q, v, a, TRUE_PI, 1e-6
    )
    tau0 = chain.regressor(q, v, a) @ TRUE_PI
    eps = 1e-6
    bump = np.zeros_like(q)
    bump[:, 1] = eps
    np.testing.assert_allclose(
        dq[..., 1], (chain.regressor(q + bump, v, a) @ TRUE_PI - tau0) / eps, atol=1e-4
    )
    np.testing.assert_allclose(
        dv[..., 1], (chain.regressor(q, v + bump, a) @ TRUE_PI - tau0) / eps, atol=1e-4
    )
    np.testing.assert_allclose(
        da[..., 1], (chain.regressor(q, v, a + bump) @ TRUE_PI - tau0) / eps, atol=1e-4
    )
    longer = _factory(TRUE_LENGTHS + eps * np.eye(3)[0])
    np.testing.assert_allclose(
        dl[..., 0], (longer.regressor(q, v, a) @ TRUE_PI - tau0) / eps, atol=1e-4
    )


def test_ik_initialization_recovers_angles_from_markers() -> None:
    chain = _factory(TRUE_LENGTHS)
    q_true = np.array([[0.4, 0.3, 0.1], [1.0, -0.5, 0.3], [-0.2, 1.2, -0.4]])
    observed = chain.marker_positions(q_true, MARKERS)
    q0 = q_true + 0.3
    q_hat = initialize_joint_angles(chain, observed, MARKERS, q0)
    np.testing.assert_allclose(q_hat, q_true, atol=1e-8)


def test_joint_fit_recovers_geometry_inputs_and_observable_inertia() -> None:
    n_trials, n_nodes = 2, 150
    trials, truth = _synthetic_trials(
        n_trials=n_trials, n_nodes=n_nodes, noise=5e-4, seed=21
    )
    rng = np.random.default_rng(1)
    pi_prior = TRUE_PI * (1.0 + rng.uniform(-0.15, 0.15, size=TRUE_PI.size))
    pi_prior[2::4] = 0.0
    geometry0 = TRUE_LENGTHS * 1.05
    model0 = _factory(geometry0)
    coefficients0 = []
    for trial, (q_true, _, _) in zip(trials, truth, strict=True):
        q_ik = initialize_trajectory(
            model0, trial.marker_positions, trial.markers, q_true[0] + 0.1
        )
        coefficients0.append(trial.basis.fit_least_squares(q_ik))
    fit = fit_trials(
        _factory,
        trials,
        coefficients0,
        geometry0,
        pi_prior,
        PlanarParameterization(),
        _priors(pi_prior, True),
        OuterOptions(),
    )

    np.testing.assert_allclose(
        fit.geometry[IDENTIFIABLE_LENGTHS],
        TRUE_LENGTHS[IDENTIFIABLE_LENGTHS],
        atol=2e-3,
    )
    np.testing.assert_allclose(fit.geometry[2], TRUE_LENGTHS[2] * 1.05, atol=1e-6)
    for coeffs, trial, (q_true, _, _) in zip(
        fit.coefficients, trials, truth, strict=True
    ):
        q_fit, _, _ = trial.basis.evaluate(coeffs)
        assert np.sqrt(np.mean((q_fit - q_true) ** 2)) < 2e-3
    u_fit = fit.inner.inputs.reshape(n_trials, n_nodes, 2)
    u_true_all = np.stack([t[2] for t in truth])
    assert (
        np.sqrt(np.mean((u_fit - u_true_all) ** 2))
        < 0.2 * np.sqrt(np.mean(u_true_all**2)) + 0.05
    )
    # inertia: only base-parameter combinations are observable from the pivot
    # row; they must be close to truth and the body must stay physically consistent
    lumped_error = np.abs(
        _lumped_base_parameters(fit.parameters) - _lumped_base_parameters(TRUE_PI)
    )
    assert np.all(lumped_error < 0.05)
    np.testing.assert_allclose(
        fit.parameters[8::4][0], TRUE_PI[8], atol=2e-3
    )  # club mass
    assert np.all(fit.parameters[0::4] > 0.0)
    assert all(r.converged for r in fit.receipts)
    assert all(r.iterations <= 30 for r in fit.receipts)
    assert all(r.cost_after <= r.cost_before + 1e-12 for r in fit.receipts)
    assert fit.receipts[-1].wall_time_s > 0.0


def test_fit_rejects_mismatched_initialisation() -> None:
    trials, _ = _synthetic_trials(1, 40, 0.0, 0)
    with pytest.raises(Exception):  # noqa: B017 - contract violation surfaces as ContractViolationError
        fit_trials(
            _factory,
            trials,
            [np.zeros((3, 3))],
            TRUE_LENGTHS,
            TRUE_PI,
            PlanarParameterization(),
            _priors(TRUE_PI, False),
            OuterOptions(),
        )
