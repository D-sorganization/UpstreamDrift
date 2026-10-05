"""Tests for inertial-parameter observability analysis (MOSAIC)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.identifiability_subspace import (
    data_information,
    posterior_observability,
    structural_observability,
)
from src.shared.python.estimation.mosaic.inner_solve import (
    InnerProblem,
    LinearAnchor,
    solve_inner,
)
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

pytestmark = pytest.mark.unit


def _setup():
    chain = PlanarChain(
        link_lengths=np.array([0.9, 0.6]), actuated=np.array([False, True])
    )
    pi = np.array([1.3, 0.585, 0.0, 0.343, 0.7, 0.175, 0.0, 0.064])
    dt, n = 0.01, 60
    t = np.arange(n) * dt
    u = (1.5 * np.sin(2 * np.pi * 1.7 * t) + 0.4 * np.cos(2 * np.pi * 3.1 * t))[
        None, :, None
    ]
    q, v = chain.rollout(np.array([[0.4, 0.3, 0.2, -0.1]]), u[:, :-1], pi, dt)
    q, v = q.reshape(-1, 2), v.reshape(-1, 2)
    a = chain.forward_dynamics(q, v, u.reshape(-1, 1), pi)
    return chain, pi, chain.regressor(q, v, a)


def test_structural_observability_has_scale_gauge_in_null_space() -> None:
    chain, pi, regressors = _setup()
    report = structural_observability(regressors, chain.unactuated_indices)
    assert report.rank < chain.n_parameters
    # the homogeneous unactuated rows cannot see a global scaling of pi
    projection = report.null_directions.T @ (pi / np.linalg.norm(pi))
    assert np.linalg.norm(projection) > 0.999
    # link-1 mass never enters the pivot torque: fully unobservable
    assert report.observable_fraction[0] < 1e-9
    assert report.parameter_names[0] == "m_0"


def _problem(chain, pi, regressors, anchors=()):
    return InnerProblem(
        regressors=regressors,
        input_matrix=chain.input_matrix,
        trial_index=np.zeros(regressors.shape[0], dtype=np.int64),
        phase_index=None,
        dynamics_weight=np.full(chain.n_links, 10.0),
        parameter_prior_mean=pi,
        parameter_prior_weight=np.full(pi.size, 1e-3),
        input_effort_weight=1e-4,
        input_smoothness_weight=0.0,
        template_weight=0.0,
        anchors=tuple(anchors),
    )


def test_total_mass_anchor_restores_scale_information() -> None:
    chain, pi, regressors = _setup()
    free = solve_inner(_problem(chain, pi, regressors))
    info_free = data_information(free)
    direction = pi / np.linalg.norm(pi)
    along_scale = direction @ info_free @ direction
    assert along_scale < 1e-6 * np.linalg.eigvalsh(info_free)[-1]

    mass_rows = np.array([1, 0, 0, 0, 1, 0, 0, 0.0])
    anchored = solve_inner(
        _problem(chain, pi, regressors, [LinearAnchor(mass_rows, 2.0, 50.0)])
    )
    along_scale_anchored = direction @ data_information(anchored) @ direction
    assert along_scale_anchored > 1e3 * max(along_scale, 1e-300)
    report = posterior_observability(anchored, relative_tolerance=1e-8)
    assert (
        report.rank
        > structural_observability(regressors, chain.unactuated_indices).rank
    )


def test_information_matrix_is_symmetric_psd() -> None:
    chain, pi, regressors = _setup()
    info = data_information(solve_inner(_problem(chain, pi, regressors)))
    np.testing.assert_allclose(info, info.T, atol=1e-9)
    assert np.linalg.eigvalsh(info).min() > -1e-8
