"""End-to-end subject pipeline: fit, open-loop replay gate, policy, signature."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.inertial import (
    PlanarParameterization,
    project_planar_consistent,
)
from src.shared.python.estimation.mosaic.outer_solve import OuterOptions
from src.shared.python.estimation.mosaic.subject_fit import ReplayConfig, fit_subject
from tests.unit.estimation.mosaic.test_outer_solve import (
    TRUE_LENGTHS,
    TRUE_PI,
    _factory,
    _priors,
    _synthetic_trials,
)

pytestmark = pytest.mark.unit


def _step_factory(geometry: np.ndarray, parameters: np.ndarray, dt: float):
    chain = _factory(geometry)

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return chain.step(x, u, parameters, dt)

    return step


def test_subject_pipeline_passes_open_loop_replay_gate() -> None:
    trials, truth = _synthetic_trials(n_trials=2, n_nodes=150, noise=5e-4, seed=8)
    rng = np.random.default_rng(2)
    pi_prior = TRUE_PI * (1.0 + rng.uniform(-0.1, 0.1, size=TRUE_PI.size))
    pi_prior[2::4] = 0.0
    pi_prior = project_planar_consistent(
        pi_prior.reshape(-1, 4) * [1, 1, 1, 1.1]
    ).ravel()
    config = ReplayConfig(
        step_factory=_step_factory,
        state_weight=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
        input_weight=np.eye(2) * 0.01,
        position_tolerance=0.15,
        initial_perturbation=np.zeros(6),
    )
    report = fit_subject(
        _factory,
        trials,
        [t[0][0] + 0.1 for t in truth],
        TRUE_LENGTHS * 1.05,
        pi_prior,
        PlanarParameterization(),
        _priors(pi_prior, True),
        OuterOptions(),
        config,
        n_phase_bins=15,
    )
    assert report.open_loop_accepted, [r.open_loop_rms for r in report.replays]
    assert all(r.closed_loop_rms <= r.open_loop_rms + 1e-12 for r in report.replays)
    assert report.observability.rank >= 4
    assert report.template is not None and report.template.shape == (15, 2)
    assert report.modes is not None and report.modes.scores.shape == (2, 1)
    assert set(report.timings_s) == {
        "initialisation",
        "joint_fit",
        "observability",
        "replay",
    }
    assert all(value > 0.0 for value in report.timings_s.values())
    assert len(report.gains) == 2 and report.gains[0].shape == (149, 2, 6)
