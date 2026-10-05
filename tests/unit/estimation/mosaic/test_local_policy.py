"""Tests for TVLQR local policy and replay acceptance (MOSAIC)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import (
    linearize_along_trajectory,
    replay,
    replay_acceptance,
    tvlqr_gains,
)
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

pytestmark = pytest.mark.unit

PI = np.array([1.3, 0.585, 0.0, 0.343, 0.7, 0.175, 0.0, 0.064, 0.3, 0.09, 0.0, 0.03])


def _nominal(n_steps: int = 120):
    chain = PlanarChain(
        link_lengths=np.array([0.9, 0.6, 0.4]), actuated=np.array([False, True, True])
    )
    dt = 0.005
    t = np.arange(n_steps) * dt
    u = np.stack(
        [0.8 * np.sin(2 * np.pi * 1.1 * t), 0.3 * np.cos(2 * np.pi * 1.1 * t)], axis=-1
    )[None]
    x0 = np.array([[-np.pi / 2 + 0.4, 0.3, -0.2, 0.0, 0.0, 0.0]])
    q, v = chain.rollout(x0, u, PI, dt)
    x_ref = np.concatenate([q[0], v[0]], axis=1)

    def step(x: np.ndarray, u_: np.ndarray) -> np.ndarray:
        return chain.step(x, u_, PI, dt)

    return step, x_ref, u[0], dt


def test_linearization_matches_direct_perturbation() -> None:
    step, x_ref, u, _ = _nominal(20)
    lin = linearize_along_trajectory(step, x_ref[:-1], u, 1e-6)
    assert lin.a.shape == (20, 6, 6) and lin.b.shape == (20, 6, 2)
    eps = 1e-6
    bump = np.zeros_like(x_ref[:-1])
    bump[:, 4] = eps
    fd = (step(x_ref[:-1] + bump, u) - step(x_ref[:-1], u)) / eps
    np.testing.assert_allclose(lin.a[:, :, 4], fd, atol=1e-4)
    bump_u = np.zeros_like(u)
    bump_u[:, 1] = eps
    fd_u = (step(x_ref[:-1], u + bump_u) - step(x_ref[:-1], u)) / eps
    np.testing.assert_allclose(lin.b[:, :, 1], fd_u, atol=1e-4)


def test_closed_loop_replay_tracks_better_than_open_loop() -> None:
    step, x_ref, u, dt = _nominal(300)
    lin = linearize_along_trajectory(step, x_ref[:-1], u, 1e-6)
    gains = tvlqr_gains(lin, np.diag([100, 100, 100, 1, 1, 1.0]), np.eye(2) * 0.01)
    assert gains.shape == (u.shape[0], 2, 6)
    assert np.all(np.isfinite(gains))
    x0 = x_ref[0] + np.array([0.05, -0.05, 0.05, 0.0, 0.0, 0.0])
    open_loop = replay(step, x0, u)
    closed_loop = replay(step, x0, u, gains=gains, reference=x_ref)

    def rms(path: np.ndarray, window: slice = slice(None)) -> float:
        error = np.linalg.norm(path[window, :3] - x_ref[window, :3], axis=1)
        return float(np.sqrt(np.mean(error**2)))

    err_open, err_closed = rms(open_loop), rms(closed_loop)
    assert err_closed < 0.5 * err_open
    tail = slice(
        200, None
    )  # after the initial transient the policy must hold the motion
    tail_open, tail_closed = rms(open_loop, tail), rms(closed_loop, tail)
    assert tail_closed < 0.2 * tail_open
    report = replay_acceptance(
        step, x_ref, u, gains, x0 - x_ref[0], position_tolerance=0.02, dt=dt
    )
    assert report.closed_loop_rms == pytest.approx(err_closed)
    assert report.open_loop_rms == pytest.approx(err_open)
    assert report.feedback_effort_rms > 0.0
    assert not report.open_loop_accepted or report.closed_loop_accepted


def test_unperturbed_replay_reproduces_reference_exactly() -> None:
    step, x_ref, u, dt = _nominal(30)
    reproduced = replay(step, x_ref[0], u)
    np.testing.assert_allclose(reproduced, x_ref, atol=1e-12)
    report = replay_acceptance(
        step, x_ref, u, None, np.zeros(6), position_tolerance=1e-9, dt=dt
    )
    assert report.open_loop_accepted and report.divergence_time_s is None
