"""Tests for src/research/differentiable/engine.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.differentiable.engine import (
    AutodiffBackend,
    ContactDifferentiableEngine,
    DifferentiableEngine,
    OptimizationResult,
)


class TestAutodiffBackend:
    def test_values(self) -> None:
        assert AutodiffBackend("numpy") is AutodiffBackend.NUMPY
        assert AutodiffBackend("jax") is AutodiffBackend.JAX
        assert AutodiffBackend("torch") is AutodiffBackend.TORCH


class TestOptimizationResult:
    def test_fields(self) -> None:
        r = OptimizationResult(
            success=True,
            optimal_states=np.zeros((3, 4)),
            optimal_controls=np.zeros((2, 2)),
            final_cost=0.5,
            iterations=3,
            gradient_norm=0.01,
        )
        assert r.success is True
        assert r.iterations == 3


class TestDifferentiableEngine:
    def test_construction(self, fake_engine) -> None:
        de = DifferentiableEngine(fake_engine, backend="numpy")
        assert de._n_q == 2
        assert de._n_v == 2
        assert de._n_x == 4
        assert de._n_u == 2

    def test_no_n_q(self) -> None:
        class Bare:
            pass

        de = DifferentiableEngine(Bare())
        assert de._n_q == 7
        assert de._n_v == 7

    def test_simulate_trajectory(self, fake_engine) -> None:
        de = DifferentiableEngine(fake_engine)
        x0 = np.zeros(4)
        u = np.zeros((3, 2))
        traj = de.simulate_trajectory(x0, u, dt=0.01)
        assert traj.shape == (4, 4)
        np.testing.assert_allclose(traj[0], x0)

    def test_compute_gradient_shape(self, fake_engine) -> None:
        de = DifferentiableEngine(fake_engine)
        x0 = np.zeros(4)
        u = np.zeros((2, 2))

        def loss(traj):  # noqa: ANN001
            return float(np.sum(traj**2))

        g = de.compute_gradient(x0, u, loss, dt=0.01)
        assert g.shape == (2, 2)

    def test_compute_jacobian(self, fake_engine) -> None:
        de = DifferentiableEngine(fake_engine)
        A, B = de.compute_jacobian(np.zeros(4), np.zeros(2), dt=0.01)
        assert A.shape == (4, 4)
        assert B.shape == (4, 2)

    def test_optimize_trajectory_adam(self, fake_engine) -> None:
        de = DifferentiableEngine(fake_engine)
        x0 = np.zeros(4)
        # Nonzero goal so gradient is nontrivial and Adam update path runs
        goal = np.array([0.5, 0.5, 0.0, 0.0])
        res = de.optimize_trajectory(
            x0, goal, horizon=2, dt=0.01, method="adam", max_iterations=2
        )
        assert isinstance(res, OptimizationResult)
        assert res.optimal_controls.shape == (2, 2)

    def test_optimize_trajectory_sgd(self, fake_engine) -> None:
        de = DifferentiableEngine(fake_engine)
        res = de.optimize_trajectory(
            np.zeros(4),
            np.array([0.5, 0.0, 0.0, 0.0]),
            horizon=2,
            method="sgd",
            max_iterations=2,
        )
        assert isinstance(res, OptimizationResult)

    def test_optimize_through_contact_nontrivial(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(
            fake_engine, contact_method="smoothed", smoothing_factor=0.01
        )
        # Pass nonzero goal so adam update path executes
        res = cd.optimize_through_contact(
            initial_state=np.zeros(4),
            goal_state=np.array([0.3, 0.0, 0.0, 0.0]),
            contact_schedule=[True, False],
            horizon=2,
            dt=0.01,
        )
        assert isinstance(res, OptimizationResult)


class TestContactDifferentiableEngine:
    def test_construction(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(
            fake_engine, contact_method="smoothed", smoothing_factor=0.02
        )
        assert cd.contact_method == "smoothed"
        assert cd.smoothing_factor == 0.02

    def test_compute_gradient_smoothed(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(fake_engine, contact_method="smoothed")
        u = np.zeros((2, 2))

        def loss(traj):  # noqa: ANN001
            return float(np.sum(traj**2))

        g = cd.compute_gradient(np.zeros(4), u, loss, dt=0.01)
        assert g.shape == (2, 2)

    def test_compute_gradient_stochastic(self, fake_engine) -> None:
        np.random.seed(0)
        cd = ContactDifferentiableEngine(fake_engine, contact_method="stochastic")
        u = np.zeros((2, 2))

        def loss(traj):  # noqa: ANN001
            return float(np.sum(traj**2))

        g = cd.compute_gradient(np.zeros(4), u, loss, dt=0.01)
        assert g.shape == (2, 2)

    def test_compute_gradient_randomized(self, fake_engine, monkeypatch) -> None:
        np.random.seed(0)
        cd = ContactDifferentiableEngine(fake_engine, contact_method="randomized")
        u = np.zeros((1, 2))

        def loss(traj):  # noqa: ANN001
            return float(np.sum(traj**2))

        # Reduce n_samples by monkeypatching the parent compute_gradient call indirectly:
        # we just run with default 10 samples but only horizon=1 so it's fast.
        g = cd.compute_gradient(np.zeros(4), u, loss, dt=0.01)
        assert g.shape == (1, 2)

    def test_pad_contact_schedule_truncate(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(fake_engine)
        assert cd._pad_contact_schedule([True, False, True], 2) == [True, False]

    def test_pad_contact_schedule_extend(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(fake_engine)
        assert cd._pad_contact_schedule([True], 3) == [True, False, False]

    def test_build_contact_loss_no_transitions(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(fake_engine)
        loss = cd._build_contact_loss(np.zeros(4), [False, False, False], 1.0)
        traj = np.zeros((4, 4))
        traj[-1] = [1, 0, 0, 0]
        # No transitions; just goal error = 1
        assert loss(traj) == pytest.approx(1.0)

    def test_build_contact_loss_with_transition(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(fake_engine)
        loss = cd._build_contact_loss(np.zeros(4), [True, False, False], 0.5)
        traj = np.zeros((4, 4))
        traj[2, 2:] = [1.0, 0.0]  # v_curr
        traj[3, 2:] = [2.0, 0.0]  # v_next
        # transition at t=0; v diff = [1,0] -> penalty = 0.5
        # final goal error = ||traj[-1] - 0||^2 = 4
        assert loss(traj) == pytest.approx(4.5)

    def test_apply_phase_smoothing(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(fake_engine, smoothing_factor=0.01)
        cd._apply_phase_smoothing([True, False], 0.01, 5.0)
        assert cd.smoothing_factor == 0.05
        cd._apply_phase_smoothing([False, False], 0.01, 5.0)
        assert cd.smoothing_factor == 0.01

    def test_optimize_through_contact(self, fake_engine) -> None:
        cd = ContactDifferentiableEngine(
            fake_engine, contact_method="smoothed", smoothing_factor=0.01
        )
        res = cd.optimize_through_contact(
            initial_state=np.zeros(4),
            goal_state=np.zeros(4),
            contact_schedule=[False, True],
            horizon=2,
            dt=0.01,
        )
        assert isinstance(res, OptimizationResult)
        # smoothing reset
        assert cd.smoothing_factor == 0.01
