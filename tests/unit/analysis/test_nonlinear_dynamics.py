"""Tests for src.shared.python.analysis.nonlinear_dynamics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.analysis.nonlinear_dynamics import NonlinearDynamicsMixin


class _Concrete(NonlinearDynamicsMixin):
    def __init__(self, n: int = 100, n_joints: int = 3) -> None:
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, 2.0, n)
        self.times = t
        self.dt = float(t[1] - t[0])
        # Quasi-periodic signal
        pos = np.zeros((n, n_joints))
        vel = np.zeros((n, n_joints))
        for j in range(n_joints):
            pos[:, j] = np.sin(2 * np.pi * (j + 1) * t) + 0.1 * rng.standard_normal(n)
            vel[:, j] = np.gradient(pos[:, j], self.dt)
        self.joint_positions = pos
        self.joint_velocities = vel


class TestRecurrenceMatrix:
    def setup_method(self) -> None:
        self.obj = _Concrete(n=60, n_joints=3)

    def test_returns_2d_array(self) -> None:
        R = self.obj.compute_recurrence_matrix()
        assert R.ndim == 2

    def test_square_matrix(self) -> None:
        R = self.obj.compute_recurrence_matrix()
        assert R.shape[0] == R.shape[1]

    def test_binary_matrix(self) -> None:
        R = self.obj.compute_recurrence_matrix()
        assert set(np.unique(R)).issubset({0, 1})

    def test_diagonal_is_one(self) -> None:
        R = self.obj.compute_recurrence_matrix()
        # Diagonal should be all recurrent (same point)
        np.testing.assert_array_equal(np.diag(R), 1)

    def test_nonlinear_dynamics_symmetric_matrix(self) -> None:
        R = self.obj.compute_recurrence_matrix()
        np.testing.assert_array_equal(R, R.T)

    def test_empty_data_returns_empty(self) -> None:
        obj = _Concrete(n=60, n_joints=0)
        obj.joint_positions = np.zeros((60, 0))
        obj.joint_velocities = np.zeros((60, 0))
        R = obj.compute_recurrence_matrix()
        assert R.shape == (0, 0)


class TestPermutationEntropy:
    def setup_method(self) -> None:
        self.obj = _Concrete(n=200, n_joints=3)

    def test_nonlinear_dynamics_returns_float(self) -> None:
        data = self.obj.joint_positions[:, 0]
        H = self.obj.compute_permutation_entropy(data=data, order=3)
        assert isinstance(H, float)

    def test_entropy_non_negative(self) -> None:
        data = self.obj.joint_positions[:, 0]
        H = self.obj.compute_permutation_entropy(data=data, order=3)
        # PE is in bits; for order=3, max is log2(6) ≈ 2.58
        assert H >= 0.0

    def test_entropy_finite(self) -> None:
        data = self.obj.joint_positions[:, 0]
        H = self.obj.compute_permutation_entropy(data=data, order=3)
        assert np.isfinite(H)

    def test_different_orders(self) -> None:
        data = self.obj.joint_positions[:, 0]
        H3 = self.obj.compute_permutation_entropy(data=data, order=3)
        H4 = self.obj.compute_permutation_entropy(data=data, order=4)
        assert isinstance(H3, float)
        assert isinstance(H4, float)


class TestLocalDivergenceRate:
    def setup_method(self) -> None:
        self.obj = _Concrete(n=100, n_joints=3)

    def test_nonlinear_dynamics_returns_tuple(self) -> None:
        result = self.obj.compute_local_divergence_rate(joint_idx=0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_times_and_rates_same_length(self) -> None:
        times, rates = self.obj.compute_local_divergence_rate(joint_idx=0)
        assert len(times) == len(rates)

    def test_rates_are_finite(self) -> None:
        _, rates = self.obj.compute_local_divergence_rate(joint_idx=0)
        assert np.all(np.isfinite(rates))
