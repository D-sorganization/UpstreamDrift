"""Shared validation and batch accessors for simulation result objects."""

from __future__ import annotations

from typing import Any

import numpy as np


class TrajectoryResultMixin:
    """Reusable DbC helpers for trajectory result containers."""

    t: np.ndarray
    states: np.ndarray

    @property
    def n_steps(self) -> int:
        return len(self.t)

    def _validate_trajectory(self, expected_state_width: int) -> None:
        assert self.t.ndim == 1, f"t must be 1D, got shape {self.t.shape}"
        assert self.states.ndim == 2, (
            f"states must be 2D, got shape {self.states.shape}"
        )
        assert self.t.size >= 1, "Trajectory must contain at least one time sample"
        assert self.states.shape[0] == self.t.size, (
            "states row count must match the number of time samples"
        )
        assert self.states.shape[1] == expected_state_width, (
            f"states must have width {expected_state_width}, got {self.states.shape[1]}"
        )
        assert np.all(np.isfinite(self.t)), "Time vector must be finite"
        assert np.all(np.isfinite(self.states)), "State trajectory must be finite"
        if self.t.size > 1:
            assert np.all(np.diff(self.t) > 0), (
                "Time vector must be strictly increasing"
            )

    def _check_idx(self, idx: int) -> None:
        assert 0 <= idx < self.n_steps, f"Index {idx} out of range [0, {self.n_steps})"

    @staticmethod
    def _assert_energy_finite(result: dict, idx: int) -> None:
        """Shared postcondition: all energy components must be finite."""
        assert all(np.isfinite(v) for v in result.values()), (
            f"Non-finite energy at idx={idx}: {result}"
        )

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Total applied torque (drive + friction) at time index.

        Default implementation: tau_drive + tau_friction.  Subclasses that
        apply torque clamping (e.g. double-pendulum with TorqueClamp) must
        override this method.
        """
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        torque_func = self.torque_func  # type: ignore[attr-defined]
        tau_drive = np.array(torque_func(self.t[idx]))
        tau_friction: np.ndarray = self.friction_torques_at(idx)  # type: ignore[attr-defined]
        return np.asarray(tau_drive + tau_friction)

    def all_positions(self) -> list[Any]:
        positions_at = self.positions_at  # type: ignore[attr-defined]
        return [positions_at(i) for i in range(self.n_steps)]

    def all_mass_matrices(self) -> list[Any]:
        mass_matrix_at = self.mass_matrix_at  # type: ignore[attr-defined]
        return [mass_matrix_at(i) for i in range(self.n_steps)]

    def all_energies(self) -> dict[str, np.ndarray]:
        energy_at = self.energy_at  # type: ignore[attr-defined]
        first = energy_at(0)
        return {
            key: np.asarray(
                [energy_at(i)[key] for i in range(self.n_steps)], dtype=float
            )
            for key in first
        }

    def all_accelerations(self) -> np.ndarray:
        accelerations_at = self.accelerations_at  # type: ignore[attr-defined]
        return np.asarray(
            [accelerations_at(i) for i in range(self.n_steps)], dtype=float
        )

    def all_torques(self) -> np.ndarray:
        torques_at = self.torques_at  # type: ignore[attr-defined]
        return np.asarray([torques_at(i) for i in range(self.n_steps)], dtype=float)

    def all_friction_torques(self) -> np.ndarray:
        friction_torques_at = self.friction_torques_at  # type: ignore[attr-defined]
        return np.asarray(
            [friction_torques_at(i) for i in range(self.n_steps)],
            dtype=float,
        )

    def all_total_torques(self) -> np.ndarray:
        total_torques_at = self.total_torques_at
        return np.asarray(
            [total_torques_at(i) for i in range(self.n_steps)],
            dtype=float,
        )
