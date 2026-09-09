"""Shared validation and batch accessors for simulation result objects."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .validation import require


class TrajectoryResultMixin:
    """Reusable DbC helpers for trajectory result containers."""

    t: np.ndarray
    states: np.ndarray

    @property
    def n_steps(self) -> int:
        return len(self.t)

    def _validate_trajectory(self, expected_state_width: int) -> None:
        require(self.t.ndim == 1, f"t must be 1D, got shape {self.t.shape}")
        require(
            self.states.ndim == 2,
            f"states must be 2D, got shape {self.states.shape}",
        )
        require(self.t.size >= 1, "Trajectory must contain at least one time sample")
        require(
            self.states.shape[0] == self.t.size,
            "states row count must match the number of time samples",
        )
        require(
            self.states.shape[1] == expected_state_width,
            f"states must have width {expected_state_width}, got {self.states.shape[1]}",
        )
        require(bool(np.all(np.isfinite(self.t))), "Time vector must be finite")
        require(
            bool(np.all(np.isfinite(self.states))), "State trajectory must be finite"
        )
        if self.t.size > 1:
            require(
                bool(np.all(np.diff(self.t) > 0)),
                "Time vector must be strictly increasing",
            )

    def _check_idx(self, idx: int) -> None:
        require(idx is not None, "idx must be provided")
        require(
            isinstance(idx, int | np.integer),
            f"idx must be an integer, got {type(idx).__name__}",
        )
        require(
            0 <= idx < self.n_steps, f"Index {idx} out of range [0, {self.n_steps})"
        )

    @staticmethod
    def _assert_energy_finite(result: dict, idx: int) -> None:
        """Shared postcondition: all energy components must be finite."""
        require(
            all(np.isfinite(v) for v in result.values()),
            f"Non-finite energy at idx={idx}: {result}",
        )

    def energy_at(self, idx: int) -> dict:
        """Energy decomposition at time index."""
        self._check_idx(idx)
        state = self.states[idx]
        kinetic = self._kinetic_energy_at(state)
        potential = self._potential_energy_at(state)
        result = {
            "kinetic": kinetic,
            "potential": potential,
            "total": kinetic + potential,
        }
        self._assert_energy_finite(result, idx)
        return result

    def _kinetic_energy_at(self, state: np.ndarray) -> float:
        raise NotImplementedError

    def _potential_energy_at(self, state: np.ndarray) -> float:
        raise NotImplementedError

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Total applied torque (drive + friction) at time index.

        Default implementation: tau_drive + tau_friction.  Subclasses that
        apply torque clamping (e.g. double-pendulum with TorqueClamp) must
        override this method.
        """
        self._check_idx(idx)
        torque_func = self.torque_func  # type: ignore[attr-defined]
        tau_drive = np.array(torque_func(self.t[idx]))
        tau_friction: np.ndarray = self.friction_torques_at(idx)  # type: ignore[attr-defined]
        return np.asarray(tau_drive + tau_friction)

    def _batch_cache(self) -> dict[str, Any]:
        """Lazily attach and return this result's batch-accessor cache.

        Simulation results are treated as immutable after construction:
        ``t``/``states`` are produced once by the integrator and only read
        afterwards, so a lazily-attached cache dict (not a dataclass
        field, to keep positional construction stable) is safe.
        """
        cache = getattr(self, "_result_batch_cache", None)
        if cache is None:
            cache = {}
            self._result_batch_cache = cache
        return cache

    def _cached_batch(self, key: str, factory: Callable[[], Any]) -> Any:
        """Return the cached batch series, computing it once via *factory*."""
        cache = self._batch_cache()
        if key not in cache:
            cache[key] = factory()
        return cache[key]

    def all_positions(self) -> list[Any]:
        def _compute() -> list[Any]:
            positions_at = self.positions_at  # type: ignore[attr-defined]
            return [positions_at(i) for i in range(self.n_steps)]

        return self._cached_batch("all_positions", _compute)

    def all_mass_matrices(self) -> list[Any]:
        def _compute() -> list[Any]:
            mass_matrix_at = self.mass_matrix_at  # type: ignore[attr-defined]
            return [mass_matrix_at(i) for i in range(self.n_steps)]

        return self._cached_batch("all_mass_matrices", _compute)

    def all_energies(self) -> dict[str, np.ndarray]:
        """Cached single-pass energy decomposition.

        ``energy_at`` is evaluated exactly once per time index (not once
        per energy key) and the per-index dicts are transposed into
        arrays (#8928).
        """

        def _compute() -> dict[str, np.ndarray]:
            energy_at = self.energy_at  # type: ignore[attr-defined]
            per_index = [energy_at(i) for i in range(self.n_steps)]
            return {
                key: np.asarray([row[key] for row in per_index], dtype=float)
                for key in per_index[0]
            }

        return self._cached_batch("all_energies", _compute)

    def all_accelerations(self) -> np.ndarray:
        def _compute() -> np.ndarray:
            accelerations_at = self.accelerations_at  # type: ignore[attr-defined]
            return np.asarray(
                [accelerations_at(i) for i in range(self.n_steps)], dtype=float
            )

        return self._cached_batch("all_accelerations", _compute)

    def all_torques(self) -> np.ndarray:
        def _compute() -> np.ndarray:
            torques_at = self.torques_at  # type: ignore[attr-defined]
            return np.asarray([torques_at(i) for i in range(self.n_steps)], dtype=float)

        return self._cached_batch("all_torques", _compute)

    def all_friction_torques(self) -> np.ndarray:
        def _compute() -> np.ndarray:
            friction_torques_at = self.friction_torques_at  # type: ignore[attr-defined]
            return np.asarray(
                [friction_torques_at(i) for i in range(self.n_steps)],
                dtype=float,
            )

        return self._cached_batch("all_friction_torques", _compute)

    def all_total_torques(self) -> np.ndarray:
        def _compute() -> np.ndarray:
            total_torques_at = self.total_torques_at
            return np.asarray(
                [total_torques_at(i) for i in range(self.n_steps)],
                dtype=float,
            )

        return self._cached_batch("all_total_torques", _compute)
