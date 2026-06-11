"""Fallback engines for the cross-engine dashboard."""

from __future__ import annotations

import numpy as np


class StubEngine:
    """Minimal steppable engine for unavailable physics packages.

    Design by Contract
    ------------------
    Pre:  name must be a non-empty string
    Post: get_state() returns two 1-D arrays of equal length
    """

    def __init__(self, name: str, n_dof: int = 2) -> None:
        if not name:
            raise ValueError("Engine stub name must be non-empty")
        self._name = name
        self._n_dof = n_dof
        self._q = np.zeros(n_dof)
        self._v = np.zeros(n_dof)

    def reset(self) -> None:
        """Reset state to zero."""
        self._q = np.zeros(self._n_dof)
        self._v = np.zeros(self._n_dof)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control as an impulse to velocity."""
        u_arr = np.asarray(u, dtype=float)
        n = min(len(u_arr), self._n_dof)
        self._v[:n] += u_arr[:n] * 0.01

    def step(self, dt: float | None = None) -> None:
        """Integrate with Euler plus damping."""
        effective_dt = dt if dt is not None else 0.01
        damping = 0.95
        self._q = self._q + self._v * effective_dt  # type: ignore[assignment]
        self._v = self._v * damping  # type: ignore[assignment]

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and velocities."""
        return self._q.copy(), self._v.copy()
