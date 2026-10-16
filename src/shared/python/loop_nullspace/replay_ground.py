"""Allocated ground loads as a known external load of a replay (NSX-31, NSX-60).

A floating-base model balances its unactuated root rows with foot-ground loads
``f_g`` (the third block of the allocation ``x = (tau, lambda, f_g)``).  The
replay's constrained provider carries the closure only, so the allocated
``f_g`` enters as a *known external generalised load* ``J_g(q)^T f_g`` with the
Jacobian re-evaluated at the replayed configuration.  This is not unilateral
contact dynamics: the contact set is the one the allocation was made for, no
contact is released or made, and a replay that leaves the contact surface keeps
pushing at the allocated force.  The load is therefore reported as part of the
replay's inputs and never as feedback.

Engine-free: the adapter supplies ``contact_names`` and
``compute_contact_jacobian(q)`` (rows ``3 i .. 3 i + 3`` per sphere).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array

__all__ = ["GroundLoadSchedule", "ground_forces_of_allocation"]


def ground_forces_of_allocation(
    x: Sequence[Array], n_actuated: int, n_multipliers: int
) -> tuple[Array, ...]:
    """The ``f_g`` block of each frame's ``x = (tau, lambda, f_g)``."""
    start = n_actuated + n_multipliers
    return tuple(np.asarray(frame, dtype=np.float64)[start:] for frame in x)


class GroundLoadSchedule:
    """``(t, q) -> J_g(q)^T f_g(t)`` from per-frame contact sets and forces.

    The contact set at time ``t`` is that of the latest frame at or before ``t``;
    forces are blended linearly between two frames that share their contact set
    and held otherwise.

    Preconditions (``ValueError``): ``times`` strictly increasing; one contact
    name tuple and one force vector (length ``3 * len(names)``) per frame; every
    name known to the adapter.
    """

    def __init__(
        self,
        adapter: Any,
        times_s: Sequence[float],
        contact_names: Sequence[Sequence[str]],
        forces: Sequence[Array],
    ) -> None:
        times = np.asarray(times_s, dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be non-empty and strictly increasing")
        if not (len(contact_names) == len(forces) == times.size):
            raise ValueError("one contact name tuple and one force vector per frame")
        known = {name: i for i, name in enumerate(adapter.contact_names)}
        self._rows: list[np.ndarray] = []
        for names, force in zip(contact_names, forces, strict=True):
            missing = [n for n in names if n not in known]
            if missing:
                raise ValueError(f"contacts unknown to the adapter: {missing}")
            if np.asarray(force).shape != (3 * len(names),):
                raise ValueError(
                    f"force must have length {3 * len(names)}, got "
                    f"{np.asarray(force).shape}"
                )
            self._rows.append(
                np.array([3 * known[n] + k for n in names for k in range(3)], dtype=int)
            )
        self._adapter = adapter
        self._times = times
        self._names = [tuple(n) for n in contact_names]
        self._forces = [np.asarray(f, dtype=np.float64) for f in forces]

    def force(self, t: float) -> tuple[np.ndarray, Array]:
        """``(row indices of the contact Jacobian, f_g)`` at time ``t``."""
        k = int(
            np.clip(np.searchsorted(self._times, t + 1e-12, side="right") - 1, 0, None)
        )
        force = self._forces[k]
        if k + 1 < self._times.size and self._names[k + 1] == self._names[k]:
            w = float(
                np.clip(
                    (t - self._times[k]) / (self._times[k + 1] - self._times[k]),
                    0.0,
                    1.0,
                )
            )
            force = (1.0 - w) * force + w * self._forces[k + 1]
        return self._rows[k], force

    def __call__(self, t: float, q: Array) -> Array:
        """Generalised load ``J_g(q)^T f_g(t)``, shape ``(nv,)``."""
        rows, force = self.force(t)
        if rows.size == 0:
            return np.zeros(int(self._adapter.nv))
        jacobian = np.asarray(self._adapter.compute_contact_jacobian(q))
        return np.asarray(jacobian[rows].T @ force)
