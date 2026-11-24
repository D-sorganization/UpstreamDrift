"""Planted-ground rows after the closure rows of a provider (#15150 Track Z, Z5).

ZTCF and ZVCF re-solve every reaction for each part (``model_aware_matching``
definition). A provider that knows only the hand-club closure leaves the
golfer floating: with ``tau = 0`` the whole body falls freely, gravity cancels
inside it, and the control part's leg torques push on nothing. The feet are
reactions too, so :class:`GroundedProvider` appends planted-ground rows (a
weld per foot, from the engine adapter) to the closure rows. The closure rows
stay first, so the first ``closure_rows(q)`` multipliers are still the hands'
and the rest are the ground's reaction on each planted body.

A weld holds a foot flat and still: heel lift and foot roll are not modelled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array

__all__ = ["GroundRows", "GroundedProvider"]

#: ``(q, v) -> (J_ground, Jdot_v_ground)``, rows in the provider's coordinates.
GroundRows = Callable[[ArrayLike, ArrayLike], tuple[Array, Array]]


class GroundedProvider:
    """A constrained provider with planted-ground rows after its closure rows.

    ``M`` and ``h`` are the wrapped provider's. Preconditions: ``ground``
    returns a ``(k, nv)`` Jacobian and a ``(k,)`` drift (``ValueError``).
    """

    def __init__(self, provider: Any, ground: GroundRows) -> None:
        self._provider = provider
        self._ground = ground

    @property
    def nv(self) -> int:
        """Number of velocity coordinates."""
        return int(self._provider.nv)

    def mass_matrix(self, q: ArrayLike) -> Array:
        """``M(q)`` of the wrapped provider."""
        return np.asarray(self._provider.mass_matrix(q), dtype=np.float64)

    def bias_forces(self, q: ArrayLike, v: ArrayLike) -> Array:
        """``h(q, v)`` of the wrapped provider."""
        return np.asarray(self._provider.bias_forces(q, v), dtype=np.float64)

    def _rows(self, q: ArrayLike, v: ArrayLike) -> tuple[Array, Array]:
        jac, drift = (np.asarray(x, dtype=np.float64) for x in self._ground(q, v))
        if jac.ndim != 2 or jac.shape[1] != self.nv or drift.shape != (jac.shape[0],):
            raise ValueError(
                f"ground rows must be (k, {self.nv}) and (k,), got {jac.shape} "
                f"and {drift.shape}"
            )
        return jac, drift

    def closure_rows(self, q: ArrayLike) -> int:
        """How many leading rows are the wrapped (hand-club) closure."""
        return int(np.asarray(self._provider.closure_jacobian(q)).shape[0])

    def closure_jacobian(self, q: ArrayLike) -> Array:
        """Closure rows, then the planted-ground rows, shape ``(m + k, nv)``."""
        jac, _ = self._rows(q, np.zeros(self.nv))
        return np.vstack([np.asarray(self._provider.closure_jacobian(q)), jac])

    def closure_drift(self, q: ArrayLike, v: ArrayLike) -> Array:
        """``Jdot v`` of the closure rows, then of the ground rows."""
        _, drift = self._rows(q, v)
        return np.concatenate([np.asarray(self._provider.closure_drift(q, v)), drift])
