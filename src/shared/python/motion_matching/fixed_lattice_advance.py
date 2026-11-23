"""Fixed-hold RK4 prefixes; neither interval enclosures nor contact certificates."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input.integrator import zoh_rk4_step

Array = NDArray[np.float64]


class FixedLatticeAdvance:
    """Evaluate a held-effort step from cached fixed-substep endpoints.

    Each interior query advances only from its preceding lattice endpoint.
    The fixed effort is never resampled. If a projected endpoint is supplied,
    a mismatch fails closed. Matching endpoints do not prove continuity or
    bound roundoff, integration error, state velocity, or contact coverage.
    Use separate instances at holds, impact edges and projection boundaries.
    """

    coverage: Literal["uncertified"] = "uncertified"

    def __init__(
        self,
        plant: Any,
        q0: Any,
        v0: Any,
        tau: Any,
        duration_s: float,
        substeps: int,
        *,
        projected_end: tuple[Any, Any] | None = None,
    ) -> None:
        q, v, effort = (np.asarray(x, dtype=float).copy() for x in (q0, v0, tau))
        if q.ndim != 1 or not q.size or v.shape != q.shape or effort.shape != q.shape:
            raise ValueError("q, v and held effort must be matching nonempty vectors")
        if not all(np.isfinite(x).all() for x in (q, v, effort)):
            raise ValueError("states and effort must be finite")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration must be finite and positive")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        self._plant, self._effort = plant, effort
        self._h = duration_s / substeps
        self._edges = np.arange(substeps + 1, dtype=float) * self._h
        self._edges[-1] = duration_s
        if not np.isfinite(self._edges).all() or np.any(np.diff(self._edges) <= 0):
            raise ValueError("lattice is not representable")
        self._states = [(q, v)]
        for _ in range(substeps):
            q, v = zoh_rk4_step(plant, q, v, effort, self._h)
            self._states.append((q.copy(), v.copy()))
        self.projection_status: Literal["unchecked", "identical"] = "unchecked"
        if projected_end is not None:
            expected_q, expected_v = (np.asarray(x, dtype=float) for x in projected_end)
            if not np.array_equal(q, expected_q) or not np.array_equal(v, expected_v):
                raise ValueError("projection boundary changes the endpoint state")
            self.projection_status = "identical"

    def __call__(self, t: float) -> tuple[Array, Array]:
        if not math.isfinite(t) or not 0 <= t <= self._edges[-1]:
            raise ValueError("time must be finite and inside this held step")
        index = int(np.searchsorted(self._edges, t, side="right") - 1)
        q, v = self._states[index]
        if t == self._edges[index]:
            return q.copy(), v.copy()
        q_next, v_next = zoh_rk4_step(
            self._plant, q, v, self._effort, float(t - self._edges[index])
        )
        return q_next.copy(), v_next.copy()
