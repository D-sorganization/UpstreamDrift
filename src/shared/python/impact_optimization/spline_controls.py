"""``SplineTorqueControl``: low-dimensional torque parameterisation.

The trust-region optimiser (C3) searches over ``K`` knot torques per joint
instead of every time sample.  Torque at any time is the piecewise-linear
interpolation of the knots, so it is linear in the knot values (convex
constraints stay convex) and holds the end values outside the knot range.

Design by Contract:
    Preconditions: at least two strictly increasing, finite knot times;
    ``n_joints`` >= 1; knot arrays of shape ``(K, n_joints)``.
    Postconditions: ``torque`` returns ``(len(times), n_joints)``, equal to
    the knots at the knot times and linear in the knots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import ensure, require


@dataclass(frozen=True, eq=False)
class SplineTorqueControl:
    """Piecewise-linear torque controls on fixed knot times."""

    knot_times_s: np.ndarray
    n_joints: int

    def __post_init__(self) -> None:
        kt = np.array(self.knot_times_s, dtype=float, copy=True)
        require(kt.ndim == 1 and kt.size >= 2, "need >= 2 knot times", kt.shape)
        require(bool(np.all(np.isfinite(kt))), "knot times must be finite")
        require(bool(np.all(np.diff(kt) > 0.0)), "knot times must strictly increase")
        require(int(self.n_joints) >= 1, "n_joints must be >= 1", self.n_joints)
        kt.setflags(write=False)
        object.__setattr__(self, "knot_times_s", kt)

    @property
    def n_knots(self) -> int:
        return int(self.knot_times_s.size)

    @property
    def n_params(self) -> int:
        return self.n_knots * int(self.n_joints)

    def unpack(self, flat: np.ndarray) -> np.ndarray:
        """Reshape a flat parameter vector to ``(n_knots, n_joints)``."""
        v = np.asarray(flat, dtype=float)
        require(v.size == self.n_params, "flat vector has wrong length", v.size)
        return v.reshape(self.n_knots, int(self.n_joints))

    def torque(self, knots: np.ndarray, times_s: np.ndarray) -> np.ndarray:
        """Interpolate ``knots`` at ``times_s`` -> ``(T, n_joints)``."""
        k = np.asarray(knots, dtype=float)
        require(
            k.shape == (self.n_knots, int(self.n_joints)),
            "knots must be (n_knots, n_joints)",
            k.shape,
        )
        t = np.asarray(times_s, dtype=float)
        require(t.ndim == 1, "times_s must be 1-D", t.shape)
        require(bool(np.all(np.isfinite(t))), "times_s must be finite")
        out = np.stack(
            [np.interp(t, self.knot_times_s, k[:, j]) for j in range(self.n_joints)],
            axis=1,
        )
        ensure(out.shape == (t.size, int(self.n_joints)), "torque shape", out.shape)
        return out
