"""Declared joint impedance for torque-driven models (NSX-93, #12722; rule V5).

``H-DECLARED-JOINT-IMPEDANCE`` is a *declared* time-windowed stiffness ``K(t)``
and damping ``D(t)`` about the reference trajectory::

    tau_imp = K(t) (q_ref - q) + D(t) (v_ref - v)

in the units of the allocated torques ([N m/rad], [N m s/rad]).  Nothing in this
module estimates a stiffness: net joint torque carries no stiffness or
co-contraction information (rule R6), so every value is the user's declaration
and every result built on it is conditioned on the hypothesis (the
``conditioning`` evidence axis, NSX-81), never reported as measured.  The cited
source (``CIT-BURDET-2001``) supports only that humans can control the
*geometry* of limb impedance selectively; it supports neither a particular
stiffness nor co-contraction, and is not used as if it did.

Matrices may carry off-diagonal terms (biarticular coupling).  They must be
symmetric positive semi-definite, so the impedance is restoring and dissipative
for any deviation (a negative stiffness would be a destabilising hypothesis and
is rejected rather than silently accepted).

A window is an indicator of ``[t_start, t_end)`` optionally shaped by raised
cosine edges of length ``ramp_s`` (weight ``(1 - cos(pi s)) / 2``), which keeps
the load continuous for the fixed-step integrator; ``ramp_s = 0`` is a hard
window.  Overlapping windows add.

``noise_inflation`` is the declared noise-inflation penalty of stiffening: the
torque-noise standard deviation of joint ``i`` is multiplied by ``1 +
kappa_i K_ii(t)`` ([kappa] = rad/(N m)).  The value is a modelling input with no
source; its only role is to show the stiffness benefit next to its declared
noise cost (V5).  Pure numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array, readonly_float
from src.shared.python.loop_nullspace.replay_spec import check_gain

__all__ = ["HYPOTHESIS_ID", "ImpedanceWindow", "JointImpedance"]

HYPOTHESIS_ID = "H-DECLARED-JOINT-IMPEDANCE"


def _as_matrix(gain: Array, label: str) -> Array:
    if gain.ndim == 1:
        return np.asarray(np.diag(gain))
    return gain


@dataclass(frozen=True)
class ImpedanceWindow:
    """Stiffness and damping declared on ``[t_start_s, t_end_s)``.

    Attributes:
        stiffness, damping: Per-joint vectors ``(n,)`` (entries ``>= 0``) or
            symmetric PSD matrices ``(n, n)``.
        ramp_s: Raised-cosine edge length inside the window (``0`` = hard); both
            edges must fit (``2 ramp_s <= t_end_s - t_start_s``).

    Raises:
        ValueError: Non-finite or negative gains, shape mismatch, an empty
            window or an edge that does not fit.
    """

    t_start_s: float
    t_end_s: float
    stiffness: Array
    damping: Array
    ramp_s: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.t_start_s) and math.isfinite(self.t_end_s)):
            raise ValueError("window times must be finite")
        if not self.t_end_s > self.t_start_s:
            raise ValueError(
                f"t_end_s ({self.t_end_s}) must exceed t_start_s ({self.t_start_s})"
            )
        if not (math.isfinite(self.ramp_s) and self.ramp_s >= 0.0):
            raise ValueError(f"ramp_s must be finite and >= 0, got {self.ramp_s}")
        if 2.0 * self.ramp_s > self.t_end_s - self.t_start_s:
            raise ValueError("ramp_s: both raised-cosine edges must fit the window")
        k = readonly_float(self.stiffness, "stiffness")
        d = readonly_float(self.damping, "damping")
        if k.ndim not in (1, 2) or k.shape != d.shape or k.size == 0:
            raise ValueError(
                f"stiffness and damping must share a (n,) or (n, n) shape: {k.shape}"
            )
        if k.ndim == 2 and k.shape[0] != k.shape[1]:
            raise ValueError(f"stiffness matrix must be square, got {k.shape}")
        check_gain(k, "stiffness")
        check_gain(d, "damping")
        object.__setattr__(self, "stiffness", k)
        object.__setattr__(self, "damping", d)

    @property
    def n_actuated(self) -> int:
        """Number of joints the window acts on."""
        return int(self.stiffness.shape[0])

    def weight(self, t: float) -> float:
        """Window weight in ``[0, 1]`` at time ``t`` (0 outside the window)."""
        if not self.t_start_s <= t < self.t_end_s:
            return 0.0
        if self.ramp_s == 0.0:
            return 1.0
        weight = 1.0
        for distance in (t - self.t_start_s, self.t_end_s - t):
            if distance < self.ramp_s:
                edge = 0.5 * (1.0 - math.cos(math.pi * distance / self.ramp_s))
                weight = min(weight, edge)
        return weight


@dataclass(frozen=True)
class JointImpedance:
    """A declared impedance layer: windows plus the declared noise penalty.

    ``noise_inflation`` is ``kappa`` ``(n,)`` (``>= 0``) or ``None`` (no penalty
    declared: the noise is then *not* inflated, which the evidence record states
    through the hypotheses it lists).

    Raises:
        ValueError: No window, windows on different joint counts, or an invalid
            ``noise_inflation``.
    """

    windows: Sequence[ImpedanceWindow]
    noise_inflation: Array | None = None
    name: str = "declared joint impedance"
    _n: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self) -> None:
        windows = tuple(self.windows)
        if len(windows) == 0:
            raise ValueError("JointImpedance needs at least one window")
        if not all(isinstance(w, ImpedanceWindow) for w in windows):
            raise TypeError("windows must be ImpedanceWindow instances")
        n = windows[0].n_actuated
        if any(w.n_actuated != n for w in windows):
            raise ValueError("every window must act on the same number of joints")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "_n", n)
        if self.noise_inflation is not None:
            kappa = readonly_float(self.noise_inflation, "noise_inflation")
            if kappa.shape != (n,) or np.any(kappa < 0.0):
                raise ValueError(
                    f"noise_inflation must be non-negative with shape ({n},)"
                )
            object.__setattr__(self, "noise_inflation", kappa)

    @classmethod
    def zero(cls, n_actuated: int) -> JointImpedance:
        """The null impedance (negative control): zero stiffness and damping."""
        if n_actuated < 1:
            raise ValueError(f"n_actuated must be >= 1, got {n_actuated}")
        zeros = np.zeros(n_actuated)
        return cls((ImpedanceWindow(0.0, 1.0, zeros, zeros),), name="zero impedance")

    @property
    def n_actuated(self) -> int:
        """Number of joints."""
        return self._n

    @property
    def hypotheses(self) -> tuple[str, ...]:
        """The registry hypothesis every result built on this layer carries."""
        return (HYPOTHESIS_ID,)

    @property
    def is_zero(self) -> bool:
        """Whether every declared stiffness and damping is exactly zero."""
        return all(not (np.any(w.stiffness) or np.any(w.damping)) for w in self.windows)

    def _sum(self, t: float, attr: str) -> Array:
        total = np.zeros((self._n, self._n))
        for window in self.windows:
            weight = window.weight(t)
            if weight != 0.0:
                total = total + weight * _as_matrix(getattr(window, attr), attr)
        return total

    def stiffness_at(self, t: float) -> Array:
        """``K(t)`` as an ``(n, n)`` matrix."""
        return self._sum(t, "stiffness")

    def damping_at(self, t: float) -> Array:
        """``D(t)`` as an ``(n, n)`` matrix."""
        return self._sum(t, "damping")

    def torque(self, q_error: ArrayLike, v_error: ArrayLike, t: float) -> Array:
        """``K(t) e + D(t) de`` for the errors ``e = ref - state`` (restoring)."""
        e = np.asarray(q_error, dtype=np.float64)
        de = np.asarray(v_error, dtype=np.float64)
        if e.shape != (self._n,) or de.shape != (self._n,):
            raise ValueError(f"errors must have shape ({self._n},)")
        return np.asarray(self.stiffness_at(t) @ e + self.damping_at(t) @ de)

    def gain_series(self, times: ArrayLike) -> tuple[Array, Array]:
        """``(K, D)`` at each time: two arrays of shape ``(T, n, n)``."""
        ts = np.asarray(times, dtype=np.float64).reshape(-1)
        k = np.stack([self.stiffness_at(float(t)) for t in ts])
        d = np.stack([self.damping_at(float(t)) for t in ts])
        return k, d

    def noise_scale_at(self, t: float) -> Array:
        """Declared noise-std multiplier ``1 + kappa_i K_ii(t)`` per joint."""
        if self.noise_inflation is None:
            return np.ones(self._n)
        return np.asarray(1.0 + self.noise_inflation * np.diag(self.stiffness_at(t)))

    def noise_scale_series(self, times: ArrayLike) -> Array:
        """:meth:`noise_scale_at` for every time, shape ``(T, n)``."""
        ts = np.asarray(times, dtype=np.float64).reshape(-1)
        return np.stack([self.noise_scale_at(float(t)) for t in ts])
