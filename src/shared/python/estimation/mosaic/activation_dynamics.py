"""Declared joint-level activation dynamics (MOSAIC actuation layer).

The applied joint torque follows the commanded torque through a *declared*
first-order lag with separate activation/deactivation time constants, hard
bounds and an optional slew-rate limit.  The model is declared (with
provenance), not identified from muscle data.

Continuous model per joint (``u`` clipped to ``[lower, upper]``)::

    da/dt = (u - a) / tau,   tau = tau_act if u > a else tau_deact

Each interval uses a zero-order-hold command, so the update is the exact
solution ``a+ = u + (a - u) exp(-dt / tau)``; a step response therefore equals
``1 - exp(-t / tau)`` to machine precision.  The optional rate limit clamps
``|a+ - a| <= rate_limit * dt`` afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require

FloatArray: TypeAlias = npt.NDArray[np.float64]

_EPS = 1e-9


def _per_joint(value: object) -> FloatArray:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    require(arr.ndim == 1 and arr.size >= 1, "parameter must be scalar or (nj,)")
    require(bool(np.all(np.isfinite(arr))), "parameter must be finite", arr)
    return arr


@dataclass(frozen=True)
class ActivationModel:
    """Declared per-joint activation model (scalars broadcast to all joints)."""

    tau_act: FloatArray
    tau_deact: FloatArray
    lower: FloatArray
    upper: FloatArray
    rate_limit: FloatArray | None = None
    provenance: str = "declared"

    def __post_init__(self) -> None:
        fields = {n: _per_joint(getattr(self, n)) for n in _REQUIRED}
        n_joints = max(v.size for v in fields.values())
        for name, val in fields.items():
            require(val.size in (1, n_joints), f"{name} size mismatch", val.size)
            object.__setattr__(self, name, np.broadcast_to(val, (n_joints,)).copy())
        if self.rate_limit is not None:
            rate = _per_joint(self.rate_limit)
            require(rate.size in (1, n_joints), "rate_limit size mismatch", rate.size)
            object.__setattr__(
                self, "rate_limit", np.broadcast_to(rate, (n_joints,)).copy()
            )
            require(bool(np.all(rate > 0.0)), "rate_limit must be > 0", rate)
        require(bool(np.all(self.tau_act > 0.0)), "tau_act must be > 0")
        require(bool(np.all(self.tau_deact > 0.0)), "tau_deact must be > 0")
        require(bool(np.all(self.lower < self.upper)), "lower must be < upper")

    @property
    def n_joints(self) -> int:
        """Number of joints the model is declared for."""
        return int(self.tau_act.size)


_REQUIRED = ("tau_act", "tau_deact", "lower", "upper")


@dataclass(frozen=True)
class TimeConstantFit:
    """Recovered time constants; ``None`` means unobservable from the data."""

    tau_act: float | None
    tau_deact: float | None
    n_act_steps: int
    n_deact_steps: int


def _check_times(times: FloatArray) -> None:
    require(times.ndim == 1 and times.size >= 2, "times must be 1-D, >= 2 samples")
    require(bool(np.all(np.isfinite(times))), "times must be finite")
    require(bool(np.all(np.diff(times) > 0.0)), "times must be strictly increasing")


def simulate_activation(
    model: ActivationModel,
    times: FloatArray,
    commands: FloatArray,
    a0: FloatArray | None = None,
) -> FloatArray:
    """Integrate the applied activation over ``times`` for ``(N, nj)`` commands.

    Preconditions: finite strictly increasing ``times``; finite ``commands``
    with one row per time and ``model.n_joints`` columns; ``a0`` (default: the
    clipped first command) finite and within bounds.
    Postcondition: the result is ``(N, nj)``, finite and within
    ``[lower, upper]``.
    """
    _check_times(times)
    require(
        commands.shape == (times.size, model.n_joints),
        "commands must be (N, nj)",
        commands.shape,
    )
    require(bool(np.all(np.isfinite(commands))), "commands must be finite")
    u = np.clip(commands, model.lower, model.upper)
    state = u[0].copy() if a0 is None else np.asarray(a0, dtype=np.float64).copy()
    require(state.shape == (model.n_joints,), "a0 must be (nj,)", state.shape)
    in_bounds = bool(np.all((state >= model.lower) & (state <= model.upper)))
    require(in_bounds, "a0 must lie within bounds", state)
    out = np.empty_like(u)
    out[0] = state
    for k in range(times.size - 1):
        dt = float(times[k + 1] - times[k])
        tau = np.where(u[k] > state, model.tau_act, model.tau_deact)
        nxt = u[k] + (state - u[k]) * np.exp(-dt / tau)
        if model.rate_limit is not None:
            step = model.rate_limit * dt
            nxt = state + np.clip(nxt - state, -step, step)
        state = np.clip(nxt, model.lower, model.upper)
        out[k + 1] = state
    ensure(bool(np.all(np.isfinite(out))), "activation must be finite")
    ensure(
        bool(np.all((out >= model.lower - _EPS) & (out <= model.upper + _EPS))),
        "activation must respect bounds",
    )
    return out


def _decay_tau(
    d0: FloatArray, d1: FloatArray, dt: float, min_steps: int
) -> float | None:
    """Least-squares ``tau`` from ``d1 = exp(-dt/tau) d0``; ``None`` if ill-posed."""
    if d0.size < min_steps:
        return None
    denom = float(np.dot(d0, d0))
    if denom <= _EPS:
        return None
    phi = float(np.dot(d0, d1)) / denom
    if not 0.0 < phi < 1.0:
        return None
    return -dt / float(np.log(phi))


def fit_time_constants(
    times: FloatArray,
    command: FloatArray,
    activation: FloatArray,
    *,
    lower: float,
    upper: float,
    min_steps: int = 3,
) -> TimeConstantFit:
    """Recover ``(tau_act, tau_deact)`` of one joint from a measured response.

    Uses the exact per-step relation ``a+ - u = exp(-dt/tau) (a - u)`` (clipped
    ``u``), separately over rising (``u > a``) and falling steps.  Assumes no
    rate limit is active.  A constant is reported as ``None`` (unobservable)
    when fewer than ``min_steps`` informative steps exist or the estimated
    decay factor is not in ``(0, 1)``.  Preconditions: uniform increasing
    ``times``; equal-length finite 1-D series; ``lower < upper``.
    """
    _check_times(times)
    require(lower < upper, "lower must be < upper")
    require(min_steps >= 1, "min_steps must be >= 1")
    require(command.shape == times.shape == activation.shape, "series length mismatch")
    require(bool(np.all(np.isfinite(command) & np.isfinite(activation))), "finite data")
    steps = np.diff(times)
    dt = float(steps.mean())
    require(bool(np.allclose(steps, dt, rtol=1e-6)), "times must be uniform", dt)
    u = np.clip(command, lower, upper)[:-1]
    d0 = activation[:-1] - u
    d1 = activation[1:] - u
    rising = d0 < -_EPS
    falling = d0 > _EPS
    return TimeConstantFit(
        tau_act=_decay_tau(d0[rising], d1[rising], dt, min_steps),
        tau_deact=_decay_tau(d0[falling], d1[falling], dt, min_steps),
        n_act_steps=int(rising.sum()),
        n_deact_steps=int(falling.sum()),
    )
