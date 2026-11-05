"""Superposition residual of forward counterfactual trajectories (Issue #14493).

At one fixed state the ZTCF drift and the ZVCF control component
``M(q)^-1 tau`` sum to the total acceleration, gravity counted once
(:func:`~.ztcf_zvcf.drift_and_control_split`). That identity is pointwise.
Fork three trajectories at ``t0`` and integrate them, and each reaches its own
state, where ``M(q)``, ``C(q, v)`` and ``g(q)`` differ; the per-trajectory
accelerations then no longer sum to the base acceleration::

    residual(t) = || base(t) - (ztcf(t) + control(t)) ||

The residual is zero only at the fork instant. Any report that sums integrated
counterfactual trajectories must show it rather than call the sum exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import check_finite, require
from src.shared.python.simulation_backends.ztcf_zvcf import (
    drift_and_control_split,
    ztcf_acceleration,
)

if TYPE_CHECKING:
    from .protocol import EquationsOfMotionProvider

__all__ = [
    "SuperpositionResidual",
    "fork_counterfactual_accelerations",
    "superposition_residual",
]


@dataclass(frozen=True)
class SuperpositionResidual:
    """Residual of ``base - (ztcf + control)`` along forked trajectories.

    Attributes:
        times: Sample times ``(T,)``, strictly increasing; ``times[0]`` is the fork.
        residual: Euclidean norm of ``base - (ztcf + control)`` per sample, ``(T,)``.
    """

    times: np.ndarray
    residual: np.ndarray

    @property
    def fork_residual(self) -> float:
        """Residual at the fork instant, where the pointwise identity holds."""
        return float(self.residual[0])

    @property
    def max_residual(self) -> float:
        """Largest residual over the series."""
        return float(self.residual.max())

    def describe(self) -> str:
        """One-line report; integrated sums are never called exact."""
        return (
            "ZTCF + ZVCF matches the base only pointwise at the fork "
            f"(residual {self.fork_residual:.3e}); the integrated sum is not exact "
            f"(max residual {self.max_residual:.3e} over {self.times.size} samples)."
        )


def _as_series(
    name: str, value: np.ndarray, shape: tuple[int, ...] | None
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    require(
        arr.ndim == 2 and arr.shape[0] > 0, f"{name} must be (T, n)", value=arr.shape
    )
    if shape is not None:
        require(arr.shape == shape, f"{name} shape must be {shape}", value=arr.shape)
    require(check_finite(arr), f"{name} must contain only finite values", value=name)
    return arr


def superposition_residual(
    base: np.ndarray,
    ztcf_traj: np.ndarray,
    control_traj: np.ndarray,
    *,
    times: np.ndarray | None = None,
) -> SuperpositionResidual:
    """Report ``||base - (ztcf + control)||`` at every sample.

    Args:
        base: Base-trajectory series ``(T, n)``, e.g. accelerations.
        ztcf_traj: The ZTCF (drift) trajectory's series, same shape.
        control_traj: The ZVCF (control) trajectory's series, same shape.
        times: Sample times ``(T,)``, strictly increasing. Defaults to ``0..T-1``.

    Returns:
        The residual series. It is zero at the fork and generally positive after.

    Raises:
        ValueError: On mismatched shapes, non-finite values or bad ``times``.
    """
    base_arr = _as_series("base", base, None)
    ztcf_arr = _as_series("ztcf_traj", ztcf_traj, base_arr.shape)
    control_arr = _as_series("control_traj", control_traj, base_arr.shape)
    count = base_arr.shape[0]
    t = np.arange(count, dtype=float) if times is None else np.asarray(times, float)
    require(t.shape == (count,), f"times must have shape ({count},)", value=t.shape)
    require(
        check_finite(t) and bool(np.all(np.diff(t) > 0.0)),
        "times must be finite and strictly increasing",
        value=t,
    )
    residual = np.linalg.norm(base_arr - (ztcf_arr + control_arr), axis=1)
    return SuperpositionResidual(times=t, residual=residual)


def _control_acceleration(
    provider: EquationsOfMotionProvider, q: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    # The control component depends on q only, so v = 0 is the ZVCF fork.
    return drift_and_control_split(provider, q, np.zeros_like(q), tau)[1]


def _total_acceleration(
    provider: EquationsOfMotionProvider, q: np.ndarray, v: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    drift, control = drift_and_control_split(provider, q, v, tau)
    return drift + control


def fork_counterfactual_accelerations(
    provider: EquationsOfMotionProvider,
    q0: np.ndarray,
    v0: np.ndarray,
    tau: np.ndarray,
    *,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fork base, ZTCF and ZVCF trajectories at ``(q0, v0)`` and integrate them.

    * base: the full dynamics under ``tau`` from ``(q0, v0)``;
    * ZTCF: the drift (``tau = 0``) from ``(q0, v0)``;
    * ZVCF: the control component ``M(q)^-1 tau`` from ``(q0, 0)``, with
      velocity and gravity off (Dieter's 2026-11-03 decision on #14497).

    Each trajectory uses semi-implicit Euler with constant ``tau`` and records
    its own acceleration at each sample, so the three series feed
    :func:`superposition_residual` directly.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces``.
        q0: Fork joint positions ``(n,)`` [rad].
        v0: Fork joint velocities ``(n,)`` [rad/s].
        tau: Constant joint torque ``(n,)`` [N*m].
        dt: Step size [s], ``> 0``.
        steps: Number of steps, ``>= 1``; ``steps + 1`` samples are returned.

    Returns:
        ``(times, base, ztcf, control)``; the last three are ``(steps + 1, n)``.

    Raises:
        ValueError: If ``dt`` or ``steps`` is not positive, or the state is invalid.
    """
    require(np.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", value=dt)
    require(
        int(steps) == steps and steps >= 1, "steps must be an integer >= 1", value=steps
    )
    q_init = np.asarray(q0, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    fields = {
        "base": (
            np.asarray(v0, float),
            lambda q, v: _total_acceleration(provider, q, v, tau_arr),
        ),
        "ztcf": (np.asarray(v0, float), lambda q, v: ztcf_acceleration(provider, q, v)),
        "control": (
            np.zeros_like(q_init),
            lambda q, _v: _control_acceleration(provider, q, tau_arr),
        ),
    }
    series: dict[str, np.ndarray] = {}
    for name, (v_init, field) in fields.items():
        q, v = q_init.copy(), v_init.copy()
        rows = []
        for _ in range(steps + 1):
            a = field(q, v)
            rows.append(a)
            v = v + dt * a
            q = q + dt * v
        series[name] = np.asarray(rows)
    times = dt * np.arange(steps + 1, dtype=float)
    return times, series["base"], series["ztcf"], series["control"]
