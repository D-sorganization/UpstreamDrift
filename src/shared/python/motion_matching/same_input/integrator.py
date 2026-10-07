"""Engine-agnostic fixed-step integration for same-input parity (#11607).

Every engine is integrated by this one loop, so an engine difference cannot
hide in the integrator: the effort is held constant over each step
(zero-order hold) and integrated with ``substeps`` classical RK4 substeps,
then :func:`project_to_closure` applies the engine's own closure residuals.
Root (pelvis) efforts must be zero.

Substeps are required, not optional: the open-loop linearisation has stiff,
stable ground-contact modes near -1e4 1/s, so a single 1 ms RK4 step has
amplification |R(lambda dt)| ~ 270 and round-off differences between engines
grow ~7x per step.  Eight substeps (0.125 ms) keep |lambda dt| ~ 1.2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input.closure import (
    ClosurePlant,
    project_to_closure,
)

Array = NDArray[np.float64]
EffortSource = Callable[[int, float, Array, Array], Array]
DEFAULT_SUBSTEPS = 8

ROOT_COORDINATES: tuple[str, ...] = (
    "TranslationInputX",
    "TranslationInputY",
    "TranslationInputZ",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
)


class DynamicsPlant(ClosurePlant, Protocol):
    """Plant with constrained accelerations and spec coordinate order."""

    coordinate_order: tuple[str, ...]

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array: ...


@dataclass(frozen=True)
class Rollout:
    """States ``q``/``v`` (steps + 1, nv), held efforts (steps, nv) and drift.

    ``pose_drift`` is the closure pose residual each step left before its
    projection; it measures what the projection removed.
    """

    time_s: Array
    q: Array
    v: Array
    efforts: Array
    pose_drift: Array


def root_indices(coordinate_order: Sequence[str]) -> Array:
    """Indices of the unactuated pelvis root coordinates."""
    missing = [name for name in ROOT_COORDINATES if name not in coordinate_order]
    if missing:
        raise ValueError(f"coordinate order lacks root coordinates {missing}")
    return np.array([list(coordinate_order).index(n) for n in ROOT_COORDINATES])


def zoh_rk4_step(
    plant: DynamicsPlant, q: Array, v: Array, tau: Array, dt_s: float
) -> tuple[Array, Array]:
    """One classical RK4 step with ``tau`` held constant over the step."""
    k1q, k1v = v, plant.acceleration(q, v, tau)
    q2, v2 = q + dt_s / 2 * k1q, v + dt_s / 2 * k1v
    k2q, k2v = v2, plant.acceleration(q2, v2, tau)
    q3, v3 = q + dt_s / 2 * k2q, v + dt_s / 2 * k2v
    k3q, k3v = v3, plant.acceleration(q3, v3, tau)
    q4, v4 = q + dt_s * k3q, v + dt_s * k3v
    k4q, k4v = v4, plant.acceleration(q4, v4, tau)
    q_next = q + dt_s / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
    v_next = v + dt_s / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    if not (np.isfinite(q_next).all() and np.isfinite(v_next).all()):
        raise FloatingPointError("Nonfinite state after RK4 step")
    return q_next, v_next


def integrate(
    plant: DynamicsPlant,
    q0: Array,
    v0: Array,
    source: EffortSource,
    *,
    steps: int,
    dt_s: float,
    substeps: int = DEFAULT_SUBSTEPS,
    project: bool = True,
) -> Rollout:
    """Integrate ``steps`` steps, asking ``source(k, t, q, v)`` for each effort.

    The effort is requested once per step at the step's start state and held
    over the step's ``substeps`` RK4 substeps.  Root entries are forced to zero.
    """
    if steps < 1 or substeps < 1 or not (np.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("steps, substeps and dt_s must be positive")
    inner_dt = dt_s / substeps
    root = root_indices(plant.coordinate_order)
    q, v = np.asarray(q0, dtype=float).copy(), np.asarray(v0, dtype=float).copy()
    qs, vs, efforts, drift = [q.copy()], [v.copy()], [], []
    for k in range(steps):
        tau = np.asarray(source(k, k * dt_s, q, v), dtype=float).copy()
        tau[root] = 0.0
        for _ in range(substeps):
            q, v = zoh_rk4_step(plant, q, v, tau, inner_dt)
        if project:
            state = project_to_closure(plant, q, v)
            drift.append(state.pose_residual_before)
            q, v = state.q, state.v
        else:
            drift.append(float(np.abs(plant.closure_pose_residual(q)).max()))
        qs.append(q.copy())
        vs.append(v.copy())
        efforts.append(tau)
    return Rollout(
        time_s=np.arange(steps + 1) * dt_s,
        q=np.array(qs),
        v=np.array(vs),
        efforts=np.array(efforts),
        pose_drift=np.array(drift),
    )


def open_loop(
    plant: DynamicsPlant,
    q0: Array,
    v0: Array,
    efforts: Array,
    *,
    dt_s: float,
    substeps: int = DEFAULT_SUBSTEPS,
    project: bool = True,
) -> Rollout:
    """Replay a fixed effort sequence (steps, nv); root columns must be zero."""
    table = np.asarray(efforts, dtype=float)
    if table.ndim != 2 or table.shape[1] != len(plant.coordinate_order):
        raise ValueError("efforts must be (steps, nv) in spec order")
    if np.any(table[:, root_indices(plant.coordinate_order)] != 0.0):
        raise ValueError("root efforts must be zero")
    return integrate(
        plant,
        q0,
        v0,
        lambda k, _t, _q, _v: table[k],
        steps=table.shape[0],
        dt_s=dt_s,
        substeps=substeps,
        project=project,
    )
