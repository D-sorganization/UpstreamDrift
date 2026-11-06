"""Full-body window rollout for the impact optimiser (SHOW-C3c, #14559).

Adapts the same-input full-body dynamics (``same_input.integrator``) to the
optimiser's injected ``Rollout``: ``torque (steps, n) -> RolloutOutcome``.  One
real v2 rollout is expensive (about 0.3 s per 1 ms step), so the adapter works
on a short *window* (for example top of backswing to impact) restarted from the
reference state at the window start, and controls only a chosen joint subset;
every other actuated joint keeps its baseline effort and the root stays zero.

* ``trajectory`` = joint angles after each step for the controlled joints
  (``(steps, n)``, the TrustRegion reference and ROM/velocity binding input);
* the impact conditions come from an injected ``impact_fn(times, q, v)`` (the
  C2 evaluator on the real model), so this module has no MuJoCo dependency.

Design by Contract:
    Preconditions: controlled joints are non-root coordinates of the plant;
    torque is ``(steps, n)``; baseline efforts are ``(steps, nv)``.
    Postconditions: the root efforts applied are zero; the run is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.trajectory_optimiser import (
    Rollout,
    RolloutOutcome,
)
from src.shared.python.motion_matching.same_input.integrator import (
    DEFAULT_POLICY,
    DynamicsPlant,
    StepPolicy,
    integrate,
    root_indices,
)
from src.shared.python.impact_optimization.impact_target import MeasuredImpact

Array = NDArray[np.float64]
ImpactFn = Callable[[Array, Array, Array], MeasuredImpact]


@dataclass(frozen=True, eq=False)
class RolloutWindow:
    """A replay window: start state, baseline efforts, step and impact reader."""

    plant: DynamicsPlant
    q0: Array
    v0: Array
    baseline_efforts: Array  # (steps, nv) in plant coordinate order
    dt_s: float
    impact_fn: ImpactFn
    policy: StepPolicy = DEFAULT_POLICY

    def __post_init__(self) -> None:
        nv = len(self.plant.coordinate_order)
        require(
            self.q0.shape == (nv,) and self.v0.shape == (nv,), "state must be (nv,)"
        )
        require(
            self.baseline_efforts.ndim == 2 and self.baseline_efforts.shape[1] == nv,
            "baseline_efforts must be (steps, nv)",
        )
        require(self.dt_s > 0.0, "dt_s must be > 0", self.dt_s)

    @property
    def steps(self) -> int:
        return int(self.baseline_efforts.shape[0])

    @property
    def times_s(self) -> Array:
        """Time of the state after each step (rows of ``trajectory``)."""
        return np.arange(1, self.steps + 1, dtype=float) * self.dt_s

    def columns(self, joints: tuple[str, ...]) -> list[int]:
        order = list(self.plant.coordinate_order)
        require(all(j in order for j in joints), f"unknown joints in {joints}")
        root = set(root_indices(self.plant.coordinate_order).tolist())
        cols = [order.index(j) for j in joints]
        require(not root.intersection(cols), "root coordinates are not controllable")
        return cols

    def baseline_torque(self, joints: tuple[str, ...]) -> Array:
        """Baseline effort of the controlled joints, ``(steps, len(joints))``."""
        return np.asarray(self.baseline_efforts[:, self.columns(joints)], dtype=float)


def make_full_body_rollout(window: RolloutWindow, joints: tuple[str, ...]) -> Rollout:
    """Optimiser rollout controlling ``joints`` over ``window``."""
    cols = window.columns(joints)

    def rollout(torque_nm: Array) -> RolloutOutcome:
        tau = np.asarray(torque_nm, dtype=float)
        require(tau.shape == (window.steps, len(cols)), "torque must be (steps, n)")
        table = window.baseline_efforts.copy()
        table[:, cols] = tau
        run = integrate(
            window.plant,
            window.q0,
            window.v0,
            lambda j, _t, _q, _v: table[j],
            steps=window.steps,
            dt_s=window.dt_s,
            policy=window.policy,
        )
        return RolloutOutcome(
            impact=window.impact_fn(run.time_s, run.q, run.v),
            torque_nm=tau,
            trajectory=np.asarray(run.q[1:, cols], dtype=float),
        )

    return rollout


def envelope_subset(
    envelope: PhysiologicalEnvelope, joints: tuple[str, ...]
) -> PhysiologicalEnvelope:
    """Envelope restricted to ``joints`` (in that order), limits unchanged."""
    require(all(j in envelope.names for j in joints), f"unknown joints in {joints}")
    idx = [envelope.names.index(j) for j in joints]
    return PhysiologicalEnvelope(
        names=tuple(joints),
        tau_max_negative_nm=envelope.tau_max_negative_nm[idx].copy(),
        tau_max_positive_nm=envelope.tau_max_positive_nm[idx].copy(),
        rom_deg=envelope.rom_deg[idx].copy(),
        vel_max_deg_s=envelope.vel_max_deg_s[idx].copy(),
        assumed_torque_slots=tuple(
            j for j in joints if j in envelope.assumed_torque_slots
        ),
        assumed_velocity_slots=tuple(
            j for j in joints if j in envelope.assumed_velocity_slots
        ),
        soft_rom_slots=tuple(j for j in joints if j in envelope.soft_rom_slots),
        unbounded_rom_slots=tuple(
            j for j in joints if j in envelope.unbounded_rom_slots
        ),
    )
