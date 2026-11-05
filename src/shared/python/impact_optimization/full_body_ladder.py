"""Guarded 100 -> 125 mph ladder on the full-body window (SHOW-C3c, #14559).

Glue between the full-body window rollout (``full_body_rollout``) and
``run_speed_continuation``: it replays the *baseline* torque first and refuses
to start unless that baseline impact is physical, so the ladder can never run
on an unrefit spec by accident (a binding-joint verdict from a non-physical
baseline would name the wrong joint as miscalibrated; supervisor ruling D24).

Design by Contract:
    Preconditions: the baseline impact has face, path and speed available and
    inside ``BaselineBand``; controlled joints are non-root plant coordinates.
    Postconditions: one kinematic (ROM / velocity) binding tuple per rung,
    computed from the rung's own rollout joint angles (radians); no feasible
    rung exceeds a ROM or velocity limit (D29: hard constraints).

The band defaults are assumptions, stated here and overridable: the fitted
driver impact is ~110 mph with ~21 deg open face and ~6.5 deg path, so
100-120 mph, |face| <= 25 deg and |path| <= 15 deg.  Wording of any verdict
built on this run is "within the controlled subset": ``held`` names the
joints kept at their baseline effort.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.full_body_rollout import (
    ImpactFn,
    RolloutWindow,
    envelope_subset,
    make_full_body_rollout,
)
from src.shared.python.impact_optimization.impact_target import (
    ImpactTarget,
    MeasuredImpact,
)
from src.shared.python.impact_optimization.kinematic_binding import (
    KinematicBinding,
    enforce_kinematic_limits,
)
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.speed_continuation import (
    ContinuationResult,
    LadderSpec,
    run_speed_continuation,
)
from src.shared.python.impact_optimization.trajectory_optimiser import OptimiserConfig
from src.shared.python.impact_optimization.trust_region import TrustRegion
from src.shared.python.motion_matching.same_input.integrator import root_indices

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BaselineBand:
    """What counts as a physical baseline impact (assumed bounds, see module)."""

    speed_mph_min: float = 100.0
    speed_mph_max: float = 120.0
    max_abs_face_deg: float = 25.0
    max_abs_path_deg: float = 15.0

    def __post_init__(self) -> None:
        require(
            0.0 < self.speed_mph_min < self.speed_mph_max,
            "need 0 < speed_mph_min < speed_mph_max",
        )
        require(self.max_abs_face_deg > 0.0, "max_abs_face_deg must be > 0")
        require(self.max_abs_path_deg > 0.0, "max_abs_path_deg must be > 0")


def require_physical_baseline(impact: MeasuredImpact, band: BaselineBand) -> None:
    """Fail closed unless the baseline impact is inside ``band``."""
    face, path = impact.face_deg, impact.path_deg
    require(
        face is not None and path is not None,
        "baseline impact: face and path must be available",
    )
    ok = (
        band.speed_mph_min <= impact.speed_mph <= band.speed_mph_max
        and abs(face) <= band.max_abs_face_deg  # type: ignore[arg-type]
        and abs(path) <= band.max_abs_path_deg  # type: ignore[arg-type]
    )
    require(
        ok,
        "baseline impact is not physical (is the spec refit?): "
        f"speed {impact.speed_mph:.1f} mph, face {face}, path {path} deg "
        f"vs band {band}",
    )


def held_joints(window: RolloutWindow, controlled: Sequence[str]) -> tuple[str, ...]:
    """Non-root plant joints that keep their baseline effort, in plant order."""
    order = tuple(window.plant.coordinate_order)
    root = {order[i] for i in root_indices(order).tolist()}
    return tuple(j for j in order if j not in root and j not in set(controlled))


@dataclass(frozen=True)
class LadderRun:
    """A guarded ladder run: baseline, rungs, per-rung kinematic bindings.

    ``result.reached_mph`` counts only rungs inside every ROM and velocity limit.
    """

    baseline: MeasuredImpact
    result: ContinuationResult
    kinematic_by_rung: tuple[tuple[KinematicBinding, ...], ...]
    controlled: tuple[str, ...]
    held: tuple[str, ...]


def run_full_body_ladder(
    window: RolloutWindow,
    joints: tuple[str, ...],
    envelope: PhysiologicalEnvelope,
    *,
    band: BaselineBand,
    trust_radius: float,
    config: OptimiserConfig | None = None,
    ladder: LadderSpec | None = None,
    base_target: ImpactTarget | None = None,
) -> LadderRun:
    """Run the speed ladder over ``joints`` after the baseline-physical check."""
    require(math.isfinite(trust_radius) and trust_radius > 0.0, "bad trust_radius")
    rollout = make_full_body_rollout(window, joints)
    base_tau = window.baseline_torque(joints)
    base = rollout(base_tau)
    require_physical_baseline(base.impact, band)
    env = envelope_subset(envelope, joints)
    trust = TrustRegion(base.trajectory, np.ones(len(joints)), trust_radius)
    raw = run_speed_continuation(
        rollout, base_tau, window.times_s, env, trust, config, base_target, ladder
    )
    result, kin = enforce_kinematic_limits(raw, env, window.times_s, degrees=False)
    return LadderRun(base.impact, result, kin, joints, held_joints(window, joints))


def make_mujoco_impact_fn(
    model: Any,
    joint_ids: Sequence[int],
    body_name: str,
    frame: Any,
    face_spec: Any,
    *,
    evaluate: Callable[..., Any] | None = None,
) -> ImpactFn:
    """C2 impact reader at the last window sample on the real MuJoCo model.

    Scatters plant-order ``(T, n)`` states into MuJoCo ``qpos`` / ``qvel`` by
    joint id (unmapped coordinates stay at ``qpos0``; only hinge and slide
    joints are accepted) and evaluates ``evaluate_rollout`` at ``impact_index = T - 1``.
    """
    from src.shared.python.impact_parameters.conditions import (
        Rollout,
        evaluate_rollout,
    )

    import mujoco

    ids = list(joint_ids)
    scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    require(
        all(int(t) in scalar for t in np.asarray(model.jnt_type)[ids]),
        "controlled joints must be hinge or slide (one qpos / dof each)",
    )
    qadr = np.asarray(model.jnt_qposadr)[ids]
    vadr = np.asarray(model.jnt_dofadr)[ids]
    run: Callable[..., Any] = evaluate or evaluate_rollout

    def impact_fn(times_s: Array, q: Array, v: Array) -> MeasuredImpact:
        require(q.shape[1] == len(ids) == v.shape[1], "state width must match ids")
        qpos = np.tile(np.asarray(model.qpos0, dtype=float), (len(times_s), 1))
        qvel = np.zeros((len(times_s), model.nv))
        qpos[:, qadr] = q
        qvel[:, vadr] = v
        impact: MeasuredImpact = run(
            model,
            Rollout(times_s, qpos, qvel),
            body_name,
            frame,
            spec=face_spec,
            impact_index=len(times_s) - 1,
        )
        return impact

    return impact_fn
