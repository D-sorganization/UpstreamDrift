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

The effort hold is a separate policy choice (#12279): ``effort_holds`` splits
each step into that many equal holds, each with its own effort sample and
``substeps / effort_holds`` RK4 substeps, while the closure projection stays
once per step.  The default (one hold per 1 ms step) is the original bundle
contract; a finer hold tracks a stiff computed-torque law far more closely.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input.closure import (
    ClosurePlant,
    project_to_closure,
)
from src.shared.python.motion_matching.impact_force import (
    ImpactForce,
    step_through_impact,
)
from src.shared.python.motion_matching.same_input.stage_trace import (
    RunPlan,
    SegmentPlan,
    StageRecorder,
    fixed_substep_sizes,
)

Array = NDArray[np.float64]
EffortSource = Callable[[int, float, Array, Array], Array]
DEFAULT_SUBSTEPS = 8
DEFAULT_EFFORT_HOLDS = 1

ROOT_COORDINATES: tuple[str, ...] = (
    "TranslationInputX",
    "TranslationInputY",
    "TranslationInputZ",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
)


@dataclass(frozen=True)
class StepPolicy:
    """How each fixed step is integrated.

    ``substeps`` RK4 substeps per step, closure projection after each step when
    ``project``, and with ``stop_on_failure`` a diverging run (nonfinite or
    singular dynamics, failed projection) ends early instead of raising.
    ``effort_holds`` effort samples per step, each held over an equal share of
    the substeps (#12279); it must divide ``substeps``.
    """

    substeps: int = DEFAULT_SUBSTEPS
    project: bool = True
    stop_on_failure: bool = False
    effort_holds: int = DEFAULT_EFFORT_HOLDS

    def __post_init__(self) -> None:
        for name in ("substeps", "effort_holds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.substeps % self.effort_holds:
            raise ValueError("effort_holds must divide substeps")

    @property
    def substeps_per_hold(self) -> int:
        """RK4 substeps integrated under one held effort sample."""
        return self.substeps // self.effort_holds


DEFAULT_POLICY = StepPolicy()


def effort_holds_for(
    effort_hold_s: float, dt_s: float, substeps: int = DEFAULT_SUBSTEPS
) -> int:
    """Holds per step for an effort hold of ``effort_hold_s`` (#12279).

    Preconditions: ``effort_hold_s`` and ``dt_s`` are positive and finite,
    ``dt_s / effort_hold_s`` is an integer (to 1e-9 relative) that divides
    ``substeps``.  Raises ``ValueError`` otherwise.
    """
    if not (np.isfinite(effort_hold_s) and effort_hold_s > 0.0):
        raise ValueError("effort_hold_s must be positive and finite")
    if not (np.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("dt_s must be positive and finite")
    ratio = dt_s / effort_hold_s
    holds = int(round(ratio))
    if holds < 1 or abs(ratio - holds) > 1e-9 * ratio:
        raise ValueError(
            f"effort_hold_s {effort_hold_s} must divide the step {dt_s} evenly"
        )
    StepPolicy(substeps=substeps, effort_holds=holds)  # holds must divide substeps
    return holds


class DynamicsPlant(ClosurePlant, Protocol):
    """Plant with constrained accelerations and spec coordinate order."""

    coordinate_order: tuple[str, ...]

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array: ...


@dataclass(frozen=True)
class Rollout:
    """States ``q``/``v`` (steps + 1, nv), held efforts (steps, nv) and drift.

    ``pose_drift`` is the closure pose residual each step left before its
    projection; it measures what the projection removed.  ``failure`` is set
    when ``StepPolicy.stop_on_failure`` ended the run early (states then stop there).
    """

    time_s: Array
    q: Array
    v: Array
    efforts: Array
    pose_drift: Array
    failure: str | None = None
    ball_impact: dict[str, Any] | None = None


def root_indices(coordinate_order: Sequence[str]) -> Array:
    """Indices of the unactuated pelvis root coordinates."""
    missing = [name for name in ROOT_COORDINATES if name not in coordinate_order]
    if missing:
        raise ValueError(f"coordinate order lacks root coordinates {missing}")
    return np.array([list(coordinate_order).index(n) for n in ROOT_COORDINATES])


def zoh_rk4_step(
    plant: DynamicsPlant,
    q: Array,
    v: Array,
    tau: Array,
    dt_s: float,
    *,
    observe: Callable[[int, float, Array, Array], None] | None = None,
    observe_drive: Callable[[Array, Array], None] | None = None,
) -> tuple[Array, Array]:
    """One classical RK4 step with ``tau`` held constant over the step."""

    def evaluate(stage: int, offset: float, state_q: Array, state_v: Array) -> Array:
        if observe is not None:
            observe(stage, offset, state_q, state_v)
        if observe_drive is None:
            return plant.acceleration(state_q, state_v, tau)
        method = getattr(plant, "acceleration_with_drive", None)
        if not callable(method):
            raise TypeError("plant cannot supply ideal drive effort evidence")
        acceleration, drive, external = method(state_q, state_v, tau)
        observe_drive(drive, external)
        return np.asarray(acceleration, dtype=float)

    k1q, k1v = v, evaluate(1, 0.0, q, v)
    q2, v2 = q + dt_s / 2 * k1q, v + dt_s / 2 * k1v
    k2q, k2v = v2, evaluate(2, dt_s / 2, q2, v2)
    q3, v3 = q + dt_s / 2 * k2q, v + dt_s / 2 * k2v
    k3q, k3v = v3, evaluate(3, dt_s / 2, q3, v3)
    q4, v4 = q + dt_s * k3q, v + dt_s * k3v
    k4q, k4v = v4, evaluate(4, dt_s, q4, v4)
    q_next = q + dt_s / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
    v_next = v + dt_s / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    if not (np.isfinite(q_next).all() and np.isfinite(v_next).all()):
        raise FloatingPointError("Nonfinite state after RK4 step")
    return q_next, v_next


class _ForcedPlant:
    """``plant`` with an extra applied generalized force ``external(q)``."""

    def __init__(self, plant: DynamicsPlant, external: Callable[[Array], Array]):
        self._plant = plant
        self._external = external

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array:
        return self._plant.acceleration(q, v, tau + self._external(q))

    def acceleration_with_drive(
        self, q: Array, v: Array, tau: Array
    ) -> tuple[Array, Array, Array]:
        method = getattr(self._plant, "acceleration_with_drive_components", None)
        if not callable(method):
            raise TypeError("plant cannot supply separate ideal drive effort evidence")
        return method(q, v, tau, self._external(q))


def _plant_poses(plant: DynamicsPlant) -> Callable[[Array], Mapping[str, Array]]:
    frames = getattr(plant, "kinematic_frames", None)
    if frames is None:
        raise TypeError("a ball impact needs a plant with kinematic_frames(q)")
    return frames  # type: ignore[no-any-return]


def integrate(
    plant: DynamicsPlant,
    q0: Array,
    v0: Array,
    source: EffortSource,
    *,
    steps: int,
    dt_s: float,
    policy: StepPolicy = DEFAULT_POLICY,
    impact: ImpactForce | None = None,
    recorder: StageRecorder | None = None,
    impact_event_observer: Callable[[float, Array, Array], None] | None = None,
) -> Rollout:
    """Integrate ``steps`` steps, asking ``source(j, t, q, v)`` for each effort.

    Each step holds ``policy.effort_holds`` efforts in turn; hold ``j``
    (global index ``k * effort_holds + h``, time ``j * dt_s / effort_holds``)
    is requested at its start state and held over ``policy.substeps_per_hold``
    RK4 substeps.  With the default single hold, ``j`` is the step index.
    Root entries are forced to zero.  With ``policy.stop_on_failure`` a
    diverging run returns the states reached so far instead of raising.
    ``impact`` adds the ball's collision force over its window (GCV-20): a
    hold containing a window edge is split there, each part keeping the
    substep size.  The closure is projected once at the end of each step.  An
    unlatched plan is latched from this plant's state; a latched one (a
    recorded bundle) is applied as is.

    Postconditions: ``q``/``v`` have ``steps + 1`` rows on the step grid and
    ``efforts`` has ``effort_holds`` rows per completed step.
    """
    if steps < 1 or not (np.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("steps and dt_s must be positive")
    root = root_indices(plant.coordinate_order)
    if recorder is not None:
        recorder.start_run(
            RunPlan(
                steps,
                dt_s,
                policy.substeps,
                policy.effort_holds,
                policy.project,
                tuple(plant.coordinate_order),
            )
        )
    try:
        q, v = np.asarray(q0, dtype=float).copy(), np.asarray(v0, dtype=float).copy()
        qs, vs, efforts, drift = [q.copy()], [v.copy()], [], []
        failure = None
        poses = None if impact is None else _plant_poses(plant)
        for k in range(steps):
            if recorder is not None:
                recorder.begin()
            try:
                q, v, impact, taus, step_drift = _held_step(
                    plant,
                    (q, v),
                    k,
                    dt_s,
                    (source, root, impact, poses),
                    policy,
                    recorder,
                    impact_event_observer,
                )
            except (ArithmeticError, np.linalg.LinAlgError, ValueError) as exc:
                if recorder is not None:
                    recorder.abort()
                if not policy.stop_on_failure:
                    if recorder is not None:
                        recorder.finish(f"step {k}: {type(exc).__name__}: {exc}")
                    raise
                failure = (
                    f"step {k} (t = {k * dt_s:.3f} s): {type(exc).__name__}: {exc}"
                )
                break
            except BaseException:
                if recorder is not None:
                    recorder.abort()
                    recorder.finish(f"step {k}: exception")
                raise
            drift.append(step_drift)
            qs.append(q.copy())
            vs.append(v.copy())
            efforts.extend(taus)
        if recorder is not None:
            recorder.finish(failure)
        return Rollout(
            time_s=np.arange(len(qs), dtype=np.float64) * dt_s,
            q=np.array(qs),
            v=np.array(vs),
            efforts=np.array(efforts).reshape(
                len(efforts), len(plant.coordinate_order)
            ),
            pose_drift=np.array(drift),
            failure=failure,
            ball_impact=None if impact is None else impact.to_record(),
        )
    except BaseException as exc:
        if recorder is not None:
            recorder.abort()
            recorder.finish(f"integration exception: {type(exc).__name__}: {exc}")
        raise


def _held_step(
    plant: DynamicsPlant,
    state: tuple[Array, Array],
    k: int,
    dt_s: float,
    inputs: tuple[EffortSource, Array, ImpactForce | None, Any],
    policy: StepPolicy,
    recorder: StageRecorder | None = None,
    impact_event_observer: Callable[[float, Array, Array], None] | None = None,
) -> tuple[Array, Array, ImpactForce | None, list[Array], float]:
    """One step: ``effort_holds`` held efforts, then the closure projection."""
    source, root, impact, poses = inputs
    q, v = state
    holds = policy.effort_holds
    hold_dt = dt_s / holds
    taus = []
    for h in range(holds):
        j = k * holds + h
        tau = np.asarray(source(j, j * hold_dt, q, v), dtype=float).copy()
        tau[root] = 0.0
        taus.append(tau)
        if impact is None:
            if recorder is not None:
                recorder.segment(
                    SegmentPlan(
                        k,
                        h,
                        0,
                        j * hold_dt,
                        hold_dt,
                        policy.substeps_per_hold,
                        False,
                        (hold_dt / policy.substeps_per_hold,)
                        * policy.substeps_per_hold,
                    )
                )
            for substep in range(policy.substeps_per_hold):
                sub_dt = hold_dt / policy.substeps_per_hold
                observe = (
                    None
                    if recorder is None
                    else recorder.observer(
                        k, h, 0, substep, j * hold_dt + substep * sub_dt, tau
                    )
                )
                q, v = zoh_rk4_step(
                    plant,
                    q,
                    v,
                    tau,
                    sub_dt,
                    observe=observe,
                    observe_drive=(
                        recorder.record_drive
                        if recorder is not None and recorder.capture_ideal_drive
                        else None
                    ),
                )
        else:
            q, v, impact = _impact_hold(
                plant,
                (q, v, tau),
                (j * hold_dt, hold_dt),
                impact,
                poses,
                policy,
                recorder,
                k,
                h,
                impact_event_observer,
            )
    pre_q, pre_v = (q.copy(), v.copy()) if recorder is not None else (q, v)
    if not policy.project:
        drift = float(np.abs(plant.closure_pose_residual(q)).max())
        if recorder is not None:
            recorder.commit(pre_q, pre_v, q, v)
        return q, v, impact, taus, drift
    projected = project_to_closure(plant, q, v)
    if recorder is not None:
        recorder.commit(pre_q, pre_v, projected.q, projected.v)
    return projected.q, projected.v, impact, taus, projected.pose_residual_before


def _impact_hold(
    plant: DynamicsPlant,
    state: tuple[Array, Array, Array],
    clock: tuple[float, float],
    impact: ImpactForce,
    poses: Any,
    policy: StepPolicy,
    recorder: StageRecorder | None = None,
    step_index: int = 0,
    hold_index: int = 0,
    impact_event_observer: Callable[[float, Array, Array], None] | None = None,
) -> tuple[Array, Array, ImpactForce]:
    """One held effort through the ball window (no closure projection)."""
    q, v, tau = state
    t, hold_dt = clock

    segment = 0

    def advance_impl(
        _t: float, h: float, q_a: Array, v_a: Array, external: Any, *, trial: bool
    ) -> tuple:
        nonlocal segment
        sizes = fixed_substep_sizes(
            h, hold_dt / policy.substeps_per_hold, max(0.0, _t - t)
        )
        parts = len(sizes)
        forced = plant if external is None else _ForcedPlant(plant, external)
        if recorder is not None and not trial:
            recorder.segment(
                SegmentPlan(
                    step_index,
                    hold_index,
                    segment,
                    _t,
                    h,
                    parts,
                    external is not None,
                    sizes,
                )
            )
        offset = 0.0
        for substep, sub_dt in enumerate(sizes):
            observe = (
                None
                if recorder is None or trial
                else recorder.observer(
                    step_index,
                    hold_index,
                    segment,
                    substep,
                    _t + offset,
                    tau,
                )
            )
            q_a, v_a = zoh_rk4_step(
                forced,  # type: ignore[arg-type]
                q_a,
                v_a,
                tau,
                sub_dt,
                observe=observe,
                observe_drive=(
                    recorder.record_drive
                    if recorder is not None
                    and not trial
                    and recorder.capture_ideal_drive
                    else None
                ),
            )
            offset += sub_dt
        if not trial:
            segment += 1
        return q_a, v_a, None

    def advance(t_a: float, h: float, q_a: Array, v_a: Array, external: Any) -> tuple:
        return advance_impl(t_a, h, q_a, v_a, external, trial=False)

    q, v, latched, _ = step_through_impact(
        impact,
        t,
        hold_dt,
        q,
        v,
        advance,
        poses=poses,
        accel=lambda qq, vv, f: plant.acceleration(qq, vv, f),
        **(
            {}
            if impact_event_observer is None
            else {"event_observer": impact_event_observer}
        ),
    )
    if latched is None:
        raise ValueError("ball impact plan was lost during the step")
    return q, v, latched


def open_loop(
    plant: DynamicsPlant,
    q0: Array,
    v0: Array,
    efforts: Array,
    *,
    dt_s: float,
    policy: StepPolicy = DEFAULT_POLICY,
    impact: ImpactForce | None = None,
    recorder: StageRecorder | None = None,
) -> Rollout:
    """Replay a fixed effort sequence (steps * effort_holds, nv).

    Root columns must be zero, and the table holds ``policy.effort_holds``
    rows per step (a bundle's ``replay_efforts()`` with its
    ``step_policy()``).  ``impact`` (normally the bundle's recorded, latched
    ball force) is applied identically in every engine.
    """
    table = np.asarray(efforts, dtype=float)
    if table.ndim != 2 or table.shape[1] != len(plant.coordinate_order):
        raise ValueError("efforts must be (steps, nv) in spec order")
    if np.any(table[:, root_indices(plant.coordinate_order)] != 0.0):
        raise ValueError("root efforts must be zero")
    steps, extra = divmod(table.shape[0], policy.effort_holds)
    if extra:
        raise ValueError(
            f"efforts rows {table.shape[0]} are not a whole number of steps "
            f"at effort_holds = {policy.effort_holds}"
        )
    return integrate(
        plant,
        q0,
        v0,
        lambda j, _t, _q, _v: table[j],
        steps=steps,
        dt_s=dt_s,
        policy=policy,
        impact=impact,
        recorder=recorder,
    )
