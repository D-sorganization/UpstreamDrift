"""Phase-gated tangent feedback with one auditable actuator boundary (F02).

The controller consumes MOSAIC's precomputed TVLQR gains; it does not fit a
model, integrate a plant, or imply native contact qualification. Call it once
at each integration-step boundary and hold its returned input for that step.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.contact_force_allocator import (
    ContactForceAllocator,
    FeasibilityStatus,
)

Array: TypeAlias = NDArray[np.float64]
CONTROL_CLOCK_ATOL_S = 1e-9


class TaskGroup(str, enum.Enum):
    PELVIS_FEET = "pelvis_feet"
    TRUNK = "trunk"
    ARMS = "arms"
    WRISTS_HANDS = "wrists_hands"
    CLUB = "club"


class TangentModel(Protocol):
    """Engine adapter boundary; no Euclidean nq=nv assumption is inferred."""

    nq: int
    nv: int

    def difference(self, q: Array, reference: Array) -> Array:
        """Current-minus-reference configuration error in tangent order."""
        ...

    def mass_inverse(self, q: Array) -> Array:
        """Positive-definite inverse mass matrix in tangent order."""
        ...


@dataclass(frozen=True)
class EuclideanTangentModel:
    """Explicit unit-mass adapter for Euclidean analytic fixtures only."""

    nv: int
    nq: int = 0

    def __post_init__(self) -> None:
        if self.nv <= 0 or self.nq not in (0, self.nv):
            raise ValueError("Euclidean adapter requires positive nq=nv")
        object.__setattr__(self, "nq", self.nv)

    def difference(self, q: Array, reference: Array) -> Array:
        return q - reference

    def mass_inverse(self, q: Array) -> Array:
        return np.eye(self.nv)


@dataclass(frozen=True)
class ActuatorMap:
    """One-to-one generalized-effort channels; roots are never actuators."""

    channel_ids: tuple[str, ...]
    generalized_indices: tuple[int, ...]
    root_indices: tuple[int, ...]
    transmission_type: str = "unit_constant"

    def __post_init__(self) -> None:
        if (
            not self.channel_ids
            or len(self.channel_ids) != len(self.generalized_indices)
            or len(set(self.channel_ids)) != len(self.channel_ids)
            or len(set(self.generalized_indices)) != len(self.generalized_indices)
            or len(set(self.root_indices)) != len(self.root_indices)
        ):
            raise ValueError("actuator channel and index identities must be unique")
        if any(not name for name in self.channel_ids):
            raise ValueError("actuator channel names must be non-empty")
        if any(index < 0 for index in (*self.generalized_indices, *self.root_indices)):
            raise ValueError("actuator indices must be non-negative")
        if set(self.generalized_indices) & set(self.root_indices):
            raise ValueError("hidden root drive is forbidden")
        if self.transmission_type != "unit_constant":
            raise ValueError("nonunit or state-dependent transmission is unsupported")

    @property
    def nu(self) -> int:
        return len(self.channel_ids)

    def selection(self, nv: int) -> Array:
        if max((*self.generalized_indices, *self.root_indices)) >= nv:
            raise ValueError("actuator or root index outside tangent state")
        result = np.zeros((self.nu, nv))
        result[np.arange(self.nu), self.generalized_indices] = 1.0
        return result


@dataclass(frozen=True)
class NominalTrajectory:
    """Frozen nominal feedforward and MOSAIC TVLQR gains in ordered channels."""

    times: Array
    q: Array
    v: Array
    feedforward: Array
    gains: Array
    phase_names: tuple[str, ...]
    channel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if (
            times.ndim != 1
            or len(times) < 2
            or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError("nominal time grid must be finite and increasing")
        steps = len(times) - 1
        if len(self.phase_names) != steps or any(not name for name in self.phase_names):
            raise ValueError("one named phase is required for each step")
        for name in ("q", "v", "feedforward", "gains"):
            value = np.asarray(getattr(self, name), dtype=float)
            expected_ndim = 3 if name == "gains" else 2
            if value.ndim != expected_ndim or value.shape[0] != steps:
                raise ValueError(f"{name} must have one row per step")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        frozen_times = times.copy()
        frozen_times.setflags(write=False)
        object.__setattr__(self, "times", frozen_times)

    @property
    def steps(self) -> int:
        return len(self.phase_names)

    def index_at(self, time_s: float) -> int:
        if not np.isfinite(time_s) or not self.times[0] <= time_s < self.times[-1]:
            raise ValueError("control time lies outside nominal horizon")
        return int(np.searchsorted(self.times, time_s, side="right") - 1)


@dataclass(frozen=True)
class TaskKinematics:
    position: Array
    velocity: Array
    jacobian: Array


class TaskProvider(Protocol):
    def sample(self, q: Array, v: Array) -> TaskKinematics:
        """Return task position, tangent velocity and Jacobian at exact state."""
        ...

    def difference(self, target: Array, actual: Array) -> Array:
        """Target-minus-actual task error in the provider's tangent chart."""
        ...


@dataclass(frozen=True)
class ControlTask:
    name: str
    group: TaskGroup
    priority: int
    active_phases: tuple[str, ...]
    provider: TaskProvider
    target_positions: Array
    target_velocities: Array
    kp: float
    kd: float
    frame_id: str
    position_unit: str

    def __post_init__(self) -> None:
        if not self.name or not self.frame_id or not self.position_unit:
            raise ValueError("task name, frame and unit are required")
        if self.priority < 0 or not self.active_phases:
            raise ValueError("task priority/phases invalid")
        if not np.isfinite([self.kp, self.kd]).all() or min(self.kp, self.kd) < 0:
            raise ValueError("task gains must be finite and non-negative")
        for name in ("target_positions", "target_velocities"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 2 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite (steps, task_dim)")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.target_positions.shape != self.target_velocities.shape:
            raise ValueError("task target position/velocity shapes differ")


@dataclass(frozen=True)
class AllocationReceipt:
    applied: Array
    contact_mode: str
    root_slack_norm: float
    predicted_ground: Array | None = None
    predicted_grip: Array | None = None


class EffortAllocator(Protocol):
    actuators: ActuatorMap

    def allocate(
        self, requested: Array, previous: Array | None, dt: float, q: Array, v: Array
    ) -> AllocationReceipt:
        """Allocate and bound one control step, rejecting infeasible physics."""
        ...


def _bounded_interval(
    lower: Array, upper: Array, rate: Array, previous: Array | None, dt: float
) -> tuple[Array, Array]:
    if previous is None:
        return lower, upper
    limited_lower = np.maximum(lower, previous - rate * dt)
    limited_upper = np.minimum(upper, previous + rate * dt)
    if np.any(limited_lower > limited_upper):
        raise ValueError("rate and actuator bounds have no feasible intersection")
    return limited_lower, limited_upper


@dataclass(frozen=True)
class DirectBoundedAllocator:
    """For explicitly contact-free fixtures; one saturation/rate boundary."""

    actuators: ActuatorMap
    lower: Array
    upper: Array
    rate: Array
    contact_free: bool

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "rate"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (self.actuators.nu,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must match actuator channels and be finite")
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if np.any(self.lower > self.upper) or np.any(self.rate <= 0):
            raise ValueError("actuator limits/rates invalid")
        if not self.contact_free:
            raise ValueError("direct allocator requires declared contact-free plant")

    def allocate(
        self, requested: Array, previous: Array | None, dt: float, q: Array, v: Array
    ) -> AllocationReceipt:
        if requested.shape != (len(v),) or not np.isfinite(requested).all():
            raise ValueError("generalized request must be finite tangent effort")
        unactuated: Array = np.delete(requested, self.actuators.generalized_indices)
        if np.any(np.abs(unactuated) > 1e-12):
            raise ValueError("direct allocation cannot inject hidden root effort")
        lower, upper = _bounded_interval(
            self.lower, self.upper, self.rate, previous, dt
        )
        desired = requested[list(self.actuators.generalized_indices)]
        return AllocationReceipt(np.clip(desired, lower, upper), "contact_free", 0.0)


class ContactGeometryProvider(Protocol):
    own_contact: bool

    def required_generalized_force(
        self, q: Array, v: Array, requested_actuator_effort: Array
    ) -> Array:
        """Return M*a+h target, including root dynamics, from native dynamics."""
        ...

    def linearization(
        self, q: Array, v: Array
    ) -> tuple[Array, Array, NDArray[np.bool_], Array]:
        """Return ground/grip Jacobians, active-contact mask and ground normal."""
        ...


@dataclass(frozen=True)
class ContactConstrainedAllocator:
    """Single call into the existing contact/grip QP; no post-hoc clipping."""

    actuators: ActuatorMap
    kernel: ContactForceAllocator
    geometry: ContactGeometryProvider
    lower: Array
    upper: Array
    rate: Array
    root_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not self.geometry.own_contact or not callable(
            getattr(self.geometry, "required_generalized_force", None)
        ):
            raise ValueError("native own-contact dynamics provider is required")
        if self.actuators.root_indices != tuple(range(6)):
            raise ValueError(
                "contact allocator requires six explicitly unactuated roots"
            )
        if tuple(self.kernel.actuated_indices) != self.actuators.generalized_indices:
            raise ValueError(
                "allocator actuator permutation differs from channel order"
            )
        if (
            self.kernel.nv <= 6
            or max(self.actuators.generalized_indices) >= self.kernel.nv
        ):
            raise ValueError("contact allocator tangent dimensions differ")
        for name in ("lower", "upper", "rate"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (self.actuators.nu,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must match finite actuator channels")
            frozen = values.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if np.any(self.lower > self.upper) or np.any(self.rate <= 0):
            raise ValueError("contact actuator limits/rates invalid")
        if not np.isfinite(self.root_tolerance) or self.root_tolerance < 0:
            raise ValueError("root tolerance must be finite and nonnegative")

    def allocate(
        self, requested: Array, previous: Array | None, dt: float, q: Array, v: Array
    ) -> AllocationReceipt:
        if requested.shape != (self.kernel.nv,) or not np.isfinite(requested).all():
            raise ValueError("contact request must be finite generalized effort")
        lower, upper = _bounded_interval(
            self.lower, self.upper, self.rate, previous, dt
        )
        j_ground, j_grip, contact_mask, ground_normal = self.geometry.linearization(
            q, v
        )
        dynamics_target = np.asarray(
            self.geometry.required_generalized_force(q, v, requested), dtype=float
        )
        if (
            dynamics_target.shape != (self.kernel.nv,)
            or not np.isfinite(dynamics_target).all()
        ):
            raise ValueError("native dynamics target must be finite generalized force")
        receipt = self.kernel.allocate(
            dynamics_target,
            j_ground,
            j_grip,
            tau_bounds=(lower, upper),
            contact_mask=contact_mask,
            ground_normal=ground_normal,
        )
        if (
            not receipt.success
            or not receipt.is_physically_feasible
            or receipt.feasibility_status != FeasibilityStatus.FEASIBLE
            or receipt.root_slack_norm > self.root_tolerance
            or np.linalg.norm(receipt.delta_tau_root) > self.root_tolerance
        ):
            raise ValueError(
                "contact/grip allocation infeasible or hidden root reserve"
            )
        return AllocationReceipt(
            receipt.tau_actuated.copy(),
            "native_own_contact_qp_prediction",
            receipt.root_slack_norm,
            receipt.f_ground.copy(),
            receipt.lambda_grip.copy(),
        )


@dataclass(frozen=True)
class ControlStep:
    time_s: float
    phase: str
    active_tasks: tuple[str, ...]
    nominal_feedforward: Array
    feedback_correction: Array
    total_requested: Array
    applied: Array
    information_pattern: str
    input_boundary: str
    contact_mode: str
    predicted_ground: Array | None
    predicted_grip: Array | None


class DistributedFeedbackController:
    """Exact-state phase/task feedback; caller owns actual native stepping."""

    def __init__(
        self,
        model: TangentModel,
        actuators: ActuatorMap,
        schedule: NominalTrajectory,
        allocator: EffortAllocator,
        *,
        tasks: Sequence[ControlTask] = (),
        enabled: bool = True,
        information_pattern: str = "exact_simulated_state",
    ) -> None:
        if information_pattern != "exact_simulated_state":
            raise ValueError("only exact simulated-state feedback is implemented")
        if schedule.channel_ids != actuators.channel_ids:
            raise ValueError("nominal input channel order differs from actuator map")
        if allocator.actuators != actuators:
            raise ValueError("allocator actuator map differs from controller")
        if (
            schedule.q.shape != (schedule.steps, model.nq)
            or schedule.v.shape != (schedule.steps, model.nv)
            or schedule.feedforward.shape != (schedule.steps, actuators.nu)
            or schedule.gains.shape != (schedule.steps, actuators.nu, 2 * model.nv)
        ):
            raise ValueError("nominal state, input or tangent gain dimensions differ")
        self.model = model
        self.actuators = actuators
        self.schedule = schedule
        self.allocator = allocator
        self.enabled = enabled
        self._selection = actuators.selection(model.nv)
        self._previous: Array | None = None
        self._next_time: float = float(schedule.times[0])
        self.tasks = tuple(sorted(tasks, key=lambda task: (task.priority, task.name)))
        if len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("task names must be unique")
        for task in self.tasks:
            if not callable(getattr(task.provider, "sample", None)) or not callable(
                getattr(task.provider, "difference", None)
            ):
                raise ValueError(
                    "task adapter lacks sample/manifold-difference capability"
                )
            if task.target_positions.shape[0] != schedule.steps:
                raise ValueError("task target horizon differs from nominal horizon")
            if not set(task.active_phases) <= set(schedule.phase_names):
                raise ValueError("task phase not present in nominal schedule")

    def _task_feedback(
        self, index: int, q: Array, v: Array, mass_inverse: Array
    ) -> tuple[Array, Array, tuple[str, ...]]:
        projection = np.eye(self.actuators.nu)
        total = np.zeros(self.actuators.nu)
        active = tuple(
            task
            for task in self.tasks
            if self.schedule.phase_names[index] in task.active_phases
        )
        for priority in sorted({task.priority for task in active}):
            group = tuple(task for task in active if task.priority == priority)
            requested = np.zeros(self.actuators.nu)
            influences = []
            for task in group:
                measured = task.provider.sample(q, v)
                m = task.target_positions.shape[1]
                if (
                    measured.position.shape != (m,)
                    or measured.velocity.shape != (m,)
                    or measured.jacobian.shape != (m, self.model.nv)
                    or not np.isfinite(measured.position).all()
                    or not np.isfinite(measured.velocity).all()
                    or not np.isfinite(measured.jacobian).all()
                ):
                    raise ValueError(
                        f"task {task.name} adapter returned invalid kinematics"
                    )
                position_error = np.asarray(
                    task.provider.difference(
                        task.target_positions[index], measured.position
                    ),
                    dtype=float,
                )
                if (
                    position_error.shape != (m,)
                    or not np.isfinite(position_error).all()
                ):
                    raise ValueError(
                        f"task {task.name} returned invalid manifold error"
                    )
                velocity_error = task.target_velocities[index] - measured.velocity
                wrench = task.kp * position_error + task.kd * velocity_error
                requested += self._selection @ measured.jacobian.T @ wrench
                influences.append(measured.jacobian @ mass_inverse @ self._selection.T)
            total += projection @ requested
            influence = np.vstack(influences) @ projection
            projection = projection @ (
                np.eye(self.actuators.nu) - np.linalg.pinv(influence) @ influence
            )
        return total, projection, tuple(task.name for task in active)

    def command_for_step(
        self, time_s: float, q: Array, v: Array, dt: float
    ) -> ControlStep:
        """Sample once; total and applied channels remain separate evidence."""
        index = self.schedule.index_at(time_s)
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        if (
            q.shape != (self.model.nq,)
            or v.shape != (self.model.nv,)
            or not np.isfinite(q).all()
            or not np.isfinite(v).all()
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("controller requires finite full state and positive step")
        if abs(time_s - self._next_time) > CONTROL_CLOCK_ATOL_S:
            raise ValueError("controller steps must be contiguous from nominal start")
        if time_s + dt > self.schedule.times[index + 1] + CONTROL_CLOCK_ATOL_S:
            raise ValueError("held control step crosses a nominal policy boundary")
        if self.enabled:
            q_error = np.asarray(
                self.model.difference(q, self.schedule.q[index]), dtype=float
            )
            inverse_mass = np.asarray(self.model.mass_inverse(q), dtype=float)
            if q_error.shape != (self.model.nv,) or not np.isfinite(q_error).all():
                raise ValueError("manifold difference must be finite nv tangent")
            if (
                inverse_mass.shape != (self.model.nv, self.model.nv)
                or not np.isfinite(inverse_mass).all()
            ):
                raise ValueError("inverse mass must be finite tangent matrix")
            if not np.allclose(inverse_mass, inverse_mass.T, rtol=1e-8, atol=1e-10):
                raise ValueError("inverse mass must be symmetric positive definite")
            try:
                np.linalg.cholesky(inverse_mass)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "inverse mass must be symmetric positive definite"
                ) from exc
            state_error = np.concatenate((q_error, v - self.schedule.v[index]))
            task_effort, projector, active = self._task_feedback(
                index, q, v, inverse_mass
            )
            feedback = (
                task_effort - projector @ self.schedule.gains[index] @ state_error
            )
        else:
            feedback = np.zeros(self.actuators.nu)
            active = ()
        nominal = self.schedule.feedforward[index].copy()
        total = nominal + feedback
        generalized = self._selection.T @ total
        allocation = self.allocator.allocate(generalized, self._previous, dt, q, v)
        applied = np.asarray(allocation.applied, dtype=float)
        if applied.shape != (self.actuators.nu,) or not np.isfinite(applied).all():
            raise ValueError("allocation produced invalid applied input")
        self._previous = applied.copy()
        self._next_time = float(time_s + dt)
        return ControlStep(
            time_s=float(time_s),
            phase=self.schedule.phase_names[index],
            active_tasks=active,
            nominal_feedforward=nominal,
            feedback_correction=feedback.copy(),
            total_requested=total,
            applied=applied.copy(),
            information_pattern="exact_simulated_state",
            input_boundary="actuator_torque",
            contact_mode=allocation.contact_mode,
            predicted_ground=allocation.predicted_ground,
            predicted_grip=allocation.predicted_grip,
        )
