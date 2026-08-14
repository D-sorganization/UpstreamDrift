"""Time-march a clubhead through the bed with an F0 solver (issue #8611).

One *shot* is the unit of work a design sweep buys: entry, submerged
travel, exit, and the impulse the sand took out of the head.  The
acceptance criterion is that it runs in **under 50 ms**, so a 1000-point
design of experiments is minutes rather than weeks.

Idealisation, stated up front
-----------------------------

* **Translation is free, rotation is prescribed.**  The head decelerates
  under the sand wrench and gravity; its angular velocity is held at the
  delivered value.  A free rigid-body rotation would be *less* honest,
  not more: the head is on a shaft held by a golfer, so neither free
  precession nor a fixed attitude is right, and prescribing the delivered
  rotation is the assumption that can actually be stated.
* **The bed is a flat half space.**  There is no divot memory, no crater,
  and no free-surface evolution -- three of the standing caveats that
  travel with every verdict.
* **The verdict for a shot is the worst verdict over its steps.**  A shot
  that was answerable for 99 steps and refused for one is refused.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elements import SurfaceElements
from .envelope import ValidityVerdict, worst_of
from .exceptions import SolverInputError
from .protocol import (
    FidelityTier,
    GranularSolver,
    IntrusionState,
    SolverResult,
    Wrench,
)

__all__ = ["HeadKinematics", "ShotResult", "ShotSettings", "simulate_shot"]

_MIN_SPEED_M_S = 1e-6


def _vector(name: str, value: ArrayLike) -> NDArray[np.float64]:
    """Coerce to a finite ``(3,)`` float array."""
    array = np.array(value, dtype=np.float64, copy=True).reshape(-1)
    if array.shape != (3,):
        raise SolverInputError(f"{name} must be a 3-vector, got {np.shape(value)!r}")
    if not np.all(np.isfinite(array)):
        raise SolverInputError(f"{name} contains non-finite values: {array!r}")
    return array


@dataclass(frozen=True)
class HeadKinematics:
    """How the head is presented to the sand at the start of the shot.

    Attributes:
        velocity_m_s: Linear velocity of the body-frame origin.
        position_m: World position of the body-frame origin.
        orientation: ``(3, 3)`` body-to-world rotation.
        angular_velocity_rad_s: Prescribed angular velocity, world frame.
    """

    velocity_m_s: NDArray[np.float64]
    position_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    orientation: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    angular_velocity_rad_s: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "velocity_m_s", _vector("velocity_m_s", self.velocity_m_s)
        )
        object.__setattr__(self, "position_m", _vector("position_m", self.position_m))
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _vector("angular_velocity_rad_s", self.angular_velocity_rad_s),
        )
        matrix = np.array(self.orientation, dtype=np.float64, copy=True)
        if matrix.shape != (3, 3):
            raise SolverInputError(f"orientation must be (3, 3), got {matrix.shape}")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10):
            raise SolverInputError("orientation is not a rotation matrix")
        object.__setattr__(self, "orientation", matrix)

    @property
    def speed_m_s(self) -> float:
        """Magnitude of the entry velocity."""
        return float(np.linalg.norm(self.velocity_m_s))


@dataclass(frozen=True)
class ShotSettings:
    """Integration settings for one shot.

    Attributes:
        time_step_s: Fixed step. 2.5e-4 s resolves the ~5-10 ms of
            submerged travel into 20-40 steps, which fits the 50 ms
            per-shot budget with room to spare.
        max_time_s: Hard stop. 10 ms is longer than any bunker contact;
            the loop normally exits on the head leaving the sand.
        free_surface_height_m: World ``z`` of the undisturbed sand.
        gravity_m_s2: Gravitational acceleration.
        include_gravity: Whether the head's weight acts during contact.
            Off by default: over a 5 ms contact gravity contributes about
            0.015 N.s against an impulse of order 5 N.s, and leaving it
            out keeps the shot a pure test of the sand model.
        start_at_first_contact: Drop the head so its lowest element
            begins exactly at the free surface, so no steps are spent in
            free flight. Deterministic, and what makes the 50 ms budget
            an honest measure of the solver rather than of the approach.
    """

    time_step_s: float = 2.5e-4
    max_time_s: float = 0.010
    free_surface_height_m: float = 0.0
    gravity_m_s2: float = 9.81
    include_gravity: bool = False
    start_at_first_contact: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("time_step_s", self.time_step_s),
            ("max_time_s", self.max_time_s),
            ("gravity_m_s2", self.gravity_m_s2),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise SolverInputError(f"{name} must be positive, got {value!r}")
        if self.time_step_s > self.max_time_s:
            raise SolverInputError(
                f"time_step_s {self.time_step_s} exceeds max_time_s {self.max_time_s}"
            )
        if not math.isfinite(self.free_surface_height_m):
            raise SolverInputError("free_surface_height_m must be finite")

    @property
    def max_steps(self) -> int:
        """Number of steps the settings allow."""
        return int(math.ceil(self.max_time_s / self.time_step_s))


@dataclass(frozen=True)
class ShotResult:
    """The trace of one shot, with its tier and verdict.

    All arrays are contiguous and share the leading axis, per ADR-0032
    structural decision 6 (no group-per-timestep).

    Attributes:
        fidelity_tier: Which tier produced the trace.
        verdict: The worst verdict over the trace.
        times_s: ``(n,)`` sample times from entry.
        positions_m: ``(n, 3)`` body-origin positions.
        velocities_m_s: ``(n, 3)`` velocities.
        forces_n: ``(n, 3)`` sand force on the head.
        torques_n_m: ``(n, 3)`` sand torque about the body origin.
        depths_m: ``(n,)`` deepest submerged point, positive downward.
        active_areas_m2: ``(n,)`` engaged surface area.
        inertial_fractions: ``(n,)`` share of force from the dynamic term.
        runtime_s: Wall-clock time the integration took.
    """

    fidelity_tier: FidelityTier
    verdict: ValidityVerdict
    times_s: NDArray[np.float64]
    positions_m: NDArray[np.float64]
    velocities_m_s: NDArray[np.float64]
    forces_n: NDArray[np.float64]
    torques_n_m: NDArray[np.float64]
    depths_m: NDArray[np.float64]
    active_areas_m2: NDArray[np.float64]
    inertial_fractions: NDArray[np.float64]
    runtime_s: float

    @property
    def n_steps(self) -> int:
        """Number of samples in the trace."""
        return int(self.times_s.shape[0])

    @property
    def peak_force_n(self) -> float:
        """Largest resultant force magnitude over the trace."""
        if self.n_steps == 0:
            return 0.0
        return float(np.linalg.norm(self.forces_n, axis=1).max())

    @property
    def impulse_n_s(self) -> NDArray[np.float64]:
        """Time integral of the sand force, trapezoidal, ``(3,)``."""
        if self.n_steps < 2:
            return np.zeros(3, dtype=np.float64)
        integrated = np.trapezoid(self.forces_n, x=self.times_s, axis=0)
        return np.asarray(integrated, dtype=np.float64)

    @property
    def entry_speed_m_s(self) -> float:
        """Speed at the first sample."""
        return float(np.linalg.norm(self.velocities_m_s[0])) if self.n_steps else 0.0

    @property
    def exit_speed_m_s(self) -> float:
        """Speed at the last sample."""
        return float(np.linalg.norm(self.velocities_m_s[-1])) if self.n_steps else 0.0

    @property
    def max_depth_m(self) -> float:
        """Deepest submerged point reached."""
        return float(self.depths_m.max()) if self.n_steps else 0.0

    @property
    def contact_duration_s(self) -> float:
        """Total time with at least one engaged element."""
        if self.n_steps < 2:
            return 0.0
        engaged = self.active_areas_m2 > 0.0
        if not engaged.any():
            return 0.0
        step = float(np.diff(self.times_s).mean())
        return float(engaged.sum()) * step

    def summary(self) -> str:
        """A statement fit for a run manifest."""
        return (
            f"tier={self.fidelity_tier.value} steps={self.n_steps} "
            f"peak={self.peak_force_n:.4g} N "
            f"impulse={np.linalg.norm(self.impulse_n_s):.4g} N.s "
            f"entry={self.entry_speed_m_s:.4g} m/s "
            f"exit={self.exit_speed_m_s:.4g} m/s "
            f"max depth={self.max_depth_m * 1e3:.4g} mm "
            f"in {self.runtime_s * 1e3:.3g} ms\n" + self.verdict.summary()
        )


def _rotation_increment(
    angular_velocity_rad_s: NDArray[np.float64], time_step_s: float
) -> NDArray[np.float64]:
    """Rodrigues exponential map for a constant angular velocity step."""
    angle = float(np.linalg.norm(angular_velocity_rad_s)) * time_step_s
    if angle <= 0.0:
        return np.eye(3, dtype=np.float64)
    axis = angular_velocity_rad_s / np.linalg.norm(angular_velocity_rad_s)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def _drop_to_surface(
    elements: SurfaceElements,
    kinematics: HeadKinematics,
    free_surface_height_m: float,
) -> NDArray[np.float64]:
    """Return the position at which the lowest element touches the surface."""
    world = elements.transformed(rotation=kinematics.orientation)
    lowest = float(world.centroids_m[:, 2].min())
    position = kinematics.position_m.copy()
    position[2] = free_surface_height_m - lowest
    return position


@dataclass(slots=True)
class _Trace:
    """The per-step columns of one shot, accumulated as Python lists.

    Held as lists and converted once at the end: growing nine NumPy
    arrays per step would reallocate on every step of a run whose whole
    budget is 50 ms.
    """

    times_s: list[float] = field(default_factory=list)
    positions_m: list[NDArray[np.float64]] = field(default_factory=list)
    velocities_m_s: list[NDArray[np.float64]] = field(default_factory=list)
    forces_n: list[NDArray[np.float64]] = field(default_factory=list)
    torques_n_m: list[NDArray[np.float64]] = field(default_factory=list)
    depths_m: list[float] = field(default_factory=list)
    active_areas_m2: list[float] = field(default_factory=list)
    inertial_fractions: list[float] = field(default_factory=list)
    verdicts: list[ValidityVerdict] = field(default_factory=list)

    def record(
        self,
        time_s: float,
        position_m: NDArray[np.float64],
        velocity_m_s: NDArray[np.float64],
        result: SolverResult,
    ) -> None:
        """Append one sample. The pose arrays are copied, not aliased."""
        self.verdicts.append(result.verdict)
        self.times_s.append(time_s)
        self.positions_m.append(position_m.copy())
        self.velocities_m_s.append(velocity_m_s.copy())
        self.forces_n.append(result.wrench.force_n.copy())
        self.torques_n_m.append(result.wrench.torque_n_m.copy())
        self.depths_m.append(result.max_depth_m)
        self.active_areas_m2.append(result.active_area_m2)
        self.inertial_fractions.append(result.inertial_fraction)

    def to_result(self, *, fidelity_tier: FidelityTier, started_s: float) -> ShotResult:
        """Freeze the columns into a :class:`ShotResult`.

        Args:
            fidelity_tier: The tier that produced the trace.
            started_s: ``time.perf_counter()`` reading taken before the
                march, against which the runtime is measured.

        Returns:
            The immutable trace, carrying the worst verdict over it.
        """
        return ShotResult(
            fidelity_tier=fidelity_tier,
            verdict=worst_of(self.verdicts),
            times_s=np.asarray(self.times_s, dtype=np.float64),
            positions_m=np.asarray(self.positions_m, dtype=np.float64).reshape(-1, 3),
            velocities_m_s=np.asarray(self.velocities_m_s, dtype=np.float64).reshape(
                -1, 3
            ),
            forces_n=np.asarray(self.forces_n, dtype=np.float64).reshape(-1, 3),
            torques_n_m=np.asarray(self.torques_n_m, dtype=np.float64).reshape(-1, 3),
            depths_m=np.asarray(self.depths_m, dtype=np.float64),
            active_areas_m2=np.asarray(self.active_areas_m2, dtype=np.float64),
            inertial_fractions=np.asarray(self.inertial_fractions, dtype=np.float64),
            runtime_s=time.perf_counter() - started_s,
        )


def _validated_shot_inputs(
    elements_body: SurfaceElements,
    kinematics: HeadKinematics,
    head_mass_kg: float,
    settings: ShotSettings | None,
) -> tuple[float, ShotSettings]:
    """Check the arguments of one shot and resolve the default settings.

    Returns:
        ``(mass_kg, settings)``.

    Raises:
        SolverInputError: If an argument is malformed.
    """
    if not isinstance(elements_body, SurfaceElements):
        raise SolverInputError(
            f"elements_body must be a SurfaceElements, got "
            f"{type(elements_body).__name__}"
        )
    if not isinstance(kinematics, HeadKinematics):
        raise SolverInputError(
            f"kinematics must be a HeadKinematics, got {type(kinematics).__name__}"
        )
    mass = float(head_mass_kg)
    if not math.isfinite(mass) or mass <= 0.0:
        raise SolverInputError(f"head_mass_kg must be positive, got {head_mass_kg!r}")
    config = ShotSettings() if settings is None else settings
    if not isinstance(config, ShotSettings):
        raise SolverInputError(
            f"settings must be a ShotSettings, got {type(config).__name__}"
        )
    return mass, config


def _march(
    solver: GranularSolver,
    elements_body: SurfaceElements,
    *,
    mass_kg: float,
    kinematics: HeadKinematics,
    config: ShotSettings,
) -> _Trace:
    """Step the head until it leaves the sand, stops, or runs out of time.

    The loop exits on the first step after contact with nothing engaged,
    or once the head has effectively stopped; otherwise it runs to
    ``config.max_steps``.

    Args:
        solver: Any solver implementing the ``GranularSolver`` protocol.
        elements_body: Surface discretisation in the body frame.
        mass_kg: Head mass.
        kinematics: Entry pose and velocity.
        config: Integration settings.

    Returns:
        The accumulated trace.

    Raises:
        OutOfEnvelopeError: If the solver refuses any step under a strict
            refusal policy.
    """
    orientation = kinematics.orientation.copy()
    position = (
        _drop_to_surface(elements_body, kinematics, config.free_surface_height_m)
        if config.start_at_first_contact
        else kinematics.position_m.copy()
    )
    velocity = kinematics.velocity_m_s.copy()
    rotating = bool(kinematics.angular_velocity_rad_s.any())
    increment = (
        _rotation_increment(kinematics.angular_velocity_rad_s, config.time_step_s)
        if rotating
        else None
    )
    # A non-rotating head is oriented once; only the translation changes
    # per step, and translation cannot invalidate a normal or an area.
    oriented = elements_body.transformed(rotation=orientation)
    weight = np.array([0.0, 0.0, -config.gravity_m_s2 * mass_kg], dtype=np.float64)

    trace = _Trace()
    contacted = False
    for step in range(config.max_steps + 1):
        world = oriented.translated(position)
        state = IntrusionState(
            world,
            velocity,
            angular_velocity_rad_s=kinematics.angular_velocity_rad_s,
            reference_point_m=position,
            free_surface_height_m=config.free_surface_height_m,
        )
        result = solver.solve(state)
        trace.record(step * config.time_step_s, position, velocity, result)

        engaged = result.n_active_elements > 0
        contacted = contacted or engaged
        if contacted and not engaged:
            break
        speed = float(np.linalg.norm(velocity))
        if contacted and speed < _MIN_SPEED_M_S:
            break

        total = result.wrench.force_n + (weight if config.include_gravity else 0.0)
        velocity = velocity + (config.time_step_s / mass_kg) * total
        position = position + config.time_step_s * velocity
        if increment is not None:
            orientation = increment @ orientation
            oriented = elements_body.transformed(rotation=orientation)

    return trace


def simulate_shot(
    solver: GranularSolver,
    elements_body: SurfaceElements,
    *,
    head_mass_kg: float,
    kinematics: HeadKinematics,
    settings: ShotSettings | None = None,
) -> ShotResult:
    """March a rigid head through the bed and return the trace.

    Args:
        solver: Any solver implementing the ``GranularSolver`` protocol.
        elements_body: Surface discretisation in the body frame.
        head_mass_kg: Head mass. A wedge head is 290-310 g.
        kinematics: Entry pose and velocity.
        settings: Integration settings; defaults are tuned so a shot
            costs a few milliseconds.

    Returns:
        The trace, its fidelity tier and the worst verdict over it.

    Raises:
        SolverInputError: If an argument is malformed.
        OutOfEnvelopeError: If the solver refuses any step under a strict
            refusal policy.
    """
    mass, config = _validated_shot_inputs(
        elements_body, kinematics, head_mass_kg, settings
    )

    started = time.perf_counter()
    trace = _march(
        solver,
        elements_body,
        mass_kg=mass,
        kinematics=kinematics,
        config=config,
    )
    return trace.to_result(fidelity_tier=solver.fidelity_tier, started_s=started)


def zero_wrench_at(position_m: ArrayLike) -> Wrench:
    """Return the null wrench about ``position_m``.

    A convenience for callers assembling a trace by hand in tests.
    """
    return Wrench.zero(position_m)
