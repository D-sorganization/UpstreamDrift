"""Marching the head's real trajectory once, as a whole shot (#8733 §3).

What this is instead of
-----------------------

:meth:`~.solver.PlaneStrainMPMSolver.solve` answers an *instantaneous*
:class:`~bunkershot3d.solvers.protocol.IntrusionState` by building its own
history: the body is reversed along its velocity until clear of the bed
and then driven back to the queried pose **at constant velocity**.  That
is a declared modelling assumption, it is recorded on the verdict, and it
is what makes an F1 answer comparable to F0's memoryless one at all.  It
is not going anywhere -- a tier comparison needs both tiers answering the
same question.

A genuine F1 shot is a different question.  Here the head is placed just
above the sand at its delivered attitude and then **integrated**: the sand
wrench decelerates it, the next pose follows from the last one, and the
wrench history is read off a single continuous solve rather than off a
prescribed approach.  Nothing tells the head where to be.  That is
strictly better and strictly more expensive, and ADR-0033 marks the
timing of peak load quotable-with-tier, which is the quantity that needs
a real trajectory to mean anything.

The shape is :func:`bunkershot3d.solvers.shot.simulate_shot`
---------------------------------------------------------------

Deliberately, down to returning a
:class:`~bunkershot3d.solvers.shot.ShotResult`, so an F1 shot goes into
:mod:`bunkershot3d.metrics` through the same door an F0 shot does.  Three
things differ, and all three are properties of the tier rather than of
this function:

* **The step is the CFL step**, microseconds rather than
  ``ShotSettings.time_step_s``'s quarter-millisecond, because the sand is
  being solved and not evaluated.  The record therefore has thousands of
  samples, and the wrench on any single one of them is step-noisy in the
  way explicit MPM contact always is.
* **The bed is finite and it remembers.**  F0's half space is infinite
  and has no divot; here the head digs a real hole in a bed of stated
  extent, and running off the end of that bed raises rather than quietly
  sweeping vacuum.
* **Rotation is prescribed, translation is free** -- the same
  idealisation :mod:`bunkershot3d.solvers.shot` states, for the same
  reason: the head is on a shaft held by a golfer, so neither free
  precession nor a fixed attitude is right, and the delivered rotation is
  the assumption that can actually be stated.

Where the step boundary is
--------------------------

Within one step the body contacts the sand at ``v^n`` and then moves by
``v^n dt``.  That is what
:meth:`~.body.RigidSection.project_grid_velocity`'s swept test predicted,
so the anti-tunnelling geometry stays exact rather than approximately
right.  The velocity update lands after the move, so the scheme is
``x^{n+1} = x^n + dt v^n``, ``v^{n+1} = v^n + dt a^n`` -- the same
ordering the prescribed-section march uses, which is what keeps the two
paths comparable step for step.

What it still may not be quoted for
------------------------------------

Everything ADR-0033 refuses, unchanged.  A whole-shot march makes the
*timing* trustworthy at tier; it does not make the magnitude quotable,
because the magnitude still depends on ``effective_width_m`` and the
leading edge is still under-resolved.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..elements import SurfaceElements
from ..envelope import GRAVITY_M_S2, ValidityVerdict
from ..exceptions import ShotTruncatedError, SolverInputError
from ..protocol import IntrusionState
from ..shot import ShotResult
from .body import RigidSection
from .grid import PlaneStrainGrid
from .solver import MPMRun, PlaneStrainMPMSolver, cfl_time_step_s
from .state import ParticleState, settled_bed
from .step import StepContext, StepDiagnostics, advance_step

__all__ = [
    "DEFAULT_EJECTA_HEADROOM_CELLS",
    "DEFAULT_TRAVEL_SPANS",
    "F1ShotResult",
    "F1ShotSettings",
    "ShotFieldRecorder",
    "simulate_f1_shot",
]

DEFAULT_EJECTA_HEADROOM_CELLS = 12.0
"""Cells of empty grid above the free surface, when none is declared.

Twice what the declared-approach path allows, because a whole shot throws
sand for milliseconds rather than for the length of one approach. It is
still an *allowance*: sand that goes higher ends the march with the grid's
own "enlarge the domain" refusal rather than being clamped, because a
clamped particle is a silent mass sink."""

DEFAULT_TRAVEL_SPANS = 6.0
"""Bed length ahead of the head, in multiples of its own section span.

A *declared allowance*, not a prediction: how far a head travels while
submerged is one of the things the march is being run to find out, so the
bed cannot be sized from it. Six spans covers every delivery the shipped
presets produce; a head that reaches the end raises, naming this setting,
rather than sweeping empty grid and reporting the resulting zero as a
result."""


@dataclass(frozen=True)
class F1ShotSettings:
    """How one whole-shot F1 march is set up and when it stops.

    Attributes:
        head_mass_kg: Head mass. A wedge head is 290-310 g.
        free_surface_height_m: World ``z`` of the undisturbed sand.
        free_flight_lead_m: How far the head's lowest point starts above
            the free surface. A short lead-in, because every step of it
            is a full MPM step spent watching nothing happen; it exists
            so the record brackets the entry crossing.
        travel_allowance_m: Bed length ahead of the head. ``None`` takes
            :data:`DEFAULT_TRAVEL_SPANS` times the section's own span.
        ejecta_headroom_m: Empty grid above the free surface for thrown
            sand. ``None`` takes
            :data:`DEFAULT_EJECTA_HEADROOM_CELLS` cells.
        max_time_s: Hard stop on the record. 20 ms covers a wedge strike
            with margin and costs nothing when the head exits earlier.
        include_gravity: Whether the head's weight acts during contact.
            Off by default: over a 5 ms contact it contributes about
            0.015 N.s against an impulse of order 5 N.s, and leaving it
            out keeps the shot a test of the sand model.
        require_exit: Whether the march must end with the sole back above
            the free surface, raising if it does not.
        min_speed_m_s: Speed below which a head still in the sand counts
            as stopped. A physical outcome, not a truncation.
        gravity_m_s2: Gravitational acceleration for the head's weight.
    """

    head_mass_kg: float
    free_surface_height_m: float = 0.0
    free_flight_lead_m: float = 0.002
    travel_allowance_m: float | None = None
    ejecta_headroom_m: float | None = None
    max_time_s: float = 0.020
    include_gravity: bool = False
    require_exit: bool = False
    min_speed_m_s: float = 0.05
    gravity_m_s2: float = GRAVITY_M_S2

    def __post_init__(self) -> None:
        positive = {
            "head_mass_kg": self.head_mass_kg,
            "max_time_s": self.max_time_s,
            "gravity_m_s2": self.gravity_m_s2,
            "min_speed_m_s": self.min_speed_m_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise SolverInputError(f"{name} must be positive, got {value!r}")
        if not math.isfinite(self.free_flight_lead_m) or self.free_flight_lead_m < 0.0:
            raise SolverInputError(
                f"free_flight_lead_m must be non-negative, got "
                f"{self.free_flight_lead_m!r}"
            )
        for name, allowance in (
            ("travel_allowance_m", self.travel_allowance_m),
            ("ejecta_headroom_m", self.ejecta_headroom_m),
        ):
            if allowance is not None and (
                not math.isfinite(allowance) or allowance <= 0.0
            ):
                raise SolverInputError(
                    f"{name} must be positive when given, got {allowance!r}"
                )
        if not math.isfinite(self.free_surface_height_m):
            raise SolverInputError("free_surface_height_m must be finite")


@dataclass(frozen=True)
class F1ShotResult:
    """One whole-shot F1 march: the trace, and the solve behind it.

    Attributes:
        shot: The trace, in the shape
            :func:`~bunkershot3d.solvers.shot.simulate_shot` returns, so
            it goes straight into :mod:`bunkershot3d.metrics`.
        run: The single continuous solve the trace was read off, carrying
            the final bed, the divot and every step's diagnostics.
        contacted: Whether the head ever touched the sand at all.
        exited: Whether the record ends with the sole back above the
            surface.
        truncated: Whether the step budget ran out first.
        travel_m: How far the head's reference point moved horizontally.
    """

    shot: ShotResult
    run: MPMRun
    contacted: bool
    exited: bool
    truncated: bool
    travel_m: float

    def peak_force_time_s(self) -> float:
        """When the force magnitude peaked, on the trace's own clock.

        ADR-0033 marks the timing of peak load quotable with tier while
        the magnitude is not, and this is the number that reading it off
        a marched trajectory rather than a declared approach was for.

        Taken off :attr:`shot` rather than off :attr:`run` because the two
        clocks are one step apart by construction: a sample is the pose
        the step was taken **from**, and the run's diagnostic is stamped
        at the end of that step.
        """
        magnitude = np.linalg.norm(self.shot.forces_n, axis=1)
        return float(self.shot.times_s[int(np.argmax(magnitude))])

    def summary(self) -> str:
        """A statement fit for a run manifest."""
        return (
            f"F1 whole-shot march: {self.run.n_steps} steps at "
            f"{self.run.time_step_s * 1e6:.3g} us, "
            f"{'exited' if self.exited else 'no exit'}"
            f"{' (truncated)' if self.truncated else ''}, "
            f"travel {self.travel_m * 1e3:.4g} mm, "
            f"peak load at {self.peak_force_time_s() * 1e3:.4g} ms, "
            f"divot {self.run.divot_depth_m() * 1e3:.4g} mm deep\n"
            + self.shot.summary()
        )


class ShotFieldRecorder(Protocol):
    """Something that wants to look at the bed as the shot marches past.

    The march advances one particle bed in place, so the only moment the
    sand field of a *whole shot* exists is during the loop. Anything that
    wants it -- issue #8729's volumetric view, most immediately -- has to
    be handed it there rather than reconstructing it afterwards from the
    final bed, which is one instant and not a record.

    Deliberately a structural protocol over solver types this module
    already imports, so :mod:`bunkershot3d.fields` can implement it
    without this module importing :mod:`bunkershot3d.fields` back.

    A recorder must not mutate what it is handed. The arrays it sees are
    the live solve; copying is the recorder's job, and
    :func:`~bunkershot3d.fields.shotcapture.capture_f1_shot_field` does
    it by sampling onto the grid rather than by holding a reference.
    """

    def begin(
        self,
        grid: PlaneStrainGrid,
        *,
        time_step_s: float,
        n_steps: int,
    ) -> None:
        """Told the march's shape before the first step.

        Args:
            grid: The lattice the whole march runs on; it does not move.
            time_step_s: The CFL step [s].
            n_steps: The step budget, so a recorder can pick a stride.
        """

    def sample(
        self,
        time_s: float,
        particles: ParticleState,
        section: RigidSection,
    ) -> None:
        """Offered one instant of the march.

        Called before the first step and after every step, so a recorder
        that keeps everything sees the undisturbed bed first. Whether to
        keep any given instant is the recorder's own decision.

        Args:
            time_s: Elapsed time at this instant [s].
            particles: The live particle bed. Not to be retained.
            section: The head's pose at this instant.
        """


@dataclass(slots=True)
class _Columns:
    """Per-step columns of the shot, accumulated as lists.

    Nine growing NumPy arrays would reallocate on every one of a few
    thousand steps; the lists are converted once at the end.
    """

    times_s: list[float] = field(default_factory=list)
    positions_m: list[NDArray[np.float64]] = field(default_factory=list)
    velocities_m_s: list[NDArray[np.float64]] = field(default_factory=list)
    orientations: list[NDArray[np.float64]] = field(default_factory=list)
    forces_n: list[NDArray[np.float64]] = field(default_factory=list)
    torques_n_m: list[NDArray[np.float64]] = field(default_factory=list)
    engaged_depths_m: list[float] = field(default_factory=list)
    sole_depths_m: list[float] = field(default_factory=list)
    active_areas_m2: list[float] = field(default_factory=list)
    inertial_fractions: list[float] = field(default_factory=list)
    steps: list[StepDiagnostics] = field(default_factory=list)


@dataclass(frozen=True)
class _Plan:
    """Everything the march loop needs, assembled once."""

    solver: PlaneStrainMPMSolver
    settings: F1ShotSettings
    context: StepContext
    particles: ParticleState
    head: RigidSection
    extra: tuple[RigidSection, ...]
    bed_x_bounds_m: tuple[float, float]
    out_of_plane_m: float
    spin_rad_s: float
    n_steps: int


def simulate_f1_shot(
    solver: PlaneStrainMPMSolver,
    state: IntrusionState,
    *,
    settings: F1ShotSettings,
    extra_bodies: Sequence[RigidSection] = (),
    recorder: ShotFieldRecorder | None = None,
) -> F1ShotResult:
    """March the head's real trajectory once and read the wrench off it.

    Args:
        solver: The F1 solver whose material, grid spacing and effective
            width the shot is run with.
        state: The **delivery**: the head's surface elements at the
            delivered attitude, its velocity, its prescribed spin and the
            free surface. The pose is used for the attitude and the
            entry, not as a pose to be driven back to -- nothing here
            prescribes where the head goes.
        settings: How the shot is set up and when it stops.
        extra_bodies: Further bodies marched alongside the head -- the
            ball of :mod:`.ball`, for instance. They are rigid and
            prescribed; only the head is integrated.
        recorder: Offered every instant of the march as it happens
            (issue #8729). The bed is advanced in place, so a whole
            shot's sand *field* only exists during the loop; a caller
            that wants it has to be handed it here. Recording changes
            nothing about the trajectory -- the recorder is only shown
            the state, never asked about it.

    Returns:
        The shot, its trace and the solve behind it.

    Raises:
        SolverInputError: If the query is malformed, if the delivery has
            no in-plane velocity, or if the head runs off the end of the
            declared bed.
        OutOfEnvelopeError: If the verdict refuses under a strict policy.
        ShotTruncatedError: If the budget ran out with the sole still in
            the sand and ``settings.require_exit`` is set.
    """
    if not isinstance(settings, F1ShotSettings):
        raise SolverInputError(
            f"settings must be an F1ShotSettings, got {type(settings).__name__}"
        )
    if not isinstance(state.elements, SurfaceElements) or len(state.elements) == 0:
        raise SolverInputError("the delivery has no surface elements")
    verdict = solver.envelope(state)
    verdict.require_usable(solver.refusal_policy)

    plan = _plan(solver, state, settings, tuple(extra_bodies))
    if recorder is not None:
        recorder.begin(
            plan.context.grid,
            time_step_s=plan.context.time_step_s,
            n_steps=plan.n_steps,
        )
    started = time.perf_counter()
    columns, head, extra, exited, truncated = _march(plan, recorder)
    contacted = any(step.n_contacts > 0 for step in columns.steps)
    run = MPMRun(
        steps=tuple(columns.steps),
        time_step_s=plan.context.time_step_s,
        particles=plan.particles,
        grid=plan.context.grid,
        section=head,
        free_surface_height_m=settings.free_surface_height_m,
        bed_x_bounds_m=plan.bed_x_bounds_m,
        extra_sections=extra,
    )
    shot = _to_shot_result(
        columns,
        solver=solver,
        verdict=verdict,
        sole_reference=_sole_offset(plan.head),
        exited=exited,
        started_s=started,
    )
    travel = abs(float(columns.positions_m[-1][0] - columns.positions_m[0][0]))
    result = F1ShotResult(shot, run, contacted, exited, truncated, travel)
    if settings.require_exit and not exited:
        raise ShotTruncatedError(
            _truncation_message(result, settings), result=shot, settings=settings
        )
    return result


def _plan(
    solver: PlaneStrainMPMSolver,
    state: IntrusionState,
    settings: F1ShotSettings,
    extra: tuple[RigidSection, ...],
) -> _Plan:
    """Place the head above the sand, size the bed, and fix the step."""
    delivered = solver.section_from_state(state)
    speed = delivered.speed_m_s
    if speed <= 0.0:
        raise SolverInputError(
            "the delivery has no in-plane velocity, so there is no trajectory "
            "to march. A whole-shot march integrates the head; it does not "
            "place it."
        )
    head = _entry_section(delivered, settings)
    span = max(float(np.ptp(head.vertices_m[:, 0])), solver.cell_size_m)
    travel = (
        DEFAULT_TRAVEL_SPANS * span
        if settings.travel_allowance_m is None
        else float(settings.travel_allowance_m)
    )
    grid, particles, bounds = _build_bed(solver, settings, head, extra, travel)

    fastest = max(
        [head.max_speed_m_s, *(body.max_speed_m_s for body in extra)], default=speed
    )
    step_s = cfl_time_step_s(
        cell_size_m=solver.cell_size_m,
        elastic_wave_speed_m_s=solver.material.elastic_wave_speed_m_s,
        max_material_speed_m_s=fastest,
        cfl_number=solver.cfl_number,
    )
    n_steps = min(int(math.ceil(settings.max_time_s / step_s)), int(solver.max_steps))
    context = StepContext(
        grid=grid,
        material=solver.material,
        node_positions_m=grid.node_positions_m(),
        time_step_s=step_s,
        datum_m=settings.free_surface_height_m - solver.bed_depth_m,
        walls=solver.walls,
        gravity_m_s2=solver.gravity_m_s2,
    )
    return _Plan(
        solver=solver,
        settings=settings,
        context=context,
        particles=particles,
        head=head,
        extra=extra,
        bed_x_bounds_m=bounds,
        out_of_plane_m=float(state.reference_point_m[1]),
        spin_rad_s=head.angular_velocity_rad_s,
        n_steps=max(n_steps, 1),
    )


def _entry_section(delivered: RigidSection, settings: F1ShotSettings) -> RigidSection:
    """Back the head up its own velocity until it is just clear of the sand.

    Analytic rather than searched: the delivery is a straight line at a
    constant velocity while the head is still in the air, so the crossing
    is a division. A level or rising delivery has no crossing to back up
    to, and is placed at its delivered pose instead of being given an
    invented descent.
    """
    lift = (
        settings.free_surface_height_m
        + settings.free_flight_lead_m
        - float(delivered.bounds_m()[0][1])
    )
    descent = -float(delivered.velocity_m_s[1])
    if lift <= 0.0 or descent <= 0.0:
        return delivered
    return delivered.translated(
        (-(lift / descent) * float(delivered.velocity_m_s[0]), lift)
    )


def _build_bed(
    solver: PlaneStrainMPMSolver,
    settings: F1ShotSettings,
    head: RigidSection,
    extra: tuple[RigidSection, ...],
    travel_m: float,
) -> tuple[PlaneStrainGrid, ParticleState, tuple[float, float]]:
    """A bed covering the entry, the declared travel and every extra body."""
    lower, upper = head.bounds_m()
    forward = 1.0 if head.velocity_m_s[0] >= 0.0 else -1.0
    margin = solver.run_in_lengths * max(float(upper[0] - lower[0]), solver.cell_size_m)
    reach = [float(lower[0]) - margin, float(upper[0]) + margin]
    reach.append(float(lower[0] if forward < 0 else upper[0]) + forward * travel_m)
    for body in extra:
        body_lower, body_upper = body.bounds_m()
        reach.extend([float(body_lower[0]) - margin, float(body_upper[0]) + margin])
    bed_lower, bed_upper = min(reach), max(reach)

    surface = settings.free_surface_height_m
    headroom = (
        DEFAULT_EJECTA_HEADROOM_CELLS * solver.cell_size_m
        if settings.ejecta_headroom_m is None
        else float(settings.ejecta_headroom_m)
    )
    ceiling = max(surface + headroom, float(upper[1]))
    grid = PlaneStrainGrid.covering(
        (bed_lower, surface - solver.bed_depth_m),
        (bed_upper, ceiling),
        solver.cell_size_m,
    )
    particles = settled_bed(
        solver.material,
        x_bounds_m=(bed_lower, bed_upper),
        free_surface_height_m=surface,
        depth_m=solver.bed_depth_m,
        cell_size_m=solver.cell_size_m,
        particles_per_cell_axis=solver.particles_per_cell_axis,
        gravity_m_s2=solver.gravity_m_s2,
    )
    return grid, particles, (bed_lower, bed_upper)


def _march(
    plan: _Plan,
    recorder: ShotFieldRecorder | None = None,
) -> tuple[_Columns, RigidSection, tuple[RigidSection, ...], bool, bool]:
    """Integrate the head until it comes out, stops, or the budget ends.

    Three things end the march, and they are not the same thing: the sole
    **crosses back out** of the sand, which is the strike being over; the
    head has effectively **stopped**, which is a physical outcome; or the
    step budget runs out, which is neither and is reported as truncation.

    The strike is bracketed on the sole's own geometry rather than on the
    contact set. A swept-node contact fires while the body is still a
    step's travel *above* the surface, so a march that opened its window
    on "any contact" would close it again on the same step and report a
    strike that never happened.

    Returns:
        ``(columns, head, extra_bodies, exited, truncated)``, the poses
        being the ones the march finished on.
    """
    settings = plan.settings
    width = plan.solver.effective_width_m
    weight = np.array(
        [0.0, -settings.gravity_m_s2 * settings.head_mass_kg], dtype=np.float64
    )
    step_s = plan.context.time_step_s
    columns = _Columns()
    head = plan.head
    extra = plan.extra
    entered = False
    elapsed = 0.0
    # The undisturbed bed, before anything has touched it. It is the
    # reference every later frame is read against: without it an animation
    # opens on sand that is already moving and gives no sense of what the
    # club changed.
    if recorder is not None:
        recorder.sample(0.0, plan.particles, head)

    for _ in range(plan.n_steps):
        elapsed += step_s
        diagnostic = advance_step(
            plan.particles, (head, *extra), plan.context, elapsed_s=elapsed
        )
        sole_depth = settings.free_surface_height_m - float(head.bounds_m()[0][1])
        _record(columns, plan, head, diagnostic, sole_depth, elapsed - step_s)
        if recorder is not None:
            recorder.sample(elapsed, plan.particles, head)

        if entered and sole_depth <= 0.0:
            return columns, head, extra, True, False
        entered = entered or sole_depth > 0.0
        if entered and head.speed_m_s < settings.min_speed_m_s:
            return columns, head, extra, False, False

        force = diagnostic.contact_force_n_per_m * width
        if settings.include_gravity:
            force = force + weight
        velocity = head.velocity_m_s + (step_s / settings.head_mass_kg) * force
        head = head.advanced(step_s).with_velocity(velocity)
        extra = tuple(body.advanced(step_s) for body in extra)
        _require_inside_bed(head, plan)

    return columns, head, extra, False, True


def _record(
    columns: _Columns,
    plan: _Plan,
    head: RigidSection,
    diagnostic: StepDiagnostics,
    sole_depth_m: float,
    time_s: float,
) -> None:
    """Append one sample, at the pose the step was taken *from*."""
    width = plan.solver.effective_width_m
    reference = head.reference_point_m
    stress = diagnostic.stress_force_n_per_m
    total = diagnostic.contact_force_n_per_m
    flux = total - stress
    stress_magnitude = float(np.hypot(stress[0], stress[1]))
    flux_magnitude = float(np.hypot(flux[0], flux[1]))
    denominator = stress_magnitude + flux_magnitude

    columns.steps.append(diagnostic)
    columns.times_s.append(time_s)
    columns.positions_m.append(
        np.array([reference[0], plan.out_of_plane_m, reference[1]])
    )
    columns.velocities_m_s.append(
        np.array([head.velocity_m_s[0], 0.0, head.velocity_m_s[1]])
    )
    columns.orientations.append(_rotation_about_y(plan.spin_rad_s * time_s))
    columns.forces_n.append(np.array([total[0] * width, 0.0, total[1] * width]))
    columns.torques_n_m.append(
        np.array([0.0, diagnostic.contact_torque_n * width, 0.0])
    )
    columns.engaged_depths_m.append(max(sole_depth_m, 0.0))
    columns.sole_depths_m.append(sole_depth_m)
    columns.active_areas_m2.append(
        float(diagnostic.n_contacts) * plan.solver.cell_size_m * width
    )
    columns.inertial_fractions.append(
        0.0 if denominator <= 0.0 else flux_magnitude / denominator
    )


def _require_inside_bed(head: RigidSection, plan: _Plan) -> None:
    """Raise once the head reaches the end of the declared bed.

    A head past the end of the bed sweeps empty grid and returns a wrench
    of zero, which reads exactly like a head that has come out of the
    sand. The two are not the same event, so this is a raise naming the
    setting that fixes it rather than a silent tail of zeros.
    """
    lower, upper = head.bounds_m()
    bed_lower, bed_upper = plan.bed_x_bounds_m
    if float(lower[0]) >= bed_lower and float(upper[0]) <= bed_upper:
        return
    raise SolverInputError(
        f"the head reached the end of the bed at x = "
        f"[{float(lower[0]) * 1e3:.4g}, {float(upper[0]) * 1e3:.4g}] mm, outside "
        f"the declared [{bed_lower * 1e3:.4g}, {bed_upper * 1e3:.4g}] mm. Past "
        "the end it sweeps empty grid and reports a zero wrench, which reads "
        "like an exit; raise F1ShotSettings.travel_allowance_m instead."
    )


def _sole_offset(head: RigidSection) -> NDArray[np.float64]:
    """The lowest point of the section, as a body-frame 3-vector.

    Plane strain has no heel-toe position, so the out-of-plane component
    is the reference point's own -- which is to say the sole reference is
    on the section plane by construction, not by measurement.
    """
    lowest = head.vertices_m[int(np.argmin(head.vertices_m[:, 1]))]
    offset = lowest - head.reference_point_m
    return np.array([offset[0], 0.0, offset[1]], dtype=np.float64)


def _rotation_about_y(angle_rad: float) -> NDArray[np.float64]:
    """The body-to-world rotation after ``angle_rad`` of the prescribed spin.

    The same sign convention :meth:`~.body.RigidSection.advanced` uses, so
    the 3-D orientation on the trace and the 2-D section it was read off
    cannot drift apart.
    """
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _to_shot_result(
    columns: _Columns,
    *,
    solver: PlaneStrainMPMSolver,
    verdict: ValidityVerdict,
    sole_reference: NDArray[np.float64],
    exited: bool,
    started_s: float,
) -> ShotResult:
    """Freeze the columns into the protocol's shot trace."""
    return ShotResult(
        fidelity_tier=solver.fidelity_tier,
        verdict=verdict,
        times_s=np.asarray(columns.times_s, dtype=np.float64),
        positions_m=np.asarray(columns.positions_m, dtype=np.float64).reshape(-1, 3),
        velocities_m_s=np.asarray(columns.velocities_m_s, dtype=np.float64).reshape(
            -1, 3
        ),
        orientations=np.asarray(columns.orientations, dtype=np.float64).reshape(
            -1, 3, 3
        ),
        forces_n=np.asarray(columns.forces_n, dtype=np.float64).reshape(-1, 3),
        torques_n_m=np.asarray(columns.torques_n_m, dtype=np.float64).reshape(-1, 3),
        engaged_depths_m=np.asarray(columns.engaged_depths_m, dtype=np.float64),
        sole_depths_m=np.asarray(columns.sole_depths_m, dtype=np.float64),
        active_areas_m2=np.asarray(columns.active_areas_m2, dtype=np.float64),
        inertial_fractions=np.asarray(columns.inertial_fractions, dtype=np.float64),
        sole_reference_body_m=np.asarray(sole_reference, dtype=np.float64),
        exited=exited,
        runtime_s=time.perf_counter() - started_s,
    )


def _truncation_message(result: F1ShotResult, settings: F1ShotSettings) -> str:
    """Say which setting stopped the shot, and where it had got to."""
    depth_mm = float(result.shot.sole_depths_m[-1]) * 1e3
    reached_s = float(result.shot.times_s[-1])
    return (
        f"the F1 whole-shot march ended with the sole still {depth_mm:.4g} mm "
        f"below the sand after {reached_s * 1e3:.4g} ms of a max_time_s window of "
        f"{settings.max_time_s!r} s, so the strike has no exit crossing; raise "
        "max_time_s, or set require_exit=False if a fixed window is what was "
        "wanted. At the CFL step this tier runs at, a longer window is bought "
        "in whole seconds of wall clock, so it is stated rather than defaulted."
    )
