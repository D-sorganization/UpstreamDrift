"""The F1 solver: 2-D plane-strain MPM implementing ``GranularSolver``.

One step, in the order it happens
---------------------------------

1. **P2G.**  Mass and APIC momentum onto the grid.
2. **Stress.**  Cauchy stress from each particle's elastic deformation
   gradient through the Hencky law, scattered as an internal force.
3. **Grid update.**  Symplectic Euler: ``v* = v + dt (f/m + g)``.
4. **Boundaries.**  Domain walls, then the club's collision projection,
   which returns the exact momentum ledger that becomes the wrench.
5. **G2P.**  Velocity, the APIC affine matrix, and the velocity gradient.
6. **Plasticity.**  ``F <- (I + dt grad v) F``, then the capped
   Drucker-Prager return map.
7. **Advection**, then the particle-level pushout backstop.

Why ``solve`` marches instead of evaluating
-------------------------------------------

:class:`~bunkershot3d.solvers.protocol.IntrusionState` is an
*instantaneous* query: a pose, a velocity, and a free surface.  F0 can
answer it directly because its constitutive shortcut is memoryless -- the
force depends only on the current depth and speed.  A continuum has a
history, and there is no such thing as "the stress at this pose" without
one.

F1 therefore supplies the history explicitly and says what it assumed:
the body is **reversed along its own velocity direction until it is clear
of the bed, then driven back to the queried pose at constant velocity**.
That straight-line constant-speed approach is a modelling assumption, it
is recorded in the result's verdict, and it is what makes an F1 answer
comparable to an F0 one at all.

What this solver may not be quoted for
--------------------------------------

ADR-0033 is explicit: F1 runs at bulk resolution (``dx ~ 1-2 mm``) for
10-100 mm flow features, the ~0.5 mm leading edge is deliberately
under-resolved, and **absolute club force stays F0's**.  The wrench
returned here exists so the two tiers can be cross-checked on shape and
timing; its magnitude additionally depends on ``effective_width_m``,
which is a declared assumption rather than a result and is therefore a
**required** constructor argument with no default.
:mod:`bunkershot3d.solvers.mpm.envelope` carries both facts onto every
verdict.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import ensure

from ..drft import DEFAULT_FEATURE_SCALES_M
from ..elements import SurfaceElements
from ..envelope import GRAVITY_M_S2, RefusalPolicy, ValidityVerdict
from ..exceptions import SolverInputError
from ..protocol import FidelityTier, IntrusionState, SolverResult, Wrench
from .body import ContactImpulse, RigidSection
from .constitutive import (
    SandContinuum,
    hencky_kirchhoff_principal,
    principal_stretches,
    reconstruct,
)
from .envelope import evaluate_f1_envelope
from .grid import (
    PlaneStrainGrid,
    affine_from_grid_velocity,
    gather_velocity,
    scatter_mass,
    scatter_momentum,
    scatter_stress_force,
    velocity_gradient,
)
from .state import (
    DomainWalls,
    ParticleState,
    SurfaceDepression,
    apply_wall_conditions,
    settled_bed,
    surface_depression,
    surface_profile_m,
)

__all__ = [
    "DEFAULT_CFL_NUMBER",
    "MPMRun",
    "PlaneStrainMPMSolver",
    "StepDiagnostics",
    "cfl_time_step_s",
]

DEFAULT_CFL_NUMBER = 0.4
"""Courant number on the elastic wave speed.

Symplectic Euler on an explicit MPM is stable to roughly ``C < 1`` for
the elastic wave alone; 0.4 leaves margin for the plastic correction and
for the club's own motion, which enters the same condition."""

_DIMENSION = 2
_MASS_FLOOR_KG = 1e-15
_MIN_APPROACH_CLEARANCE_CELLS = 2.0
_SURFACE_BINS = 64

_NO_CONTACT = ContactImpulse(
    node_index=np.zeros(0, dtype=np.int64),
    impulse_n_s=np.zeros((0, _DIMENSION)),
    position_m=np.zeros((0, _DIMENSION)),
    stress_force_n=np.zeros(_DIMENSION),
    n_swept=0,
)
"""The ledger of a step with no intruder at all.

A bed marched without a club is a *closed* system, which is the only
configuration whose momentum budget is an identity rather than a balance
against a boundary -- so it is what the conservation cases run on."""


def cfl_time_step_s(
    *,
    cell_size_m: float,
    elastic_wave_speed_m_s: float,
    max_material_speed_m_s: float,
    cfl_number: float = DEFAULT_CFL_NUMBER,
) -> float:
    """Return the CFL-limited timestep, computed rather than pinned.

    ``dt = C dx / (c_p + v_max)`` with ``c_p = sqrt((lambda + 2 mu) / rho)``
    the dilatational wave speed of the *material* -- so a stiffer sand
    shortens the step by itself and no constant needs revisiting -- and
    ``v_max`` the fastest material or body speed in the problem, which is
    what stops the club from crossing more than a fraction of a cell in
    one step.

    Args:
        cell_size_m: Grid ``dx``.
        elastic_wave_speed_m_s: ``c_p``.
        max_material_speed_m_s: Fastest particle or body speed.
        cfl_number: Courant number.

    Returns:
        The timestep in seconds.

    Raises:
        SolverInputError: If any input is unusable, or if the resulting
            step is not positive and finite. This is a ``raise`` and not
            an ``assert``: ``python -O`` strips assertions, and a CFL
            condition that evaporates under an optimisation flag is worse
            than none at all.
    """
    size = float(cell_size_m)
    wave = float(elastic_wave_speed_m_s)
    speed = float(max_material_speed_m_s)
    number = float(cfl_number)
    if not math.isfinite(size) or size <= 0.0:
        raise SolverInputError(f"cell_size_m must be positive, got {cell_size_m!r}")
    if not math.isfinite(wave) or wave <= 0.0:
        raise SolverInputError(
            f"elastic wave speed must be positive, got {elastic_wave_speed_m_s!r}"
        )
    if not math.isfinite(speed) or speed < 0.0:
        raise SolverInputError(
            f"max material speed must be finite and non-negative, got "
            f"{max_material_speed_m_s!r}"
        )
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise SolverInputError(f"cfl_number must lie in (0, 1], got {cfl_number!r}")
    step = number * size / (wave + speed)
    if not math.isfinite(step) or step <= 0.0:
        raise SolverInputError(
            f"the CFL condition produced an unusable timestep {step!r} s"
        )
    return step


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """What one step did, kept so the run is inspectable rather than opaque.

    Attributes:
        time_s: Simulation time at the end of the step.
        contact_force_n_per_m: ``(2,)`` total in-plane force on the body,
            per unit out-of-plane width.
        stress_force_n_per_m: ``(2,)`` the stress-and-weight part of that
            force. See :meth:`MPMRun.force_split` for what the split
            means and, just as importantly, what it does not.
        contact_torque_n: Torque on the body about ``+y``, per unit width.
        n_contacts: Grid nodes the club projected.
        n_swept: Of those, nodes reached only by the swept test.
        n_pushed_out: Particles the backstop had to reposition.
        n_yielded: Particles the return map moved.
        n_capped: Particles that hit the compressive cap.
        kinetic_energy_j_per_m: Translational kinetic energy.
        elastic_energy_j_per_m: Stored Hencky strain energy.
        gravitational_energy_j_per_m: Potential energy above the floor.
        linear_momentum_kg_m_s: ``(2,)`` total particle momentum.
        total_mass_kg_per_m: Total particle mass. Invariant.
    """

    time_s: float
    contact_force_n_per_m: NDArray[np.float64]
    stress_force_n_per_m: NDArray[np.float64]
    contact_torque_n: float
    n_contacts: int
    n_swept: int
    n_pushed_out: int
    n_yielded: int
    n_capped: int
    kinetic_energy_j_per_m: float
    elastic_energy_j_per_m: float
    gravitational_energy_j_per_m: float
    linear_momentum_kg_m_s: NDArray[np.float64]
    total_mass_kg_per_m: float


@dataclass(frozen=True)
class MPMRun:
    """The whole trace of one F1 march, and what can be read off it.

    Attributes:
        steps: One :class:`StepDiagnostics` per step, in order.
        time_step_s: The CFL step the march used.
        particles: Final particle state.
        grid: The background grid.
        section: Final club pose, or ``None`` for a bed marched with no
            intruder at all -- which is what the conservation cases need,
            since a closed system is the only one whose momentum budget
            is an identity rather than a balance against a boundary.
        free_surface_height_m: The undisturbed surface the run started
            from.
        bed_x_bounds_m: Horizontal extent of the bed.
    """

    steps: tuple[StepDiagnostics, ...]
    time_step_s: float
    particles: ParticleState
    grid: PlaneStrainGrid
    section: RigidSection | None
    free_surface_height_m: float
    bed_x_bounds_m: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.steps:
            raise SolverInputError(
                "an F1 run with no steps has nothing to report; a zero-step run "
                "would return a zero wrench that reads as a result"
            )

    @property
    def n_steps(self) -> int:
        """Number of steps marched."""
        return len(self.steps)

    @property
    def duration_s(self) -> float:
        """Simulation time spanned."""
        return self.steps[-1].time_s

    def contact_force_history_n_per_m(self) -> NDArray[np.float64]:
        """``(n_steps, 2)`` in-plane force on the body over the run."""
        return np.array([step.contact_force_n_per_m for step in self.steps])

    def time_history_s(self) -> NDArray[np.float64]:
        """``(n_steps,)`` step end times."""
        return np.array([step.time_s for step in self.steps])

    def peak_force_time_s(self) -> float:
        """When the force magnitude peaked.

        ADR-0033 marks the timing of peak load quotable with tier while
        the magnitude is not, so it is reported separately rather than
        inferred by a caller from a history it has to scan itself.
        """
        magnitude = np.linalg.norm(self.contact_force_history_n_per_m(), axis=1)
        return float(self.steps[int(np.argmax(magnitude))].time_s)

    def averaged_force_n_per_m(self, window_s: float) -> NDArray[np.float64]:
        """Mean in-plane force over the final ``window_s`` of the run.

        Explicit MPM contact is step-noisy -- a node enters or leaves the
        projected set discretely -- so a single step's force is not the
        quantity anybody means.  The window is an argument rather than a
        constant so that a caller cannot be given an average whose length
        it did not choose.

        Args:
            window_s: Length of the trailing window.

        Returns:
            ``(2,)`` mean force per unit width.

        Raises:
            SolverInputError: If the window is not positive.
        """
        if not math.isfinite(window_s) or window_s <= 0.0:
            raise SolverInputError(f"window_s must be positive, got {window_s!r}")
        times = self.time_history_s()
        selected = times >= (times[-1] - window_s)
        return self.contact_force_history_n_per_m()[selected].mean(axis=0)

    def averaged_stress_force_n_per_m(self, window_s: float) -> NDArray[np.float64]:
        """Mean stress-and-weight part of the force over the same window."""
        if not math.isfinite(window_s) or window_s <= 0.0:
            raise SolverInputError(f"window_s must be positive, got {window_s!r}")
        times = self.time_history_s()
        selected = times >= (times[-1] - window_s)
        history = np.array([step.stress_force_n_per_m for step in self.steps])
        return history[selected].mean(axis=0)

    def force_split(
        self, window_s: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Split the reaction into its stress-borne and momentum-flux parts.

        The two add to the total exactly, by construction:

        * the **stress** part is ``sum_i (f_i^internal + m_i g)`` over the
          projected nodes -- what the sand's own stress and weight were
          already delivering there;
        * the **momentum-flux** part is the remainder of the contact
          impulse, which is the momentum needed to bring that sand up to
          the club's velocity.

        This is *not* F0's ``alpha |z~|`` versus ``lambda rho v_n^2``
        decomposition and must not be read as one.  It is the analogous
        physical partition -- quasi-static resistance against ram
        pressure -- computed from an exact momentum ledger rather than
        from a fitted form, which is what makes the two tiers'
        depth/inertia crossovers comparable at all.

        Args:
            window_s: Length of the trailing averaging window.

        Returns:
            ``(stress_part, momentum_flux_part)``, both ``(2,)``.
        """
        total = self.averaged_force_n_per_m(window_s)
        stress = self.averaged_stress_force_n_per_m(window_s)
        return stress, total - stress

    def surface_depression(self, *, n_bins: int = _SURFACE_BINS) -> SurfaceDepression:
        """The divot the *sand* carries: how deep, and how much section.

        Read off the particles, because in MPM the free surface *is*
        wherever the particles stop.  The depth alone was enough for the
        cross-check ADR-0033 shipped; comparing against F0's divot needs
        the area as well, because
        :class:`~bunkershot3d.metrics.divot.DivotMetrics` reports a
        section area and a mass and there is nothing to compare a depth
        against there.

        Args:
            n_bins: Horizontal bins across the bed.

        Returns:
            The measurement, carrying its own empty-bin count so a caller
            can see whether the area is complete or a lower bound.

        Raises:
            SolverInputError: If no bin holds a particle.
        """
        return surface_depression(
            self.particles,
            free_surface_height_m=self.free_surface_height_m,
            x_bounds_m=self.bed_x_bounds_m,
            n_bins=n_bins,
        )

    def divot_depth_m(self, *, n_bins: int = _SURFACE_BINS) -> float:
        """Deepest depression of the free surface below its original level.

        Read off the particles, because in MPM the free surface *is*
        wherever the particles stop.  Bins the sand has left entirely are
        skipped rather than treated as infinitely deep.

        Args:
            n_bins: Horizontal bins across the bed.

        Returns:
            Depth in metres, non-negative.
        """
        _, heights = surface_profile_m(
            self.particles, x_bounds_m=self.bed_x_bounds_m, n_bins=n_bins
        )
        populated = heights[np.isfinite(heights)]
        if populated.size == 0:
            return 0.0
        return max(float(self.free_surface_height_m - populated.min()), 0.0)

    def max_pushed_out(self) -> int:
        """Worst single step's pushout count, the anti-tunnelling telltale."""
        return max(step.n_pushed_out for step in self.steps)


@dataclass(frozen=True)
class PlaneStrainMPMSolver:
    """The F1 tier: 2-D plane-strain MPM, swappable with the F0 solver.

    Attributes:
        material: The continuum, derived from a
            :class:`~bunkershot3d.sand.state.SandState`.
        cell_size_m: Grid ``dx``. ADR-0033 specifies bulk resolution,
            1-2 mm; finer does not make the tier quotable for club force,
            it only makes it slower.
        effective_width_m: The out-of-plane width a per-unit-width
            plane-strain load is multiplied by to become a force. **A
            modelling assumption, not a result**, which is why it has no
            default: ADR-0033 requires it to be recorded before any
            magnitude comparison against F0.
        bed_depth_m: Depth of sand below the free surface.
        run_in_lengths: Bed length ahead of and behind the body, in
            multiples of the body's own horizontal extent.
        particles_per_cell_axis: MPM quadrature density.
        cfl_number: Courant number for the timestep.
        gravity_m_s2: Gravitational acceleration.
        contact_friction: Club-on-sand Coulomb friction.
        walls: Domain wall conditions.
        refusal_policy: What happens on a refused verdict.
        feature_scales_m: Scales the envelope is judged at.
        max_steps: Hard cap on the march.
        averaging_window_s: Trailing window the reported wrench is
            averaged over.
    """

    material: SandContinuum
    cell_size_m: float
    effective_width_m: float
    bed_depth_m: float = 0.10
    run_in_lengths: float = 1.5
    particles_per_cell_axis: int = 2
    cfl_number: float = DEFAULT_CFL_NUMBER
    gravity_m_s2: float = GRAVITY_M_S2
    contact_friction: float = 0.3
    walls: DomainWalls = field(default_factory=DomainWalls)
    refusal_policy: RefusalPolicy = RefusalPolicy.STRICT
    feature_scales_m: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_SCALES_M)
    )
    max_steps: int = 20000
    averaging_window_s: float = 2.0e-4

    def __post_init__(self) -> None:
        if not isinstance(self.material, SandContinuum):
            raise SolverInputError(
                f"material must be a SandContinuum, got {type(self.material).__name__}"
            )
        positive = {
            "cell_size_m": self.cell_size_m,
            "effective_width_m": self.effective_width_m,
            "bed_depth_m": self.bed_depth_m,
            "gravity_m_s2": self.gravity_m_s2,
            "averaging_window_s": self.averaging_window_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise SolverInputError(f"{name} must be positive, got {value!r}")
        if self.run_in_lengths < 0.0 or not math.isfinite(self.run_in_lengths):
            raise SolverInputError(
                f"run_in_lengths must be non-negative, got {self.run_in_lengths!r}"
            )
        if int(self.max_steps) < 1:
            raise SolverInputError(
                f"max_steps must be positive, got {self.max_steps!r}"
            )
        if not self.feature_scales_m:
            raise SolverInputError("at least one feature scale is required")
        object.__setattr__(self, "feature_scales_m", dict(self.feature_scales_m))

    # ------------------------------------------------------------ protocol

    @property
    def fidelity_tier(self) -> FidelityTier:
        """Always :attr:`~bunkershot3d.solvers.protocol.FidelityTier.F1`."""
        return FidelityTier.F1

    def envelope(self, state: IntrusionState) -> ValidityVerdict:
        """Judge a query without marching a single step.

        Args:
            state: The intrusion query.

        Returns:
            The F1 verdict. Refusal is a value here and becomes an
            exception only in :meth:`solve`, under the refusal policy.
        """
        self._require_state(state)
        velocities = state.element_velocities_m_s()
        speeds = np.sqrt(np.einsum("ij,ij->i", velocities, velocities))
        depths = -state.element_depths_m()
        submerged = depths > 0.0
        return self._verdict(
            speed_m_s=float(speeds.max()) if speeds.size else 0.0,
            submerged_depth_m=(
                float(depths[submerged].max()) if bool(submerged.any()) else 0.0
            ),
        )

    def solve(self, state: IntrusionState) -> SolverResult:
        """March the sand up to the queried pose and report the wrench.

        Args:
            state: The intrusion query.

        Returns:
            The wrench about ``state.reference_point_m``, the F1 tier, and
            the validity verdict -- always all three.

        Raises:
            OutOfEnvelopeError: If the verdict refuses and the policy is
                strict.
            SolverInputError: If the query is malformed, or if the query
                has no in-plane motion for the approach to be built from.
        """
        self._require_state(state)
        verdict = self.envelope(state)
        verdict.require_usable(self.refusal_policy)
        run = self.run(state)
        return self._result(state, run, verdict)

    # --------------------------------------------------------------- march

    def run(self, state: IntrusionState) -> MPMRun:
        """Build the bed and the approach, then march to the queried pose.

        Exposed separately from :meth:`solve` because the verification
        suite needs the whole trace -- conservation residuals, the energy
        history, the free surface -- and not only the resultant.

        Args:
            state: The intrusion query.

        Returns:
            The run.
        """
        self._require_state(state)
        section, approach_distance_m = self._approach(state)
        grid, particles, bounds = self._build_bed(state, section, approach_distance_m)
        step_s = self.time_step_s(state)
        n_steps = self._approach_steps(state, approach_distance_m, step_s)
        return self.march(
            particles,
            section,
            grid,
            n_steps=n_steps,
            time_step_s=step_s,
            free_surface_height_m=state.free_surface_height_m,
            bed_x_bounds_m=bounds,
        )

    def time_step_s(self, state: IntrusionState) -> float:
        """The CFL step for this query, computed and checked at runtime."""
        velocities = state.element_velocities_m_s()
        speeds = np.sqrt(np.einsum("ij,ij->i", velocities, velocities))
        return cfl_time_step_s(
            cell_size_m=self.cell_size_m,
            elastic_wave_speed_m_s=self.material.elastic_wave_speed_m_s,
            max_material_speed_m_s=float(speeds.max()) if speeds.size else 0.0,
            cfl_number=self.cfl_number,
        )

    def march(
        self,
        particles: ParticleState,
        section: RigidSection | None,
        grid: PlaneStrainGrid,
        *,
        n_steps: int,
        time_step_s: float,
        free_surface_height_m: float,
        bed_x_bounds_m: tuple[float, float],
        damping_per_step: float = 0.0,
    ) -> MPMRun:
        """Integrate ``n_steps`` of the scheme and return the trace.

        Args:
            particles: The bed, advanced in place.
            section: The club at its starting pose.
            grid: The background grid.
            n_steps: How many steps to take.
            time_step_s: The step, normally from :meth:`time_step_s`.
            free_surface_height_m: The undisturbed surface, for the divot.
            bed_x_bounds_m: Horizontal extent of the bed.
            damping_per_step: Fraction of the nodal velocity removed each
                step. Zero for a shot; the verification suite uses it to
                relax a column to static equilibrium, which is the only
                honest way to compare against a *static* analytic answer.

        Returns:
            The run.

        Raises:
            SolverInputError: If the step count exceeds ``max_steps``, or
                if the body would cross more than one cell in a step --
                the runtime assertion of the CFL condition, which is a
                ``raise`` so that ``python -O`` cannot remove it.
        """
        steps = int(n_steps)
        if steps < 1:
            raise SolverInputError(f"n_steps must be positive, got {n_steps!r}")
        if steps > int(self.max_steps):
            raise SolverInputError(
                f"the requested march is {steps} steps, over the {self.max_steps} "
                "cap; raise max_steps deliberately rather than letting a run grow "
                "without bound"
            )
        if not math.isfinite(damping_per_step) or not 0.0 <= damping_per_step < 1.0:
            raise SolverInputError(
                f"damping_per_step must lie in [0, 1), got {damping_per_step!r}"
            )
        self._require_courant(section, time_step_s)

        datum_m = float(free_surface_height_m) - self.bed_depth_m
        node_positions = grid.node_positions_m()
        diagnostics: list[StepDiagnostics] = []
        moving = section
        elapsed = 0.0
        for _ in range(steps):
            elapsed += time_step_s
            report = self._advance(
                particles,
                moving,
                grid,
                node_positions,
                time_step_s=time_step_s,
                elapsed_s=elapsed,
                datum_m=datum_m,
                damping_per_step=damping_per_step,
            )
            diagnostics.append(report)
            if moving is not None:
                moving = moving.advanced(time_step_s)

        return MPMRun(
            steps=tuple(diagnostics),
            time_step_s=float(time_step_s),
            particles=particles,
            grid=grid,
            section=moving,
            free_surface_height_m=float(free_surface_height_m),
            bed_x_bounds_m=bed_x_bounds_m,
        )

    # ------------------------------------------------------------ one step

    def _advance(
        self,
        particles: ParticleState,
        section: RigidSection | None,
        grid: PlaneStrainGrid,
        node_positions: NDArray[np.float64],
        *,
        time_step_s: float,
        elapsed_s: float,
        datum_m: float,
        damping_per_step: float,
    ) -> StepDiagnostics:
        """One full MPM step, mutating ``particles`` in place."""
        stencil = grid.interpolate(particles.position_m)
        nodal_mass = scatter_mass(grid, stencil, particles.mass_kg)
        nodal_momentum = scatter_momentum(
            grid,
            stencil,
            particles.mass_kg,
            particles.velocity_m_s,
            particles.affine,
        )
        live = nodal_mass > _MASS_FLOOR_KG
        node_velocity = np.zeros_like(nodal_momentum)
        node_velocity[live] = nodal_momentum[live] / nodal_mass[live, None]

        stress, elastic_energy = self._cauchy_stress(particles)
        internal_force = scatter_stress_force(
            grid, stencil, self._current_volume(particles), stress
        )
        weight = np.zeros_like(internal_force)
        weight[:, 1] = -nodal_mass * self.gravity_m_s2
        applied_force = internal_force + weight

        updated = node_velocity.copy()
        updated[live] += time_step_s * applied_force[live] / nodal_mass[live, None]
        if damping_per_step > 0.0:
            updated *= 1.0 - damping_per_step
        apply_wall_conditions(grid, updated, self.walls)
        impulse = _NO_CONTACT
        if section is not None:
            updated, impulse = section.project_grid_velocity(
                node_positions,
                updated,
                nodal_mass,
                time_step_s=time_step_s,
                stress_force_n=applied_force,
            )

        particles.velocity_m_s = gather_velocity(stencil, updated)
        particles.affine = affine_from_grid_velocity(grid, stencil, updated)
        gradient = velocity_gradient(stencil, updated)
        identity = np.eye(_DIMENSION)
        trial = (identity + time_step_s * gradient) @ particles.deformation_gradient

        left, stretches, right = principal_stretches(trial)
        projected, yielded, capped = self.material.project(np.log(stretches))
        particles.deformation_gradient = reconstruct(left, projected, right)

        particles.position_m = (
            particles.position_m + time_step_s * particles.velocity_m_s
        )
        pushed = 0
        if section is not None:
            particles.position_m, particles.velocity_m_s, pushed = section.push_out(
                particles.position_m, particles.velocity_m_s
            )

        reference = (
            np.zeros(_DIMENSION) if section is None else section.reference_point_m
        )
        return StepDiagnostics(
            time_s=float(elapsed_s),
            contact_force_n_per_m=impulse.force_on_body_n(time_step_s),
            stress_force_n_per_m=-impulse.stress_force_n,
            contact_torque_n=impulse.torque_on_body_n_m(time_step_s, reference),
            n_contacts=impulse.n_contacts,
            n_swept=impulse.n_swept,
            n_pushed_out=int(pushed),
            n_yielded=int(yielded.sum()),
            n_capped=int(capped.sum()),
            kinetic_energy_j_per_m=particles.kinetic_energy_j(),
            elastic_energy_j_per_m=float(elastic_energy),
            gravitational_energy_j_per_m=particles.gravitational_energy_j(
                self.gravity_m_s2, datum_m
            ),
            linear_momentum_kg_m_s=particles.linear_momentum_kg_m_s(),
            total_mass_kg_per_m=particles.total_mass_kg,
        )

    def _cauchy_stress(
        self, particles: ParticleState
    ) -> tuple[NDArray[np.float64], float]:
        """Cauchy stress and stored energy from the elastic deformation.

        For an isotropic model the Kirchhoff stress is coaxial with the
        *left* stretch, so ``tau = U diag(tau_i) U^T`` and
        ``sigma = tau / J``.  The stored energy is the Hencky strain
        energy ``mu ||eps||^2 + (lambda / 2) tr(eps)^2``, which the
        conservation suite needs to close the energy budget.
        """
        left, stretches, _ = principal_stretches(particles.deformation_gradient)
        strain = np.log(stretches)
        kirchhoff = hencky_kirchhoff_principal(
            strain,
            shear_modulus_pa=self.material.shear_modulus_pa,
            lame_lambda_pa=self.material.lame_lambda_pa,
        )
        jacobian = stretches.prod(axis=1)
        principal = kirchhoff / jacobian[:, None]
        stress = np.einsum("nik,nk,njk->nij", left, principal, left)

        trace = strain.sum(axis=1)
        density = (
            self.material.shear_modulus_pa * np.einsum("ij,ij->i", strain, strain)
            + 0.5 * self.material.lame_lambda_pa * trace**2
        )
        energy = float((particles.initial_volume_m2 * density).sum())
        return stress, energy

    @staticmethod
    def _current_volume(particles: ParticleState) -> NDArray[np.float64]:
        """``V = J V_0``, the deformed particle area."""
        jacobian = np.linalg.det(particles.deformation_gradient)
        return particles.initial_volume_m2 * jacobian

    # ------------------------------------------------------------- set-up

    def _require_state(self, state: IntrusionState) -> None:
        """Precondition: ``state`` is a usable intrusion query."""
        if not isinstance(state, IntrusionState):
            raise SolverInputError(
                f"expected an IntrusionState, got {type(state).__name__}"
            )
        if not isinstance(state.elements, SurfaceElements):
            raise SolverInputError(
                "intrusion state must carry a SurfaceElements structure of arrays"
            )
        if len(state.elements) == 0:
            raise SolverInputError("the intruder has no surface elements")

    def _require_courant(
        self, section: RigidSection | None, time_step_s: float
    ) -> None:
        """Runtime assertion of the CFL condition, as a ``raise``."""
        if section is None:
            return
        travel = section.max_speed_m_s * float(time_step_s)
        if travel > self.cell_size_m:
            raise SolverInputError(
                f"the club would cross {travel * 1e3:.4g} mm in one step on a "
                f"{self.cell_size_m * 1e3:.4g} mm grid, so sand could pass through "
                "it. Lower cfl_number or refine the grid; this is checked with a "
                "raise rather than an assert because python -O strips assertions."
            )

    def section_from_state(self, state: IntrusionState) -> RigidSection:
        """Project the 3-D body onto the swing plane as a convex section.

        The element centroids are dropped onto ``(x, z)`` and hulled.
        Heel-to-toe structure is discarded here, visibly, because plane
        strain has nowhere to put it.

        Args:
            state: The intrusion query.

        Returns:
            The section, carrying the body's in-plane velocity.
        """
        centroids = state.elements.centroids_m
        points = np.stack([centroids[:, 0], centroids[:, 2]], axis=1)
        velocity = np.array(
            [state.velocity_m_s[0], state.velocity_m_s[2]], dtype=np.float64
        )
        reference = np.array(
            [state.reference_point_m[0], state.reference_point_m[2]], dtype=np.float64
        )
        return RigidSection.from_points(
            points,
            velocity_m_s=velocity,
            angular_velocity_rad_s=float(state.angular_velocity_rad_s[1]),
            reference_point_m=reference,
            friction=self.contact_friction,
        )

    def _approach(self, state: IntrusionState) -> tuple[RigidSection, float]:
        """Reverse the body along its velocity until it is clear of the bed.

        Returns:
            ``(starting_section, approach_distance_m)``.

        Raises:
            SolverInputError: If the body has no in-plane speed, so there
                is no approach direction to reverse along. F1 has no
                answer for a static intruder and says so rather than
                inventing a history.
        """
        section = self.section_from_state(state)
        speed = section.speed_m_s
        if speed <= 0.0:
            raise SolverInputError(
                "the query has no in-plane velocity, so F1 has no approach "
                "direction to build the deformation history from. A continuum "
                "force depends on how the body got there; there is no "
                "instantaneous answer to return."
            )
        direction = section.velocity_m_s / speed
        lower, upper = section.bounds_m()
        clearance = _MIN_APPROACH_CLEARANCE_CELLS * self.cell_size_m
        if direction[1] < 0.0:
            # Descending: back off until the lowest point clears the surface.
            distance = (state.free_surface_height_m + clearance - lower[1]) / (
                -direction[1]
            )
        else:
            # Level or rising: there is no height to back off to, so run in
            # horizontally from beyond the body's own length.
            span = float(upper[0] - lower[0])
            distance = (span + clearance) / max(abs(direction[0]), 1e-9)
        distance = max(float(distance), clearance)
        return section.translated(-distance * direction), distance

    def _approach_steps(
        self, state: IntrusionState, approach_distance_m: float, time_step_s: float
    ) -> int:
        """How many steps the approach takes at the queried speed."""
        section = self.section_from_state(state)
        steps = int(math.ceil(approach_distance_m / (section.speed_m_s * time_step_s)))
        return max(steps, 1)

    def _build_bed(
        self,
        state: IntrusionState,
        starting_section: RigidSection,
        approach_distance_m: float,
    ) -> tuple[PlaneStrainGrid, ParticleState, tuple[float, float]]:
        """Build a grid and a settled bed wide enough for the whole approach."""
        final = self.section_from_state(state)
        lower_start, upper_start = starting_section.bounds_m()
        lower_end, upper_end = final.bounds_m()
        span = float(max(upper_end[0] - lower_end[0], self.cell_size_m))
        margin = self.run_in_lengths * span
        bed_lower = float(min(lower_start[0], lower_end[0])) - margin
        bed_upper = float(max(upper_start[0], upper_end[0])) + margin
        surface = state.free_surface_height_m
        floor = surface - self.bed_depth_m

        grid = PlaneStrainGrid.covering(
            (bed_lower, min(floor, float(min(lower_start[1], lower_end[1])))),
            (bed_upper, max(surface, float(max(upper_start[1], upper_end[1])))),
            self.cell_size_m,
        )
        particles = settled_bed(
            self.material,
            x_bounds_m=(bed_lower, bed_upper),
            free_surface_height_m=surface,
            depth_m=self.bed_depth_m,
            cell_size_m=self.cell_size_m,
            particles_per_cell_axis=self.particles_per_cell_axis,
            gravity_m_s2=self.gravity_m_s2,
        )
        _ = approach_distance_m
        return grid, particles, (bed_lower, bed_upper)

    # -------------------------------------------------------------- result

    def _verdict(
        self, *, speed_m_s: float, submerged_depth_m: float
    ) -> ValidityVerdict:
        """Assemble the F1 validity verdict for one query."""
        return evaluate_f1_envelope(
            speed_m_s=speed_m_s,
            feature_lengths_m=self.feature_scales_m,
            grain_diameter_m=self.material.grain_diameter_m,
            cell_size_m=self.cell_size_m,
            submerged_depth_m=submerged_depth_m,
            effective_width_m=self.effective_width_m,
            gravity_m_s2=self.gravity_m_s2,
        )

    def _result(
        self, state: IntrusionState, run: MPMRun, verdict: ValidityVerdict
    ) -> SolverResult:
        """Turn a march into the protocol's :class:`SolverResult`."""
        window = min(self.averaging_window_s, run.duration_s)
        stress_part, flux_part = run.force_split(window)
        total = stress_part + flux_part
        width = self.effective_width_m

        force = np.array([total[0] * width, 0.0, total[1] * width])
        torque_y = float(
            np.mean(
                [step.contact_torque_n for step in run.steps[-_torque_window(run) :]]
            )
        )
        torque = np.array([0.0, torque_y * width, 0.0])
        wrench = Wrench(force, torque, state.reference_point_m)
        ensure(
            bool(np.isfinite(wrench.force_n).all()),
            "F1 produced a non-finite resultant force",
            value=wrench.force_n,
        )

        contacts = [step.n_contacts for step in run.steps]
        depths = -state.element_depths_m()
        return SolverResult(
            wrench=wrench,
            fidelity_tier=self.fidelity_tier,
            verdict=verdict,
            depth_force_n=np.array(
                [stress_part[0] * width, 0.0, stress_part[1] * width]
            ),
            inertial_force_n=np.array(
                [flux_part[0] * width, 0.0, flux_part[1] * width]
            ),
            n_active_elements=int(contacts[-1]),
            active_area_m2=float(contacts[-1]) * self.cell_size_m * width,
            max_depth_m=max(float(depths.max()) if depths.size else 0.0, 0.0),
        )


def _torque_window(run: MPMRun) -> int:
    """Steps in the trailing averaging window, at least one."""
    return max(1, min(run.n_steps, len(run.steps) // 10 or 1))
