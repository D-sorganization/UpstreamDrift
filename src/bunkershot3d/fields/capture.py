"""Reading the sand field out of an F1 march (issue #8710).

Where the numbers come from
---------------------------

Nothing here forms a new quantity.  Every array is produced by the
solver's own transfer operators, applied to the solver's own particle
state, so a stored field is the field the solve actually had rather than
a reconstruction of it:

* **Density** is the nodal mass from
  :func:`~bunkershot3d.solvers.mpm.grid.scatter_mass` divided by the cell
  area.  Plane-strain masses are kg/m and the cell area is m^2, so the
  quotient is kg/m^3 with no fudge factor.
* **Velocity** is the nodal momentum from
  :func:`~bunkershot3d.solvers.mpm.grid.scatter_momentum` -- the same
  APIC transfer the next step would perform -- divided by that mass.
* **Shear rate** is ``sqrt(2 D : D)`` with ``D = sym(grad v)``, formed at
  the particles with
  :func:`~bunkershot3d.solvers.mpm.grid.velocity_gradient` and then
  mass-weighted back onto the nodes.  Forming it at the particles rather
  than by differencing the nodal field matters: an empty node beside a
  full one differences to an enormous false shear right along the free
  surface, which is exactly where the interesting flow is.

Empty is not zero
-----------------

A node with no sand gets density 0 and shear rate ``nan``.  The density
is a real measurement -- there is no material there -- but a shear rate
of zero would assert that the sand at the free surface is not shearing,
which is a different and false claim.  ``nan`` is the honest value and
every downstream view masks on it, the same convention
:func:`~bunkershot3d.solvers.mpm.state.surface_profile_m` already uses
for an emptied divot bin.

Marching in strides, not re-running
-----------------------------------

Capture drives :meth:`~bunkershot3d.solvers.mpm.solver.PlaneStrainMPMSolver.march`
in blocks of ``stride`` steps from a single
:class:`~bunkershot3d.solvers.mpm.solver.MPMSetup`, sampling between
blocks.  The bed is advanced in place, so the trajectory is bit-for-bit
the trajectory ``run()`` would have produced; the only difference is that
somebody looked at it on the way past.

What the kinematics are
-----------------------

F1 supplies a **declared straight-line constant-velocity approach**, not
a marched swing (issue #8733 holds whole-shot marching).  An animation of
an approach and an animation of a shot look identical, so the assumption
is written into :class:`~bunkershot3d.fields.schema.FieldProvenance` and
travels with every frame.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DValueError
from ..solvers.envelope import RefusalPolicy, ValidityVerdict
from ..solvers.mpm.envelope import RefusedQuantity
from ..solvers.mpm.grid import (
    PlaneStrainGrid,
    scatter_mass,
    scatter_momentum,
    velocity_gradient,
)
from ..solvers.mpm.body import RigidSection
from ..solvers.mpm.solver import MPMRun, MPMSetup, PlaneStrainMPMSolver, StepDiagnostics
from ..solvers.mpm.state import ParticleState
from ..solvers.protocol import IntrusionState
from .schema import (
    FieldLayout,
    FieldProvenance,
    GridGeometry,
    OccupancyRule,
    RetentionPolicy,
    RetentionRecord,
    SandFieldSeries,
)

__all__ = [
    "F1_KINEMATICS_NOTE",
    "GridFieldSample",
    "capture_f1_field",
    "sample_grid_field",
]

F1_KINEMATICS_NOTE = (
    "declared straight-line constant-velocity approach to the queried pose "
    "(F1 has no marched swing; whole-shot marching is issue #8733)"
)
"""How the body's motion was supplied, in words, for the provenance record.

Stated rather than implied because a field animated from a declared
approach and a field animated from a marched shot are indistinguishable
on screen and are not the same claim."""

_MASS_FLOOR_KG = 1e-15
"""Nodal mass below which a node is treated as carrying no sand.

The same floor the solver's own grid update uses, so the field's notion
of "empty" and the solve's notion of "empty" cannot drift apart."""

_DIMENSION = 2


class GridFieldSample:
    """One instant of the nodal field, before it joins a series.

    Attributes:
        velocity_m_s: ``(n_nodes, 2)`` nodal velocities.
        density_kg_m3: ``(n_nodes,)`` nodal densities; zero where empty.
        shear_rate_1_s: ``(n_nodes,)`` shear rates, ``nan`` where empty,
            or ``None`` when the caller did not ask for them.
    """

    __slots__ = ("density_kg_m3", "shear_rate_1_s", "velocity_m_s")

    def __init__(
        self,
        velocity_m_s: NDArray[np.float64],
        density_kg_m3: NDArray[np.float64],
        shear_rate_1_s: NDArray[np.float64] | None,
    ) -> None:
        """Store the three arrays of one sampled instant."""
        self.velocity_m_s = velocity_m_s
        self.density_kg_m3 = density_kg_m3
        self.shear_rate_1_s = shear_rate_1_s


def sample_grid_field(
    grid: PlaneStrainGrid,
    particles: ParticleState,
    *,
    include_shear_rate: bool = True,
) -> GridFieldSample:
    """Transfer the particle state onto the grid as velocity/density/shear.

    Args:
        grid: The background grid.
        particles: The bed at the instant being sampled.
        include_shear_rate: Whether to form the shear rate, which costs a
            second gather and a per-particle 2x2 contraction.

    Returns:
        The sampled instant.
    """
    stencil = grid.interpolate(particles.position_m)
    nodal_mass = scatter_mass(grid, stencil, particles.mass_kg)
    nodal_momentum = scatter_momentum(
        grid, stencil, particles.mass_kg, particles.velocity_m_s, particles.affine
    )
    live = nodal_mass > _MASS_FLOOR_KG

    velocity = np.zeros_like(nodal_momentum)
    velocity[live] = nodal_momentum[live] / nodal_mass[live, None]
    density = nodal_mass / grid.cell_volume_m2

    shear: NDArray[np.float64] | None = None
    if include_shear_rate:
        gradient = velocity_gradient(stencil, velocity)
        rate_of_strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
        magnitude = np.sqrt(
            2.0 * np.einsum("nij,nij->n", rate_of_strain, rate_of_strain)
        )
        # scatter_mass is the partition-of-unity scatter of a per-particle
        # scalar; handing it m_p * gamma_p makes the nodal value a
        # mass-weighted average once divided by the nodal mass.
        weighted = scatter_mass(grid, stencil, particles.mass_kg * magnitude)
        shear = np.full(grid.n_nodes, np.nan)
        shear[live] = weighted[live] / nodal_mass[live]

    return GridFieldSample(velocity, density, shear)


def capture_f1_field(
    solver: PlaneStrainMPMSolver,
    state: IntrusionState,
    *,
    policy: RetentionPolicy | None = None,
    refusal_policy: RefusalPolicy | None = None,
) -> tuple[SandFieldSeries, MPMRun]:
    """March an F1 query and keep the sand field as it goes.

    Args:
        solver: The F1 solver.
        state: The intrusion query.
        policy: What to keep. Defaults to
            :class:`~bunkershot3d.fields.schema.RetentionPolicy`'s own
            defaults, which target 120 frames at float32.
        refusal_policy: Overrides the solver's policy for this capture.
            A field of a refused query is legitimate to *look* at while
            being illegitimate to quote, and the refusal travels in the
            provenance either way, so a caller may ask for one
            deliberately.

    Returns:
        ``(series, run)`` -- the stored field and the full march trace,
        so a caller gets the wrench history without paying for a second
        solve.

    Raises:
        BunkerShot3DValueError: If ``solver`` is not an F1 solver.
        OutOfEnvelopeError: If the verdict refuses under the policy in
            force.
    """
    if not isinstance(solver, PlaneStrainMPMSolver):
        raise BunkerShot3DValueError(
            "capture_f1_field needs a PlaneStrainMPMSolver; other tiers write "
            f"their own capture against the same schema, got {type(solver).__name__}"
        )
    keep = RetentionPolicy() if policy is None else policy
    verdict = solver.envelope(state)
    verdict.require_usable(
        solver.refusal_policy if refusal_policy is None else refusal_policy
    )

    setup = solver.prepare(state)
    indices, geometry, cropped_note = _crop(setup.grid, keep)
    times, samples, outlines, steps = _march_and_sample(solver, setup, keep, indices)
    run = MPMRun(
        steps=tuple(steps),
        time_step_s=setup.time_step_s,
        particles=setup.particles,
        grid=setup.grid,
        section=setup.section.advanced(setup.time_step_s * setup.n_steps),
        free_surface_height_m=setup.free_surface_height_m,
        bed_x_bounds_m=setup.bed_x_bounds_m,
    )

    stride = keep.stride_for(setup.n_steps)
    record = RetentionRecord(
        policy=keep,
        steps_marched=setup.n_steps,
        time_stride=stride,
        frames_kept=len(times),
        time_step_s=setup.time_step_s,
        samples_in_domain=setup.grid.n_nodes,
        samples_kept=int(indices.size),
        dropped=_dropped_lines(keep, setup, stride, indices, cropped_note),
    )
    store = np.dtype(keep.store_dtype)
    return (
        SandFieldSeries(
            time_s=np.asarray(times, dtype=np.float64),
            velocity_m_s=np.stack(
                [sample.velocity_m_s for sample in samples], axis=0
            ).astype(store, copy=False),
            density_kg_m3=np.stack(
                [sample.density_kg_m3 for sample in samples], axis=0
            ).astype(store, copy=False),
            shear_rate_1_s=(
                None
                if not keep.include_shear_rate
                else np.stack(
                    [_require_shear(sample.shear_rate_1_s) for sample in samples],
                    axis=0,
                ).astype(store, copy=False)
            ),
            positions_m=None,
            layout=FieldLayout.GRID,
            geometry=geometry,
            provenance=_provenance(solver, state, setup, verdict),
            retention=record,
            occupancy=OccupancyRule(
                reference_density_kg_m3=float(solver.material.density_kg_m3)
            ),
            body_outline_m=np.stack(outlines, axis=0).astype(store, copy=False),
        ),
        run,
    )


def _require_shear(shear: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Unwrap a shear array the policy said would be there."""
    if shear is None:  # pragma: no cover - guarded by the policy flag
        raise BunkerShot3DValueError(
            "the retention policy asked for the shear rate but none was formed"
        )
    return shear


def _march_and_sample(
    solver: PlaneStrainMPMSolver,
    setup: MPMSetup,
    policy: RetentionPolicy,
    indices: NDArray[np.int64],
) -> tuple[
    list[float],
    list[GridFieldSample],
    list[NDArray[np.float64]],
    list[StepDiagnostics],
]:
    """March in stride-sized blocks, sampling the field between them.

    The undisturbed bed is sampled first, at ``t = 0``, because it is the
    reference every later frame is read against: without it the animation
    opens on sand that is already moving and gives no sense of what the
    club changed.

    The intruder's own outline is kept alongside each frame, because a
    velocity field with no body in it cannot answer the question the
    field was computed for.
    """
    stride = policy.stride_for(setup.n_steps)
    times = [0.0]
    samples = [_take(setup, policy, indices)]
    steps: list[StepDiagnostics] = []
    section: RigidSection = setup.section
    outlines: list[NDArray[np.float64]] = [np.asarray(section.vertices_m)]
    remaining = setup.n_steps
    elapsed = 0.0
    while remaining > 0:
        block = min(stride, remaining)
        run = solver.march(
            setup.particles,
            section,
            setup.grid,
            n_steps=block,
            time_step_s=setup.time_step_s,
            free_surface_height_m=setup.free_surface_height_m,
            bed_x_bounds_m=setup.bed_x_bounds_m,
        )
        steps.extend(_shifted(diagnostic, elapsed) for diagnostic in run.steps)
        if run.section is None:  # pragma: no cover - setup always carries one
            raise BunkerShot3DValueError(
                "a capture march lost its club section, so the next block would "
                "advance an empty bed"
            )
        section = run.section
        elapsed += block * setup.time_step_s
        remaining -= block
        times.append(elapsed)
        samples.append(_take(setup, policy, indices))
        outlines.append(np.asarray(section.vertices_m))
    return times, samples, outlines, steps


def _shifted(diagnostic: StepDiagnostics, offset_s: float) -> StepDiagnostics:
    """Re-time one block's diagnostics onto the whole march's clock.

    ``march`` restarts its elapsed time at zero for every call, so a
    strided capture would otherwise report the same few timestamps over
    and over and the wrench history would be unreadable.
    """
    if offset_s == 0.0:
        return diagnostic
    return StepDiagnostics(
        time_s=diagnostic.time_s + offset_s,
        contact_force_n_per_m=diagnostic.contact_force_n_per_m,
        stress_force_n_per_m=diagnostic.stress_force_n_per_m,
        contact_torque_n=diagnostic.contact_torque_n,
        n_contacts=diagnostic.n_contacts,
        n_swept=diagnostic.n_swept,
        n_pushed_out=diagnostic.n_pushed_out,
        n_yielded=diagnostic.n_yielded,
        n_capped=diagnostic.n_capped,
        kinetic_energy_j_per_m=diagnostic.kinetic_energy_j_per_m,
        elastic_energy_j_per_m=diagnostic.elastic_energy_j_per_m,
        gravitational_energy_j_per_m=diagnostic.gravitational_energy_j_per_m,
        linear_momentum_kg_m_s=diagnostic.linear_momentum_kg_m_s,
        total_mass_kg_per_m=diagnostic.total_mass_kg_per_m,
    )


def _take(
    setup: MPMSetup, policy: RetentionPolicy, indices: NDArray[np.int64]
) -> GridFieldSample:
    """Sample the field and apply the spatial crop."""
    sample = sample_grid_field(
        setup.grid, setup.particles, include_shear_rate=policy.include_shear_rate
    )
    return GridFieldSample(
        velocity_m_s=sample.velocity_m_s[indices],
        density_kg_m3=sample.density_kg_m3[indices],
        shear_rate_1_s=(
            None if sample.shear_rate_1_s is None else sample.shear_rate_1_s[indices]
        ),
    )


def _crop(
    grid: PlaneStrainGrid, policy: RetentionPolicy
) -> tuple[NDArray[np.int64], GridGeometry, str]:
    """Flat node indices kept by the policy, and the geometry they form.

    Cropping is done on whole node lines so the kept samples are still a
    uniform lattice; a ragged crop would force every downstream view to
    fall back on scattered interpolation for no saving worth having.
    """
    count_x, count_z = grid.node_counts
    full = GridGeometry(
        origin_m=grid.origin_m,
        cell_size_m=grid.cell_size_m,
        shape=(count_x, count_z),
        axis_names=("x", "z"),
    )
    if policy.region_m is None:
        return np.arange(grid.n_nodes, dtype=np.int64), full, ""

    lower, upper = policy.region_m
    if len(lower) != _DIMENSION:
        raise BunkerShot3DValueError(
            f"region_m must have {_DIMENSION} components for a plane-strain "
            f"field, got {policy.region_m!r}"
        )
    keeps: list[NDArray[np.int64]] = []
    for axis in range(_DIMENSION):
        coordinates = full.axis_coordinates_m(axis)
        inside = np.flatnonzero(
            (coordinates >= lower[axis]) & (coordinates <= upper[axis])
        ).astype(np.int64)
        if inside.size == 0:
            raise BunkerShot3DValueError(
                f"the crop region {policy.region_m!r} keeps no node on axis "
                f"{full.axis_names[axis]}; a field cropped to nothing is not a "
                "smaller field, it is an empty one"
            )
        keeps.append(inside)
    kept_x, kept_z = keeps
    flat = (kept_x[:, None] * count_z + kept_z[None, :]).ravel().astype(np.int64)
    geometry = GridGeometry(
        origin_m=np.array(
            [
                full.axis_coordinates_m(0)[kept_x[0]],
                full.axis_coordinates_m(1)[kept_z[0]],
            ]
        ),
        cell_size_m=grid.cell_size_m,
        shape=(int(kept_x.size), int(kept_z.size)),
        axis_names=("x", "z"),
    )
    note = (
        f"cropped to x in [{lower[0]:.4g}, {upper[0]:.4g}] m, "
        f"z in [{lower[1]:.4g}, {upper[1]:.4g}] m"
    )
    return flat, geometry, note


def _dropped_lines(
    policy: RetentionPolicy,
    setup: MPMSetup,
    stride: int,
    indices: NDArray[np.int64],
    crop_note: str,
) -> tuple[str, ...]:
    """Say, in words, everything this capture threw away."""
    lines: list[str] = []
    if stride > 1:
        lines.append(
            f"temporal: kept 1 step in {stride} of {setup.n_steps}, so "
            f"{setup.n_steps - math.ceil(setup.n_steps / stride)} steps are not "
            f"stored ({stride * setup.time_step_s * 1e6:.3g} us between frames)"
        )
    dropped_samples = setup.grid.n_nodes - int(indices.size)
    if dropped_samples > 0:
        lines.append(
            f"spatial: {crop_note}, so {dropped_samples} of "
            f"{setup.grid.n_nodes} samples are not stored"
        )
    if policy.store_dtype != "float64":
        lines.append(
            f"precision: solved in float64, stored as {policy.store_dtype} "
            f"(relative precision {policy.relative_precision:.2g})"
        )
    if not policy.include_shear_rate:
        lines.append("shear rate: not formed, so it cannot be recovered from this file")
    return tuple(lines)


def _provenance(
    solver: PlaneStrainMPMSolver,
    state: IntrusionState,
    setup: MPMSetup,
    verdict: ValidityVerdict,
) -> FieldProvenance:
    """Everything needed to trace this field to its run and regenerate it."""
    material = solver.material
    settings: dict[str, float | int | str] = {
        "cell_size_m": float(solver.cell_size_m),
        "effective_width_m": float(solver.effective_width_m),
        "bed_depth_m": float(solver.bed_depth_m),
        "run_in_lengths": float(solver.run_in_lengths),
        "particles_per_cell_axis": int(solver.particles_per_cell_axis),
        "cfl_number": float(solver.cfl_number),
        "gravity_m_s2": float(solver.gravity_m_s2),
        "contact_friction": float(solver.contact_friction),
        "time_step_s": float(setup.time_step_s),
        "n_steps": int(setup.n_steps),
        "approach_distance_m": float(setup.approach_distance_m),
        "free_surface_height_m": float(setup.free_surface_height_m),
        "wall_lower_x": str(solver.walls.lower_x.value),
        "wall_upper_x": str(solver.walls.upper_x.value),
        "wall_lower_z": str(solver.walls.lower_z.value),
        "wall_upper_z": str(solver.walls.upper_z.value),
        "sand_density_kg_m3": float(material.density_kg_m3),
        "sand_shear_modulus_pa": float(material.shear_modulus_pa),
        "sand_lame_lambda_pa": float(material.lame_lambda_pa),
        "sand_friction_angle_deg": float(material.friction_angle_deg),
        "sand_cohesion_pa": float(material.cohesion_pa),
        "sand_grain_diameter_m": float(material.grain_diameter_m),
        "sand_alpha": float(material.alpha),
        "sand_cap_volumetric_strain": float(material.cap_volumetric_strain),
        "sand_tip_volumetric_strain": float(material.tip_volumetric_strain),
    }
    return FieldProvenance(
        fidelity_tier=solver.fidelity_tier,
        envelope_status=verdict.status,
        solver_name=f"{type(solver).__module__}.{type(solver).__name__}",
        kinematics=F1_KINEMATICS_NOTE,
        peak_speed_m_s=float(state.speed_m_s),
        caveats=tuple(caveat.value for caveat in verdict.caveats),
        reasons=tuple(verdict.reasons),
        refused=tuple(quantity.value for quantity in RefusedQuantity),
        settings=settings,
        seeds=(),
    )
