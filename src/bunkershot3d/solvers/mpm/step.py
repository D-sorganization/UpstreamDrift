"""One MPM step, extracted from the solver so a march can be driven step by step.

The scheme itself, in the order it happens
------------------------------------------

1. **P2G.**  Mass and APIC momentum onto the grid.
2. **Stress.**  Cauchy stress from each particle's elastic deformation
   gradient through the Hencky law, scattered as an internal force.
3. **Grid update.**  Symplectic Euler: ``v* = v + dt (f/m + g)``.
4. **Boundaries.**  Domain walls, then every body's collision projection
   in :func:`~.contact.contact_order`, which returns one exact momentum
   ledger per body.
5. **G2P.**  Velocity, the APIC affine matrix, and the velocity gradient.
6. **Plasticity.**  ``F <- (I + dt grad v) F``, then the capped
   Drucker-Prager return map.
7. **Advection**, then the particle-level pushout backstop.

Why it lives here rather than on the solver
-------------------------------------------

:class:`~.solver.PlaneStrainMPMSolver` owns three different things: the
scheme, the query-to-bed set-up, and the validity envelope.  Only the
first of those is what a whole-shot march (#8733 §3) needs, and it needs
it **one step at a time**, because the head's next pose depends on the
wrench this step produced.  Reaching into the solver's privates to get
that would make the march a friend of the solver; passing a
:class:`StepContext` makes it a caller.

The context carries the material and the walls rather than the solver, so
this module has no import back into :mod:`.solver` and the dependency
runs one way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..envelope import GRAVITY_M_S2
from ..exceptions import SolverInputError
from .body import RigidSection
from .constitutive import (
    SandContinuum,
    hencky_kirchhoff_principal,
    principal_stretches,
    reconstruct,
)
from .contact import (
    BodyContact,
    apply_body_contacts,
    ledger_from_impulses,
    push_out_bodies,
)
from .grid import (
    PlaneStrainGrid,
    affine_from_grid_velocity,
    gather_velocity,
    scatter_mass,
    scatter_momentum,
    scatter_stress_force,
    velocity_gradient,
)
from .state import DomainWalls, ParticleState, apply_wall_conditions

__all__ = [
    "StepContext",
    "StepDiagnostics",
    "advance_step",
    "cauchy_stress",
    "current_volume",
]

_DIMENSION = 2
_MASS_FLOOR_KG = 1e-15


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """What one step did, kept so the run is inspectable rather than opaque.

    The scalar contact fields describe the **primary** body -- the first
    entry of the step's body sequence, which is the clubhead in every
    shipped configuration -- so that a caller written before the ball
    existed reads the same number it always read. Every body's ledger,
    the primary one included, is in :attr:`body_contacts`.

    Attributes:
        time_s: Simulation time at the end of the step.
        contact_force_n_per_m: ``(2,)`` total in-plane force on the
            primary body, per unit out-of-plane width.
        stress_force_n_per_m: ``(2,)`` the stress-and-weight part of that
            force. See :meth:`~.solver.MPMRun.force_split` for what the
            split means and, just as importantly, what it does not.
        contact_torque_n: Torque on the primary body about ``+y``, per
            unit width.
        n_contacts: Grid nodes the primary body projected.
        n_swept: Of those, nodes reached only by the swept test.
        n_pushed_out: Particles the backstop had to reposition, over
            **all** bodies -- it is a health check on the scheme rather
            than a per-body load.
        n_yielded: Particles the return map moved.
        n_capped: Particles that hit the compressive cap.
        kinetic_energy_j_per_m: Translational kinetic energy.
        elastic_energy_j_per_m: Stored Hencky strain energy.
        gravitational_energy_j_per_m: Potential energy above the floor.
        linear_momentum_kg_m_s: ``(2,)`` total particle momentum.
        total_mass_kg_per_m: Total particle mass. Invariant.
        body_contacts: One ledger per body, in the order the caller
            supplied the bodies. Empty for a bed marched with no intruder.
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
    body_contacts: tuple[BodyContact, ...] = ()

    @property
    def n_bodies(self) -> int:
        """How many bodies this step projected against."""
        return len(self.body_contacts)

    def total_impulse_on_sand_n_s(self) -> NDArray[np.float64]:
        """``(2,)`` impulse every body together applied to the sand.

        The order the projections were applied in cannot change this sum:
        each body's ledger is the momentum change *at its own stage*, and
        the stages telescope.
        """
        if not self.body_contacts:
            return np.zeros(_DIMENSION, dtype=np.float64)
        return np.sum(
            [contact.impulse_on_sand_n_s for contact in self.body_contacts], axis=0
        )


@dataclass(frozen=True)
class StepContext:
    """Everything one step needs that does not change between steps.

    Attributes:
        grid: The background grid.
        material: The continuum.
        node_positions_m: ``(n_nodes, 2)`` node positions, formed once
            because the grid does not move.
        time_step_s: The step.
        datum_m: Height the gravitational energy is measured above.
        walls: Domain wall conditions.
        gravity_m_s2: Gravitational acceleration.
        damping_per_step: Fraction of the nodal velocity removed each
            step. Zero for a shot; the verification suite uses it to
            relax a column to static equilibrium, which is the only
            honest way to compare against a *static* analytic answer.
    """

    grid: PlaneStrainGrid
    material: SandContinuum
    node_positions_m: NDArray[np.float64]
    time_step_s: float
    datum_m: float
    walls: DomainWalls = field(default_factory=DomainWalls)
    gravity_m_s2: float = GRAVITY_M_S2
    damping_per_step: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_step_s) or self.time_step_s <= 0.0:
            raise SolverInputError(
                f"time_step_s must be positive, got {self.time_step_s!r}"
            )
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise SolverInputError(
                f"gravity_m_s2 must be positive, got {self.gravity_m_s2!r}"
            )
        if (
            not math.isfinite(self.damping_per_step)
            or not 0.0 <= self.damping_per_step < 1.0
        ):
            raise SolverInputError(
                f"damping_per_step must lie in [0, 1), got {self.damping_per_step!r}"
            )


def advance_step(
    particles: ParticleState,
    bodies: Sequence[RigidSection],
    context: StepContext,
    *,
    elapsed_s: float,
) -> StepDiagnostics:
    """Integrate one full MPM step, mutating ``particles`` in place.

    Args:
        particles: The bed. Advanced in place.
        bodies: Rigid intruders. Empty marches a closed system; the first
            entry is the primary body whose load the scalar diagnostics
            report.
        context: The grid, the material and the step.
        elapsed_s: Simulation time at the **end** of this step, recorded
            on the diagnostics.

    Returns:
        The step's diagnostics, one ledger per body included.
    """
    grid = context.grid
    time_step_s = context.time_step_s
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

    stress, elastic_energy = cauchy_stress(context.material, particles)
    internal_force = scatter_stress_force(
        grid, stencil, current_volume(particles), stress
    )
    weight = np.zeros_like(internal_force)
    weight[:, 1] = -nodal_mass * context.gravity_m_s2
    applied_force = internal_force + weight

    updated = node_velocity.copy()
    updated[live] += time_step_s * applied_force[live] / nodal_mass[live, None]
    if context.damping_per_step > 0.0:
        updated *= 1.0 - context.damping_per_step
    apply_wall_conditions(grid, updated, context.walls)
    updated, impulses = apply_body_contacts(
        bodies,
        context.node_positions_m,
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
    projected, yielded, capped = context.material.project(np.log(stretches))
    particles.deformation_gradient = reconstruct(left, projected, right)

    particles.position_m = particles.position_m + time_step_s * particles.velocity_m_s
    particles.position_m, particles.velocity_m_s, pushed = push_out_bodies(
        bodies, particles.position_m, particles.velocity_m_s
    )

    ledgers = ledger_from_impulses(bodies, impulses, pushed, time_step_s)
    primary = ledgers[0] if ledgers else None
    zero: NDArray[np.float64] = np.zeros(_DIMENSION, dtype=np.float64)
    return StepDiagnostics(
        time_s=float(elapsed_s),
        contact_force_n_per_m=zero if primary is None else primary.force_n_per_m,
        stress_force_n_per_m=(
            zero.copy() if primary is None else primary.stress_force_n_per_m
        ),
        contact_torque_n=0.0 if primary is None else primary.torque_n,
        n_contacts=0 if primary is None else primary.n_contacts,
        n_swept=0 if primary is None else primary.n_swept,
        n_pushed_out=int(sum(pushed)),
        n_yielded=int(yielded.sum()),
        n_capped=int(capped.sum()),
        kinetic_energy_j_per_m=particles.kinetic_energy_j(),
        elastic_energy_j_per_m=float(elastic_energy),
        gravitational_energy_j_per_m=particles.gravitational_energy_j(
            context.gravity_m_s2, context.datum_m
        ),
        linear_momentum_kg_m_s=particles.linear_momentum_kg_m_s(),
        total_mass_kg_per_m=particles.total_mass_kg,
        body_contacts=ledgers,
    )


def cauchy_stress(
    material: SandContinuum, particles: ParticleState
) -> tuple[NDArray[np.float64], float]:
    """Cauchy stress and stored energy from the elastic deformation.

    For an isotropic model the Kirchhoff stress is coaxial with the
    *left* stretch, so ``tau = U diag(tau_i) U^T`` and ``sigma = tau / J``.
    The stored energy is the Hencky strain energy
    ``mu ||eps||^2 + (lambda / 2) tr(eps)^2``, which the conservation
    suite needs to close the energy budget.

    Args:
        material: The continuum.
        particles: The bed.

    Returns:
        ``(stress, stored_energy_j_per_m)``.
    """
    left, stretches, _ = principal_stretches(particles.deformation_gradient)
    strain = np.log(stretches)
    kirchhoff = hencky_kirchhoff_principal(
        strain,
        shear_modulus_pa=material.shear_modulus_pa,
        lame_lambda_pa=material.lame_lambda_pa,
    )
    jacobian = stretches.prod(axis=1)
    principal = kirchhoff / jacobian[:, None]
    stress = np.einsum("nik,nk,njk->nij", left, principal, left)

    trace = strain.sum(axis=1)
    density = (
        material.shear_modulus_pa * np.einsum("ij,ij->i", strain, strain)
        + 0.5 * material.lame_lambda_pa * trace**2
    )
    energy = float((particles.initial_volume_m2 * density).sum())
    return stress, energy


def current_volume(particles: ParticleState) -> NDArray[np.float64]:
    """``V = J V_0``, the deformed particle area."""
    jacobian = np.linalg.det(particles.deformation_gradient)
    return particles.initial_volume_m2 * jacobian
