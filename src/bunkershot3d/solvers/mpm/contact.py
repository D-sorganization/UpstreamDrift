"""Contact between the sand and **several** rigid bodies in one step (#8733 §2).

One body was enough while F1 had only a clubhead.  ADR-0033 puts the ball
in the plane-strain plane as a second body, so a step now has to project
the grid against a *sequence* of sections and keep one exact momentum
ledger for each of them.

Why the order has to be chosen rather than inherited
-----------------------------------------------------

The projection in :meth:`~.body.RigidSection.project_grid_velocity` is a
**velocity-level constraint written straight onto the node**.  When two
bodies overlap the same node, the second projection overwrites the first,
so the constraint that holds exactly at that node is the *last* one
applied and the earlier body's non-penetration may be left violated.
That makes the answer depend on the order, and an order that comes from
"whichever sequence the caller happened to build" is not a modelling
decision -- it is an accident that changes results when an argument list
is rearranged.

:func:`contact_order` therefore fixes the order from the bodies
themselves:

**Slowest first, fastest last, ties in the caller's order.**

Two reasons, in the order they matter:

1. **The fastest body is the one that can tunnel.**  Everything in
   :mod:`.body` about not letting sand through the club -- the CFL bound
   on body travel, the swept-node test, the pushout backstop -- scales
   with body speed.  Applying the fastest body last is what makes its
   non-penetration exact, and it is the body for which "exact" is worth
   paying for.  A club at 25 m/s crosses a 4 mm cell in 160 us; a ball
   sitting on the sand crosses nothing.
2. **The order is a function of state, not of arguments.**  Two callers
   who pass the same bodies in different sequences get the same answer,
   which is the property that makes the choice testable at all.  The
   stable tie-break keeps it deterministic when two bodies genuinely
   share a speed.

What is *not* order-dependent is the momentum ledger.  Each body records
``m_i (v_i^after - v_i^before)`` **at its own stage**, so the stages
telescope: whatever the order, the impulses sum to the total momentum the
contact projections moved.  That is the identity the conservation test
pins, and it is why a second body does not need a second force model.

An iterated projection -- sweeping the bodies until every constraint holds
at once -- would remove the dependence entirely.  It is not done here:
the bodies in a bunker shot overlap at a handful of nodes for a handful
of steps, the cost is a full distance-field evaluation per body per
sweep, and an unconverged iteration would trade a *stated* bias for an
unstated one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import SolverInputError
from .body import ContactImpulse, RigidSection

__all__ = [
    "NO_CONTACT",
    "BodyContact",
    "apply_body_contacts",
    "contact_order",
    "ledger_from_impulses",
    "push_out_bodies",
]

_DIMENSION = 2

NO_CONTACT = ContactImpulse(
    node_index=np.zeros(0, dtype=np.int64),
    impulse_n_s=np.zeros((0, _DIMENSION)),
    position_m=np.zeros((0, _DIMENSION)),
    stress_force_n=np.zeros(_DIMENSION),
    n_swept=0,
)
"""The ledger of a step with no intruder at all.

A bed marched without a body is a *closed* system, which is the only
configuration whose momentum budget is an identity rather than a balance
against a boundary -- so it is what the conservation cases run on."""


@dataclass(frozen=True, slots=True)
class BodyContact:
    """One body's share of one step, as an exact momentum ledger.

    The force on the body is ``-sum_i J_i / dt`` by Newton's third law, so
    there is no separate force model that could disagree with the
    accounting: :attr:`impulse_on_sand_n_s` and :attr:`force_n_per_m` are
    the same number read from the two ends.

    Attributes:
        force_n_per_m: ``(2,)`` in-plane force on the body, per unit
            out-of-plane width.
        stress_force_n_per_m: ``(2,)`` the stress-and-weight part of that
            force.
        torque_n: Torque on the body about ``+y``, about its own
            reference point, per unit width.
        impulse_on_sand_n_s: ``(2,)`` impulse this body applied **to the
            sand**, per unit width. The signed quantity the conservation
            identity is written in.
        n_contacts: Grid nodes this body projected.
        n_swept: Of those, nodes reached only by the swept test.
        n_pushed_out: Particles this body's backstop had to reposition.
        nodes: The node-resolved ledger the summary above was reduced
            from -- the same :class:`~.body.ContactImpulse` the projection
            returned, kept rather than discarded. A summed impulse cannot
            say *where on the body* the sand arrived, and that is exactly
            what #8712 asks of the ball, so the resolution is retained on
            the ledger rather than recomputed later from a pose that has
            since moved. It costs the projected nodes of one step, which
            is tens of rows.
    """

    force_n_per_m: NDArray[np.float64]
    stress_force_n_per_m: NDArray[np.float64]
    torque_n: float
    impulse_on_sand_n_s: NDArray[np.float64]
    n_contacts: int
    n_swept: int
    n_pushed_out: int
    nodes: ContactImpulse = NO_CONTACT


def contact_order(bodies: Sequence[RigidSection]) -> tuple[int, ...]:
    """Indices of ``bodies`` in the order their projections are applied.

    Ascending fastest-material-point speed, so the **fastest body is
    projected last** and its non-penetration is the constraint that holds
    exactly at any node two bodies share. Ties keep the caller's order,
    which makes the result deterministic without making it depend on the
    argument sequence in the cases that matter.

    Args:
        bodies: The bodies in one step, in the caller's own order.

    Returns:
        A permutation of ``range(len(bodies))``.

    Raises:
        SolverInputError: If any entry is not a
            :class:`~.body.RigidSection`, or if a body reports a
            non-finite speed -- a body whose speed cannot be compared
            cannot be ordered, and silently sorting it to one end would
            hide the fault.
    """
    speeds: list[float] = []
    for index, body in enumerate(bodies):
        if not isinstance(body, RigidSection):
            raise SolverInputError(
                f"bodies[{index}] must be a RigidSection, got {type(body).__name__}"
            )
        speed = body.max_speed_m_s
        if not math.isfinite(speed):
            raise SolverInputError(
                f"bodies[{index}] reports a non-finite speed {speed!r}, so the "
                "contact order is undefined"
            )
        speeds.append(speed)
    return tuple(sorted(range(len(speeds)), key=lambda index: speeds[index]))


def apply_body_contacts(
    bodies: Sequence[RigidSection],
    node_position_m: NDArray[np.float64],
    node_velocity_m_s: NDArray[np.float64],
    node_mass_kg: NDArray[np.float64],
    *,
    time_step_s: float,
    stress_force_n: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], tuple[ContactImpulse, ...]]:
    """Project the grid against every body, in :func:`contact_order`.

    Args:
        bodies: The bodies, in the caller's own order.
        node_position_m: ``(n_nodes, 2)`` node positions.
        node_velocity_m_s: ``(n_nodes, 2)`` nodal velocities after the
            force update and the wall conditions.
        node_mass_kg: ``(n_nodes,)`` nodal masses.
        time_step_s: The step.
        stress_force_n: ``(n_nodes, 2)`` internal-plus-weight force, for
            the stress/momentum-flux split.

    Returns:
        ``(projected_velocity, impulses)``. The impulses are aligned with
        ``bodies``, not with the order they were applied in, so a caller
        never has to undo the permutation.
    """
    if not bodies:
        return node_velocity_m_s, ()
    velocity = node_velocity_m_s
    impulses: list[ContactImpulse] = [NO_CONTACT] * len(bodies)
    for index in contact_order(bodies):
        velocity, impulses[index] = bodies[index].project_grid_velocity(
            node_position_m,
            velocity,
            node_mass_kg,
            time_step_s=time_step_s,
            stress_force_n=stress_force_n,
        )
    return velocity, tuple(impulses)


def push_out_bodies(
    bodies: Sequence[RigidSection],
    positions_m: NDArray[np.float64],
    velocity_m_s: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, ...]]:
    """Run every body's particle backstop, in the same order as contact.

    The backstop is geometric rather than a constraint, but it is applied
    in :func:`contact_order` for the same reason: a particle repaired onto
    the slow body's surface may still be inside the fast one, and the last
    repair is the one that stands.

    Args:
        bodies: The bodies, in the caller's own order.
        positions_m: ``(n, 2)`` particle positions after advection.
        velocity_m_s: ``(n, 2)`` particle velocities.

    Returns:
        ``(positions, velocities, counts)``, the counts aligned with
        ``bodies``.
    """
    if not bodies:
        return positions_m, velocity_m_s, ()
    positions = positions_m
    velocities = velocity_m_s
    counts = [0] * len(bodies)
    for index in contact_order(bodies):
        positions, velocities, counts[index] = bodies[index].push_out(
            positions, velocities
        )
    return positions, velocities, tuple(counts)


def ledger_from_impulses(
    bodies: Sequence[RigidSection],
    impulses: Sequence[ContactImpulse],
    pushed: Sequence[int],
    time_step_s: float,
) -> tuple[BodyContact, ...]:
    """Turn one step's raw impulses into one :class:`BodyContact` each.

    Args:
        bodies: The bodies, in the caller's order.
        impulses: Their ledgers, in the same order.
        pushed: Their pushout counts, in the same order.
        time_step_s: The step.

    Returns:
        One ledger per body, in the caller's order.

    Raises:
        SolverInputError: If the three sequences disagree in length --
            a misaligned ledger would attribute one body's load to
            another and still look plausible.
    """
    if not (len(bodies) == len(impulses) == len(pushed)):
        raise SolverInputError(
            f"bodies, impulses and pushout counts must align, got "
            f"{len(bodies)}, {len(impulses)} and {len(pushed)}"
        )
    return tuple(
        BodyContact(
            force_n_per_m=impulse.force_on_body_n(time_step_s),
            stress_force_n_per_m=-impulse.stress_force_n,
            torque_n=impulse.torque_on_body_n_m(time_step_s, body.reference_point_m),
            impulse_on_sand_n_s=(
                impulse.impulse_n_s.sum(axis=0)
                if impulse.n_contacts
                else np.zeros(_DIMENSION, dtype=np.float64)
            ),
            n_contacts=impulse.n_contacts,
            n_swept=impulse.n_swept,
            n_pushed_out=int(count),
            nodes=impulse,
        )
        for body, impulse, count in zip(bodies, impulses, pushed, strict=True)
    )
