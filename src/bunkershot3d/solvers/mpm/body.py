"""The clubhead as a rigid moving boundary in the plane-strain section.

The intruder is a **convex polygon** swept through the grid, and the sand
meets it through a velocity-level collision projection applied on the
grid nodes (Stomakhin, Schroeder, Chai, Teran & Selle 2013, *"A material
point method for snow simulation"*, ACM Trans. Graph. **32**(4):102,
section 8) with a Coulomb friction cone.

Why the projection is on the grid and not on the particles
----------------------------------------------------------

The grid already carries the momentum for the step, so a projection there
is an exact momentum exchange: the impulse removed from the sand is the
impulse delivered to the club, and it can be summed into a wrench without
a second force model.  Projecting particles instead would leave the grid
velocity -- which is what advects everything -- still pointing into the
club, and the reaction would have to be modelled separately.

Not tunnelling through the club
-------------------------------

Three mechanisms, because a single one is not enough at 25 m/s:

1. **The CFL step includes the body speed**, so the club advances less
   than a fraction of a cell per step and cannot skip over a node layer
   (:mod:`bunkershot3d.solvers.mpm.solver` forms and checks it).
2. **Swept-node collision.**  A node is collided if it is inside the body
   *now* or will be inside after this step's body motion.  A node the
   club is about to reach is stopped before the club arrives rather than
   after it has passed.
3. **Particle pushout after advection.**  Anything that still ends up
   inside the body is placed back on its surface with its inward normal
   velocity removed.  This is the backstop, and the solver reports how
   often it fires, because a pushout that fires constantly means the
   first two mechanisms are not doing their job.

The convex-polygon restriction
------------------------------

A wedge section -- sole, leading edge, bounce, face -- is very nearly
convex, and convexity buys an exact inside test and an unambiguous
nearest-point normal.  A non-convex section would need a real level set;
this is recorded as a limitation rather than approximated, and
:meth:`RigidSection.from_points` states plainly that it takes the convex
hull of what it is given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import SolverInputError

__all__ = [
    "ContactImpulse",
    "RigidSection",
    "convex_hull_2d",
    "coulomb_cone_projection",
    "plane_torque_about_y",
]

_DIMENSION = 2
_MIN_EDGE_LENGTH_M = 1e-12
_DEGENERATE_DISTANCE_M = 1e-15


def plane_torque_about_y(
    lever_m: NDArray[np.float64], force_n: NDArray[np.float64]
) -> float:
    """Out-of-plane torque of in-plane levers and forces.

    In the world frame the section spans ``x`` (along the path) and ``z``
    (up), so the only torque a plane-strain model can produce is about
    ``y``.  Written out because the sign is easy to get wrong:
    ``(r x F) . y_hat = r_z F_x - r_x F_z``, which is the *negative* of
    the usual scalar cross product of the ``(x, z)`` pair.

    Args:
        lever_m: ``(n, 2)`` levers from the reference point, ``(x, z)``.
        force_n: ``(n, 2)`` forces, ``(x, z)``.

    Returns:
        The summed torque about ``+y`` in N m (per unit width).
    """
    return float((lever_m[:, 1] * force_n[:, 0] - lever_m[:, 0] * force_n[:, 1]).sum())


def coulomb_cone_projection(
    relative_m_s: NDArray[np.float64],
    normal: NDArray[np.float64],
    *,
    friction: float,
) -> NDArray[np.float64]:
    """Project relative velocities onto the Coulomb friction cone.

    A **separating** node is returned untouched -- sand may leave the club
    freely, which is what makes a divot open behind the sole instead of
    the club dragging a vacuum. An **approaching** node has its normal
    component removed and its tangential component reduced inside the
    cone, sticking when the cone closes.

    The tangential update is ``v_t' = v_t + mu v_n v_t / |v_t|`` with
    ``v_n < 0``: the tangential speed is *reduced* by the friction bound
    rather than replaced by it, so a frictionless club leaves the
    tangential velocity entirely alone. That is the case this scaling has
    to get right, and it is why the bound appears as a subtraction.

    Args:
        relative_m_s: ``(k, 2)`` nodal velocity relative to the body.
        normal: ``(k, 2)`` outward unit contact normals.
        friction: Coulomb friction coefficient.

    Returns:
        ``(k, 2)`` projected relative velocities.
    """
    normal_speed = np.einsum("ij,ij->i", relative_m_s, normal)
    approaching = normal_speed < 0.0
    tangential = relative_m_s - normal_speed[:, None] * normal
    tangential_speed = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
    cone_limit = -float(friction) * normal_speed
    sticking = tangential_speed <= cone_limit
    safe_speed = np.where(tangential_speed > 0.0, tangential_speed, 1.0)
    sliding_scale = np.where(sticking, 0.0, 1.0 - cone_limit / safe_speed)
    projected = np.where(sticking[:, None], 0.0, sliding_scale[:, None] * tangential)
    return np.where(approaching[:, None], projected, relative_m_s)


def convex_hull_2d(points_m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Counter-clockwise convex hull by Andrew's monotone chain.

    Implemented here rather than taken from SciPy because ADR-0033 scopes
    F1 to NumPy alone, and a monotone chain is thirty lines.

    Args:
        points_m: ``(n, 2)`` points.

    Returns:
        ``(k, 2)`` hull vertices, counter-clockwise, without a repeated
        first point.

    Raises:
        SolverInputError: If fewer than three points survive, or if they
            are collinear -- a section with no area cannot be an
            intruder, and returning a degenerate polygon would let that
            failure surface later as a zero force.
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != _DIMENSION or points.shape[1] != _DIMENSION:
        raise SolverInputError(f"points_m must have shape (n, 2), got {points.shape!r}")
    if not np.all(np.isfinite(points)):
        raise SolverInputError("hull points contain non-finite values")
    unique = np.unique(points, axis=0)
    if unique.shape[0] < 3:
        raise SolverInputError(
            f"a section needs at least 3 distinct points, got {unique.shape[0]}"
        )
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    ordered = unique[order]

    def _half(sequence: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        chain: list[NDArray[np.float64]] = []
        for point in sequence:
            while len(chain) >= 2:
                first = chain[-1] - chain[-2]
                second = point - chain[-2]
                if first[0] * second[1] - first[1] * second[0] > 0.0:
                    break
                chain.pop()
            chain.append(point)
        return chain

    lower = _half(ordered)
    upper = _half(ordered[::-1])
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] < 3:
        raise SolverInputError(
            "the section's points are collinear, so the intruder has no area; "
            "a zero-area body would return a zero force rather than an error"
        )
    return hull


@dataclass(frozen=True, slots=True)
class ContactImpulse:
    """What one step's contact projection did, as an exact momentum ledger.

    The force on the club is ``-sum_i J_i / dt`` by Newton's third law, so
    this is the whole of F1's contact wrench: there is no separate force
    model to disagree with the momentum accounting.

    Attributes:
        node_index: Indices of the nodes that were projected.
        impulse_n_s: ``(k, 2)`` impulse **the body applied to the sand**,
            in N s per unit width.
        position_m: ``(k, 2)`` positions the impulses acted at.
        stress_force_n: ``(2,)`` the stress-and-weight part of the
            reaction, ``sum_i (f_i^int + m_i g)`` over projected nodes.
        n_swept: Nodes collided only because the body will reach them
            within this step, rather than because it already had.
    """

    node_index: NDArray[np.int64]
    impulse_n_s: NDArray[np.float64]
    position_m: NDArray[np.float64]
    stress_force_n: NDArray[np.float64]
    n_swept: int

    @property
    def n_contacts(self) -> int:
        """Number of projected nodes."""
        return int(self.node_index.size)

    def force_on_body_n(self, time_step_s: float) -> NDArray[np.float64]:
        """Total in-plane force on the body, N per unit width."""
        if self.n_contacts == 0:
            return np.zeros(_DIMENSION, dtype=np.float64)
        return -self.impulse_n_s.sum(axis=0) / time_step_s

    def torque_on_body_n_m(
        self, time_step_s: float, reference_point_m: NDArray[np.float64]
    ) -> float:
        """Torque on the body about ``+y``, N m per unit width."""
        if self.n_contacts == 0:
            return 0.0
        forces = -self.impulse_n_s / time_step_s
        return plane_torque_about_y(self.position_m - reference_point_m, forces)


@dataclass(frozen=True)
class RigidSection:
    """A convex polygon intruder with a rigid-body velocity.

    Attributes:
        vertices_m: ``(k, 2)`` counter-clockwise hull vertices in world
            ``(x, z)``.
        velocity_m_s: ``(2,)`` linear velocity of ``reference_point_m``.
        angular_velocity_rad_s: Rotation rate about ``+y``.
        reference_point_m: ``(2,)`` point the velocity refers to.
        friction: Coulomb friction coefficient between club and sand.
    """

    vertices_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    angular_velocity_rad_s: float
    reference_point_m: NDArray[np.float64]
    friction: float

    def __init__(
        self,
        vertices_m: ArrayLike,
        *,
        velocity_m_s: ArrayLike = (0.0, 0.0),
        angular_velocity_rad_s: float = 0.0,
        reference_point_m: ArrayLike | None = None,
        friction: float = 0.3,
    ) -> None:
        vertices = np.array(vertices_m, dtype=np.float64, copy=True)
        if vertices.ndim != _DIMENSION or vertices.shape[1] != _DIMENSION:
            raise SolverInputError(
                f"vertices_m must have shape (k, 2), got {vertices.shape!r}"
            )
        if vertices.shape[0] < 3:
            raise SolverInputError(
                f"a section needs at least 3 vertices, got {vertices.shape[0]}"
            )
        if not np.all(np.isfinite(vertices)):
            raise SolverInputError("section vertices contain non-finite values")
        velocity = np.array(velocity_m_s, dtype=np.float64, copy=True).reshape(-1)
        if velocity.shape != (_DIMENSION,) or not np.all(np.isfinite(velocity)):
            raise SolverInputError(
                f"velocity_m_s must be a finite 2-vector, got {velocity_m_s!r}"
            )
        spin = float(angular_velocity_rad_s)
        if not math.isfinite(spin):
            raise SolverInputError(
                f"angular_velocity_rad_s must be finite, got {angular_velocity_rad_s!r}"
            )
        reference = (
            vertices.mean(axis=0)
            if reference_point_m is None
            else np.array(reference_point_m, dtype=np.float64, copy=True).reshape(-1)
        )
        if reference.shape != (_DIMENSION,) or not np.all(np.isfinite(reference)):
            raise SolverInputError(
                f"reference_point_m must be a finite 2-vector, got "
                f"{reference_point_m!r}"
            )
        mu = float(friction)
        if not math.isfinite(mu) or mu < 0.0:
            raise SolverInputError(f"friction must be non-negative, got {friction!r}")
        if _signed_area(vertices) < 0.0:
            vertices = vertices[::-1].copy()
        for array in (vertices, velocity, reference):
            array.flags.writeable = False
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "angular_velocity_rad_s", spin)
        object.__setattr__(self, "reference_point_m", reference)
        object.__setattr__(self, "friction", mu)

    # ------------------------------------------------------------ geometry

    @classmethod
    def from_points(
        cls,
        points_m: ArrayLike,
        *,
        velocity_m_s: ArrayLike = (0.0, 0.0),
        angular_velocity_rad_s: float = 0.0,
        reference_point_m: ArrayLike | None = None,
        friction: float = 0.3,
    ) -> RigidSection:
        """Build from an arbitrary point cloud by taking its convex hull.

        The hull is taken, not assumed: a wedge section is very nearly
        convex, and saying so here is honest about the one shape feature
        this tier cannot represent -- a genuinely re-entrant grind is
        filled in.

        Args:
            points_m: ``(n, 2)`` points in the section plane.
            velocity_m_s: Linear velocity of the reference point.
            angular_velocity_rad_s: Rotation rate about ``+y``.
            reference_point_m: Point the velocity refers to.
            friction: Club-on-sand Coulomb friction.

        Returns:
            The section.
        """
        hull = convex_hull_2d(np.asarray(points_m, dtype=np.float64))
        return cls(
            hull,
            velocity_m_s=velocity_m_s,
            angular_velocity_rad_s=angular_velocity_rad_s,
            reference_point_m=reference_point_m,
            friction=friction,
        )

    @property
    def area_m2(self) -> float:
        """Section area, positive."""
        return abs(_signed_area(self.vertices_m))

    @property
    def speed_m_s(self) -> float:
        """Magnitude of the reference-point velocity."""
        return float(np.hypot(self.velocity_m_s[0], self.velocity_m_s[1]))

    @property
    def max_speed_m_s(self) -> float:
        """Fastest material point on the body, vertices included."""
        return float(np.linalg.norm(self.velocity_at(self.vertices_m), axis=1).max())

    def bounds_m(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(lower, upper)`` axis-aligned bounds of the section."""
        return self.vertices_m.min(axis=0), self.vertices_m.max(axis=0)

    def translated(self, offset_m: ArrayLike) -> RigidSection:
        """Return the same section moved rigidly by ``offset_m``."""
        offset = np.asarray(offset_m, dtype=np.float64).reshape(-1)
        if offset.shape != (_DIMENSION,) or not np.all(np.isfinite(offset)):
            raise SolverInputError(
                f"offset_m must be a finite 2-vector, got {offset_m!r}"
            )
        return RigidSection(
            self.vertices_m + offset,
            velocity_m_s=self.velocity_m_s,
            angular_velocity_rad_s=self.angular_velocity_rad_s,
            reference_point_m=self.reference_point_m + offset,
            friction=self.friction,
        )

    def with_velocity(
        self,
        velocity_m_s: ArrayLike,
        *,
        angular_velocity_rad_s: float | None = None,
    ) -> RigidSection:
        """Return the same section, in the same pose, moving differently.

        What a *marched* body needs and a prescribed one does not: the
        head's velocity is an unknown of the shot, so a step ends by
        rebuilding the section around a new one rather than by mutating
        a frozen value.

        Args:
            velocity_m_s: ``(2,)`` new linear velocity of the reference
                point.
            angular_velocity_rad_s: New rotation rate about ``+y``, or
                ``None`` to keep the current one.

        Returns:
            The section.
        """
        return RigidSection(
            self.vertices_m,
            velocity_m_s=velocity_m_s,
            angular_velocity_rad_s=(
                self.angular_velocity_rad_s
                if angular_velocity_rad_s is None
                else angular_velocity_rad_s
            ),
            reference_point_m=self.reference_point_m,
            friction=self.friction,
        )

    def advanced(self, time_step_s: float) -> RigidSection:
        """Return the section after ``time_step_s`` of its own motion.

        Rotation is applied about the reference point, so a spinning head
        is advanced exactly rather than by its translation alone.

        Args:
            time_step_s: The step.

        Returns:
            The advanced section.
        """
        step = float(time_step_s)
        if not math.isfinite(step):
            raise SolverInputError(f"time_step_s must be finite, got {time_step_s!r}")
        moved = self.vertices_m + self.velocity_m_s * step
        reference = self.reference_point_m + self.velocity_m_s * step
        if self.angular_velocity_rad_s != 0.0:
            # Rotation about +y takes (x, z) -> (x cos + z sin, -x sin + z cos).
            angle = self.angular_velocity_rad_s * step
            cosine = math.cos(angle)
            sine = math.sin(angle)
            lever = moved - reference
            moved = reference + np.stack(
                [
                    lever[:, 0] * cosine + lever[:, 1] * sine,
                    -lever[:, 0] * sine + lever[:, 1] * cosine,
                ],
                axis=1,
            )
        return RigidSection(
            moved,
            velocity_m_s=self.velocity_m_s,
            angular_velocity_rad_s=self.angular_velocity_rad_s,
            reference_point_m=reference,
            friction=self.friction,
        )

    def velocity_at(self, points_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rigid-body velocity at world points.

        ``v + omega y_hat x r`` which, in the ``(x, z)`` plane, is
        ``v + omega (r_z, -r_x)``.

        Args:
            points_m: ``(n, 2)`` world points.

        Returns:
            ``(n, 2)`` velocities.
        """
        points = np.asarray(points_m, dtype=np.float64)
        if self.angular_velocity_rad_s == 0.0:
            return np.broadcast_to(self.velocity_m_s, points.shape)
        lever = points - self.reference_point_m
        spin = self.angular_velocity_rad_s * np.stack(
            [lever[:, 1], -lever[:, 0]], axis=1
        )
        return self.velocity_m_s + spin

    def signed_distance(
        self, points_m: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Signed distance to the section and the outward unit normal.

        Negative inside.  The normal is the true gradient direction of the
        distance field: for an outside point it is the direction away from
        the nearest boundary point, which is correct in the corner regions
        where a max-of-half-planes surrogate would give the wrong
        direction; for an inside point it is the outward normal of the
        nearest edge.

        Args:
            points_m: ``(n, 2)`` world points.

        Returns:
            ``(distance, normal)`` of shapes ``(n,)`` and ``(n, 2)``.
        """
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != _DIMENSION or points.shape[1] != _DIMENSION:
            raise SolverInputError(
                f"points_m must have shape (n, 2), got {points.shape!r}"
            )
        start = self.vertices_m
        end = np.roll(self.vertices_m, -1, axis=0)
        edge = end - start
        length_squared = np.einsum("ij,ij->i", edge, edge)
        length_squared = np.where(
            length_squared > _MIN_EDGE_LENGTH_M**2, length_squared, 1.0
        )

        relative = points[:, None, :] - start[None, :, :]
        parameter = np.clip(
            np.einsum("nkj,kj->nk", relative, edge) / length_squared, 0.0, 1.0
        )
        closest = start[None, :, :] + parameter[:, :, None] * edge[None, :, :]
        separation = points[:, None, :] - closest
        distance = np.sqrt(np.einsum("nkj,nkj->nk", separation, separation))

        nearest = np.argmin(distance, axis=1)
        rows = np.arange(points.shape[0])
        magnitude = distance[rows, nearest]

        # Counter-clockwise winding puts the interior to the left of every
        # edge, so a point inside is left of all of them.
        left_of = (
            edge[None, :, 0] * relative[:, :, 1] - edge[None, :, 1] * relative[:, :, 0]
        )
        inside = np.all(left_of >= 0.0, axis=1)

        edge_normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
        edge_normal = edge_normal / np.sqrt(length_squared)[:, None]
        outward = edge_normal[nearest]
        usable = magnitude > _DEGENERATE_DISTANCE_M
        away = np.where(
            usable[:, None],
            separation[rows, nearest] / np.where(usable, magnitude, 1.0)[:, None],
            outward,
        )
        normal = np.where(inside[:, None], outward, away)
        return np.where(inside, -magnitude, magnitude), normal

    def contains(self, points_m: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Per-point inside test."""
        distance, _ = self.signed_distance(points_m)
        return distance < 0.0

    # ------------------------------------------------------------- contact

    def project_grid_velocity(
        self,
        node_position_m: NDArray[np.float64],
        node_velocity_m_s: NDArray[np.float64],
        node_mass_kg: NDArray[np.float64],
        *,
        time_step_s: float,
        stress_force_n: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], ContactImpulse]:
        """Apply the collision projection and return the momentum ledger.

        For every node the body occupies now or will occupy after this
        step, the sand's velocity relative to the body is decomposed on
        the outward normal.  A separating node is left alone -- sand may
        leave the club freely, which is what makes a divot open behind the
        sole instead of the club dragging a vacuum.  An approaching node
        has its normal component removed and its tangential component
        reduced inside the Coulomb cone, sticking when the cone closes.

        Args:
            node_position_m: ``(n_nodes, 2)`` node positions.
            node_velocity_m_s: ``(n_nodes, 2)`` nodal velocities after the
                force update.
            node_mass_kg: ``(n_nodes,)`` nodal masses.
            time_step_s: The step, used for the swept test and the ledger.
            stress_force_n: ``(n_nodes, 2)`` internal-plus-weight force at
                each node, recorded so the reaction can be split into its
                stress-borne and momentum-flux parts. Optional.

        Returns:
            ``(projected_velocity, impulse)``.

        Raises:
            SolverInputError: If the step is not positive and finite.
        """
        step = float(time_step_s)
        if not math.isfinite(step) or step <= 0.0:
            raise SolverInputError(f"time_step_s must be positive, got {step!r}")

        index, contact_normal, n_swept = self._active_nodes(
            node_position_m, node_mass_kg, step
        )
        if index.size == 0:
            empty: NDArray[np.float64] = np.zeros((0, _DIMENSION), dtype=np.float64)
            return node_velocity_m_s, ContactImpulse(
                node_index=np.zeros(0, dtype=np.int64),
                impulse_n_s=empty,
                position_m=empty,
                stress_force_n=np.zeros(_DIMENSION, dtype=np.float64),
                n_swept=0,
            )

        positions = node_position_m[index]
        body_velocity = self.velocity_at(positions)
        relative = node_velocity_m_s[index] - body_velocity
        new_relative = coulomb_cone_projection(
            relative, contact_normal, friction=self.friction
        )

        updated = node_velocity_m_s.copy()
        updated[index] = new_relative + body_velocity
        impulse = node_mass_kg[index, None] * (
            updated[index] - node_velocity_m_s[index]
        )

        stress_total = (
            np.zeros(_DIMENSION, dtype=np.float64)
            if stress_force_n is None
            else stress_force_n[index].sum(axis=0)
        )
        return updated, ContactImpulse(
            node_index=index,
            impulse_n_s=impulse,
            position_m=positions,
            stress_force_n=stress_total,
            n_swept=n_swept,
        )

    def _active_nodes(
        self,
        node_position_m: NDArray[np.float64],
        node_mass_kg: NDArray[np.float64],
        time_step_s: float,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], int]:
        """Which nodes this body collides, and on which normal.

        A node is collided if the body is inside it *now* or will be after
        this step's motion -- the second of the three anti-tunnelling
        mechanisms. A node the body has not reached yet is projected on
        the normal it will be hit with, not the one it happens to sit near
        now, which is the difference between stopping sand in front of the
        sole and stopping it beside the sole.

        Args:
            node_position_m: ``(n_nodes, 2)`` node positions.
            node_mass_kg: ``(n_nodes,)`` nodal masses; an empty node has
                no momentum to exchange and is skipped.
            time_step_s: The step, for the swept test.

        Returns:
            ``(index, contact_normal, n_swept)``.
        """
        distance, normal = self.signed_distance(node_position_m)
        occupied = distance < 0.0
        swept_distance, swept_normal = self.advanced(time_step_s).signed_distance(
            node_position_m
        )
        swept_only = (~occupied) & (swept_distance < 0.0)
        live = (occupied | swept_only) & (node_mass_kg > 0.0)
        index = np.flatnonzero(live)
        if index.size == 0:
            return index, np.zeros((0, _DIMENSION), dtype=np.float64), 0
        contact_normal = np.where(
            occupied[index, None], normal[index], swept_normal[index]
        )
        return index, contact_normal, int(swept_only[index].sum())

    def push_out(
        self,
        positions_m: NDArray[np.float64],
        velocity_m_s: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Place any particle inside the body back onto its surface.

        The backstop of the three anti-tunnelling mechanisms.  The
        particle is moved along the outward normal to the surface and its
        inward normal velocity is removed; the tangential component is
        left alone, so a particle sliding along the sole is not braked by
        the geometric repair.

        Args:
            positions_m: ``(n, 2)`` particle positions after advection.
            velocity_m_s: ``(n, 2)`` particle velocities.

        Returns:
            ``(positions, velocities, n_pushed)``.
        """
        distance, normal = self.signed_distance(positions_m)
        inside = distance < 0.0
        count = int(inside.sum())
        if count == 0:
            return positions_m, velocity_m_s, 0
        positions = positions_m.copy()
        velocities = velocity_m_s.copy()
        index = np.flatnonzero(inside)
        positions[index] -= distance[index, None] * normal[index]
        body_velocity = self.velocity_at(positions[index])
        relative = velocities[index] - body_velocity
        normal_speed = np.einsum("ij,ij->i", relative, normal[index])
        entering = normal_speed < 0.0
        relative = np.where(
            entering[:, None],
            relative - normal_speed[:, None] * normal[index],
            relative,
        )
        velocities[index] = relative + body_velocity
        return positions, velocities, count


def _signed_area(vertices_m: NDArray[np.float64]) -> float:
    """Shoelace area; positive for counter-clockwise winding."""
    following = np.roll(vertices_m, -1, axis=0)
    return 0.5 * float(
        (vertices_m[:, 0] * following[:, 1] - following[:, 0] * vertices_m[:, 1]).sum()
    )
