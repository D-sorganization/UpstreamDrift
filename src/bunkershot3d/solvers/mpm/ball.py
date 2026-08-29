"""The ball as a rigid circular section in the plane-strain plane (ADR-0033).

ADR-0033 decides that "the ball becomes a body inside F1, as a rigid
circular section in the plane-strain plane, coupled to the sand by the
same traction integration used for the sole".  That is a modelling
decision taken in the ADR, not one left for the visual layers to
discover, and it is cheap to honour: :class:`.body.RigidSection` already
carries a convex polygon, a rigid velocity field, a swept collision test
and an exact momentum ledger, so a circle is a polygonal approximation
plus a second body in the step loop.

Two facts travel with the ball, in the API rather than in a comment
--------------------------------------------------------------------

**It is an infinite cylinder, not a sphere.**  Plane strain has no third
dimension, so the body the sand meets is a cylinder of unlimited length
whose section happens to be the ball's great circle.  Everything that
crosses onto it is therefore *per unit width*, and every quantity whose
definition needs the third dimension -- a mass, a moment of inertia, a
surface area -- is wrong for a golf ball.  :meth:`BallSection.line_mass_kg_per_m`
gives the honest per-unit-width quantity and says so in its name;
:meth:`BallSection.sphere_mass_kg` raises rather than returning
``rho (4/3) pi R^3``, because that number would be quoted.

**The below-equator / face-side split is in-plane and qualitative.**
Issue #8712 wants to show where on the ball the sand arrives.  Above and
below the equator, and near-side against far-side, are directions the
plane-strain section *has*, so they can be reported -- but only as a
share of an impulse that is itself per unit width on a body of the wrong
three-dimensional shape, which is why :class:`BallContactSplit` reports
fractions, flags itself
:attr:`~BallContactSplit.is_qualitative`, and carries the cylinder note
into its own summary.  Any heel-toe or lateral distribution is a
direction the model does not have, and
:meth:`BallSection.heel_toe_split` raises
:attr:`~bunkershot3d.solvers.mpm.envelope.RefusedQuantity.OUT_OF_PLANE`
rather than approximating it.

Ball launch is not here
-----------------------

The ball body exists so that "sand reaching the ball" acquires a referent
it never had at F0.  It is not a launch model:
:meth:`BallSection.launch_velocity_m_s` raises
:attr:`~bunkershot3d.solvers.mpm.envelope.RefusedQuantity.BALL_LAUNCH`,
and ball speed, launch angle and spin stay on F0's momentum-transfer path
(#8657).

Why the polygon has equal area rather than equal radius
-------------------------------------------------------

An inscribed polygon displaces less sand than the circle it stands for;
a circumscribed one displaces more.  Both biases are systematic and both
survive refinement of the *grid*, so they would show up as a quiet offset
in the flux onto the ball rather than as noise.  Scaling the circumradius
by ``sqrt(2 pi / (n sin(2 pi / n)))`` makes the section area exactly
``pi R^2``, so the polygon crosses the circle -- inradius inside,
circumradius outside -- and the displaced area is unbiased at every facet
count.  :func:`n_facets_for_cell_size` then picks a facet count fine
enough that the remaining flat spots are smaller than one grid cell,
which is the only scale at which the sand can see them at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import OutOfEnvelopeError, SolverInputError
from .body import ContactImpulse, RigidSection
from .envelope import RefusedQuantity, require_quotable

__all__ = [
    "BALL_DIAMETER_M",
    "BALL_RADIUS_M",
    "DEFAULT_BALL_FACETS",
    "MIN_BALL_FACETS",
    "PLANE_STRAIN_BALL_NOTE",
    "BallContactSplit",
    "BallSection",
    "circular_section",
    "n_facets_for_cell_size",
]

BALL_DIAMETER_M = 0.042672
"""Minimum conforming ball diameter, 1.680 in (R&A/USGA Rule 4.2a).

The *minimum* rather than a measured mean, because it is the only
diameter that is a rule rather than a sample."""

BALL_RADIUS_M = 0.5 * BALL_DIAMETER_M
"""Half of :data:`BALL_DIAMETER_M`."""

DEFAULT_BALL_FACETS = 24
"""Facets of the polygonal circle when no grid is quoted.

At 24 facets the equal-area polygon's radius departs from the circle's by
at most 0.57 % outward and 0.29 % inward, which is well inside the
1-2 mm bulk resolution ADR-0033 specifies for the tier."""

MIN_BALL_FACETS = 8
"""Fewest facets a circular section may be built from.

Below eight the polygon stops reading as a circle in a rendered field,
which is what the ball is for."""

PLANE_STRAIN_BALL_NOTE = (
    "In plane strain the ball is an infinite cylinder, not a sphere: the "
    "section is the ball's great circle but the body extends without limit "
    "out of plane. Everything that reaches it is a flux per unit width, and "
    "any quantity that needs the third dimension -- a mass, a moment of "
    "inertia, a surface area -- is wrong for a golf ball. Ball launch stays "
    "on F0's momentum-transfer path (#8657)."
)
"""The geometry statement that travels with every ball quantity."""

_DIMENSION = 2
_MIN_DIRECTION_NORM = 1e-12


def circular_section(
    centre_m: ArrayLike,
    radius_m: float,
    *,
    n_facets: int = DEFAULT_BALL_FACETS,
    velocity_m_s: ArrayLike = (0.0, 0.0),
    angular_velocity_rad_s: float = 0.0,
    friction: float = 0.3,
) -> RigidSection:
    """Return an equal-area regular polygon standing in for a circle.

    Args:
        centre_m: ``(2,)`` centre in world ``(x, z)``. Becomes the
            section's reference point, so the rigid velocity field and
            the torque are both taken about the centre.
        radius_m: Radius of the circle the polygon stands for.
        n_facets: Number of facets, at least :data:`MIN_BALL_FACETS`.
        velocity_m_s: Linear velocity of the centre.
        angular_velocity_rad_s: Rotation rate about ``+y``.
        friction: Coulomb friction between the body and the sand.

    Returns:
        The section, whose area is ``pi radius_m ** 2`` to round-off.

    Raises:
        SolverInputError: If the radius is not positive and finite, or if
            too few facets are asked for.
    """
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise SolverInputError(f"radius_m must be positive, got {radius_m!r}")
    facets = int(n_facets)
    if facets < MIN_BALL_FACETS:
        raise SolverInputError(
            f"n_facets must be at least {MIN_BALL_FACETS}, got {n_facets!r}: below "
            "that the polygon no longer reads as a circle, which is what the ball "
            "body is for"
        )
    centre = np.asarray(centre_m, dtype=np.float64).reshape(-1)
    if centre.shape != (_DIMENSION,) or not np.all(np.isfinite(centre)):
        raise SolverInputError(f"centre_m must be a finite 2-vector, got {centre_m!r}")

    # Equal area: (1/2) n r^2 sin(2 pi / n) = pi R^2.
    wedge = 2.0 * math.pi / facets
    circumradius = radius * math.sqrt(wedge / math.sin(wedge))
    angles = np.arange(facets, dtype=np.float64) * wedge
    vertices = centre + circumradius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )
    return RigidSection(
        vertices,
        velocity_m_s=velocity_m_s,
        angular_velocity_rad_s=angular_velocity_rad_s,
        reference_point_m=centre,
        friction=friction,
    )


def n_facets_for_cell_size(
    *, radius_m: float, cell_size_m: float, minimum: int = MIN_BALL_FACETS
) -> int:
    """Fewest facets whose chord fits inside one grid cell.

    A facet the grid cannot resolve is a flat spot the sand cannot feel,
    so there is no reason to pay for more of them -- and no reason to
    accept fewer, since a chord longer than a cell puts a visible corner
    into the contact set.

    Args:
        radius_m: Radius of the circle.
        cell_size_m: Grid ``dx``.
        minimum: Floor on the returned count.

    Returns:
        The facet count.

    Raises:
        SolverInputError: If the radius or the cell size is not positive.
    """
    radius = float(radius_m)
    cell = float(cell_size_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise SolverInputError(f"radius_m must be positive, got {radius_m!r}")
    if not math.isfinite(cell) or cell <= 0.0:
        raise SolverInputError(f"cell_size_m must be positive, got {cell_size_m!r}")
    ratio = min(cell / (2.0 * radius), 1.0)
    needed = int(math.ceil(math.pi / math.asin(ratio)))
    return max(needed, int(minimum), MIN_BALL_FACETS)


@dataclass(frozen=True, slots=True)
class BallContactSplit:
    """Where the impulse that reached the ball landed, qualitatively.

    Every vector here is an impulse **per unit out-of-plane width**, on a
    body that is an infinite cylinder rather than a sphere.  The pairs sum
    to :attr:`total_n_s` exactly, because each contact node is assigned to
    exactly one half of each pair.

    ADR-0033 permits this split to be reported and permits it to be
    reported *qualitatively*: which side of the ball the sand arrives on,
    and roughly in what proportion.  It does not permit the numbers to be
    read as a load distribution on a golf ball, which is why
    :attr:`is_qualitative` is a field on the value rather than a sentence
    in a caption that a figure will outlive.

    Attributes:
        below_equator_n_s: ``(2,)`` impulse on nodes below the centre.
        above_equator_n_s: ``(2,)`` impulse on nodes at or above it.
        face_side_n_s: ``(2,)`` impulse on the half the club approaches
            from.
        far_side_n_s: ``(2,)`` impulse on the half away from the club.
        total_n_s: ``(2,)`` total impulse in the ledger.
        n_contacts: Grid nodes the ball projected.
    """

    below_equator_n_s: NDArray[np.float64]
    above_equator_n_s: NDArray[np.float64]
    face_side_n_s: NDArray[np.float64]
    far_side_n_s: NDArray[np.float64]
    total_n_s: NDArray[np.float64]
    n_contacts: int

    @property
    def is_qualitative(self) -> bool:
        """Always True. The split is a direction, never a magnitude."""
        return True

    @property
    def total_magnitude_n_s(self) -> float:
        """Magnitude of the total impulse, per unit width."""
        return float(np.hypot(self.total_n_s[0], self.total_n_s[1]))

    @property
    def below_equator_fraction(self) -> float:
        """Share of the impulse magnitude that landed below the equator."""
        return _fraction(self.below_equator_n_s, self.above_equator_n_s)

    @property
    def face_side_fraction(self) -> float:
        """Share of the impulse magnitude that landed on the near half."""
        return _fraction(self.face_side_n_s, self.far_side_n_s)

    def heel_toe_fraction(self) -> NoReturn:
        """Refuse: heel-toe is a direction plane strain does not have.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def summary(self) -> str:
        """A statement fit for a figure caption or a run manifest."""
        return (
            f"ball contact split (qualitative, in-plane only): "
            f"{self.below_equator_fraction * 100:.0f}% below the equator, "
            f"{self.face_side_fraction * 100:.0f}% on the near half, over "
            f"{self.n_contacts} grid nodes carrying "
            f"{self.total_magnitude_n_s:.4g} N.s per metre of width.\n  "
            + PLANE_STRAIN_BALL_NOTE
        )


@dataclass(frozen=True)
class BallSection:
    """A golf ball as a rigid circular plane-strain body.

    The ball is held as its :class:`~bunkershot3d.solvers.mpm.body.RigidSection`
    together with the circle that section stands for, so the radius the
    polygon approximates survives every advance rather than having to be
    re-inferred from vertices.

    Attributes:
        section: The polygonal body the solver contacts. Its reference
            point is the ball centre.
        radius_m: Radius of the circle the section stands for.
        n_facets: Facets in that section.
    """

    section: RigidSection
    radius_m: float
    n_facets: int

    def __post_init__(self) -> None:
        radius = float(self.radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise SolverInputError(f"radius_m must be positive, got {self.radius_m!r}")
        if int(self.n_facets) < MIN_BALL_FACETS:
            raise SolverInputError(
                f"n_facets must be at least {MIN_BALL_FACETS}, got {self.n_facets!r}"
            )
        if not isinstance(self.section, RigidSection):
            raise SolverInputError(
                f"section must be a RigidSection, got {type(self.section).__name__}"
            )
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "n_facets", int(self.n_facets))

    # ---------------------------------------------------------- construction

    @classmethod
    def at(
        cls,
        centre_m: ArrayLike,
        *,
        radius_m: float = BALL_RADIUS_M,
        n_facets: int = DEFAULT_BALL_FACETS,
        velocity_m_s: ArrayLike = (0.0, 0.0),
        angular_velocity_rad_s: float = 0.0,
        friction: float = 0.3,
    ) -> BallSection:
        """Build a ball centred on a stated point.

        Args:
            centre_m: ``(2,)`` centre in world ``(x, z)``.
            radius_m: Ball radius, defaulting to the conforming minimum.
            n_facets: Facets of the polygonal circle. Pass
                :func:`n_facets_for_cell_size` when the grid is known.
            velocity_m_s: Linear velocity of the centre.
            angular_velocity_rad_s: Rotation rate about ``+y``.
            friction: Coulomb friction between ball and sand.

        Returns:
            The ball.
        """
        return cls(
            circular_section(
                centre_m,
                radius_m,
                n_facets=n_facets,
                velocity_m_s=velocity_m_s,
                angular_velocity_rad_s=angular_velocity_rad_s,
                friction=friction,
            ),
            float(radius_m),
            int(n_facets),
        )

    @classmethod
    def resting_on(
        cls,
        *,
        x_m: float,
        free_surface_height_m: float = 0.0,
        radius_m: float = BALL_RADIUS_M,
        n_facets: int = DEFAULT_BALL_FACETS,
        friction: float = 0.3,
    ) -> BallSection:
        """Build a ball sitting on the undisturbed surface, at rest.

        A ball in a bunker is normally *sunk* into the sand rather than
        tangent to it, and this places the tangent case deliberately: it
        is the one position that follows from the surface height alone,
        so a caller who wants a plugged lie states its own centre through
        :meth:`at` instead of inheriting a guess.

        Args:
            x_m: Horizontal position of the centre.
            free_surface_height_m: World ``z`` of the undisturbed sand.
            radius_m: Ball radius.
            n_facets: Facets of the polygonal circle.
            friction: Coulomb friction between ball and sand.

        Returns:
            The ball, tangent to the free surface and at rest.
        """
        return cls.at(
            (float(x_m), float(free_surface_height_m) + float(radius_m)),
            radius_m=radius_m,
            n_facets=n_facets,
            friction=friction,
        )

    # -------------------------------------------------------------- geometry

    @property
    def centre_m(self) -> NDArray[np.float64]:
        """``(2,)`` ball centre, which is the section's reference point."""
        return self.section.reference_point_m

    @property
    def velocity_m_s(self) -> NDArray[np.float64]:
        """``(2,)`` linear velocity of the centre."""
        return self.section.velocity_m_s

    @property
    def section_area_m2(self) -> float:
        """Area of the great circle, per unit out-of-plane width."""
        return self.section.area_m2

    def advanced(self, time_step_s: float) -> BallSection:
        """Return the ball after ``time_step_s`` of its own motion."""
        return self.with_section(self.section.advanced(time_step_s))

    def with_section(self, section: RigidSection) -> BallSection:
        """Return the same ball carrying a differently posed section.

        The solver hands back a bare :class:`RigidSection`; this puts the
        circle it stands for back around it, so the radius and the facet
        count are never re-derived from vertices.

        Args:
            section: The new section.

        Returns:
            The ball.
        """
        return BallSection(section, self.radius_m, self.n_facets)

    def geometry_note(self) -> str:
        """The infinite-cylinder statement, for a caption or a manifest."""
        return PLANE_STRAIN_BALL_NOTE

    # ------------------------------------------------------------ quantities

    def line_mass_kg_per_m(self, density_kg_m3: float) -> float:
        """Mass **per metre of out-of-plane width**, ``rho pi R^2``.

        The honest plane-strain quantity, named so it cannot be mistaken
        for a ball mass: multiplied by any effective width it gives the
        mass of a slice of an infinite cylinder, which is not 45.93 g and
        is not meant to be.

        Args:
            density_kg_m3: Bulk density of the ball.

        Returns:
            Line density in kg per metre of width.

        Raises:
            SolverInputError: If the density is not positive and finite.
        """
        density = float(density_kg_m3)
        if not math.isfinite(density) or density <= 0.0:
            raise SolverInputError(
                f"density_kg_m3 must be positive, got {density_kg_m3!r}"
            )
        return density * self.section_area_m2

    def sphere_mass_kg(self, density_kg_m3: float) -> NoReturn:
        """Refuse: this body has no sphere to take a mass of.

        Args:
            density_kg_m3: Ignored; the refusal does not depend on it.

        Raises:
            OutOfEnvelopeError: Always. ``rho (4/3) pi R^3`` is the mass of
                a sphere this model does not contain, and a number that
                can be returned will be quoted.
        """
        _ = density_kg_m3
        raise OutOfEnvelopeError(
            "F1's ball has no sphere mass to report.\n  " + PLANE_STRAIN_BALL_NOTE,
            verdict=None,
        )

    def launch_velocity_m_s(self) -> NoReturn:
        """Refuse: ball launch is F0's, on the #8657 momentum-transfer path.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.BALL_LAUNCH)
        raise AssertionError("unreachable")  # pragma: no cover

    def heel_toe_split(self) -> NoReturn:
        """Refuse: heel-toe is a direction plane strain does not have.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def lateral_distribution(self) -> NoReturn:
        """Refuse: there is no lateral axis to distribute anything over.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    # ----------------------------------------------------------------- split

    def split_contact(
        self, impulse: ContactImpulse, *, approach_direction: ArrayLike
    ) -> BallContactSplit:
        """Split one step's ledger below/above the equator and near/far.

        The equator is the horizontal through the ball centre, and the
        near half is the one the club arrives from -- the half whose
        outward normal opposes ``approach_direction``.  Both are directions
        the plane-strain section genuinely has, so both may be reported;
        neither is a load distribution on a golf ball.

        Args:
            impulse: The ledger this ball's contact projection returned.
            approach_direction: ``(2,)`` direction the club travels in.
                Need not be normalised; only its sign structure is used.

        Returns:
            The qualitative split.

        Raises:
            SolverInputError: If the approach direction has no length, so
                there is no near half to name.
        """
        direction = np.asarray(approach_direction, dtype=np.float64).reshape(-1)
        if direction.shape != (_DIMENSION,) or not np.all(np.isfinite(direction)):
            raise SolverInputError(
                f"approach_direction must be a finite 2-vector, got "
                f"{approach_direction!r}"
            )
        norm = float(np.hypot(direction[0], direction[1]))
        if norm < _MIN_DIRECTION_NORM:
            raise SolverInputError(
                "approach_direction has no length, so the ball has no near half "
                "to name; pass the club's travel direction"
            )

        vectors = np.asarray(impulse.impulse_n_s, dtype=np.float64).reshape(-1, 2)
        positions = np.asarray(impulse.position_m, dtype=np.float64).reshape(-1, 2)
        total = vectors.sum(axis=0) if vectors.size else np.zeros(2, dtype=np.float64)
        if vectors.shape[0] == 0:
            zero: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
            return BallContactSplit(
                zero, zero.copy(), zero.copy(), zero.copy(), total, 0
            )

        lever = positions - self.centre_m
        below = lever[:, 1] < 0.0
        near = (lever @ (direction / norm)) < 0.0
        return BallContactSplit(
            below_equator_n_s=vectors[below].sum(axis=0),
            above_equator_n_s=vectors[~below].sum(axis=0),
            face_side_n_s=vectors[near].sum(axis=0),
            far_side_n_s=vectors[~near].sum(axis=0),
            total_n_s=total,
            n_contacts=int(vectors.shape[0]),
        )


def _fraction(part: NDArray[np.float64], other: NDArray[np.float64]) -> float:
    """Share of the summed magnitudes carried by ``part``.

    Taken on magnitudes rather than on the resultant so that two halves
    pushing in opposite directions do not report as no contact at all.
    """
    first = float(np.hypot(part[0], part[1]))
    second = float(np.hypot(other[0], other[1]))
    total = first + second
    if total <= 0.0:
        return 0.0
    return first / total
