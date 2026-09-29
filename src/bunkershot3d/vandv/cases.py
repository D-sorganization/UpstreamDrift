"""Analytic verification cases with closed-form answers (issue #8616).

Code verification, so **no experimental data appears here**.  Each case
has an exact answer derived on paper, and the only error left in the
comparison is the solver's own arithmetic and quadrature.

The three cases
---------------

**Quasi-static flat plate.**  A downward-facing horizontal plate of area
``A`` at depth ``|z|``, moving straight down.  The local frame collapses
to ``beta = 0``, ``psi = 0``, ``gamma = pi/2``, the generic response is
purely vertical, the surface-friction cutoff has no tangential component
to act on, and with ``delta_h = 0`` the depth force is exactly::

    F_z = xi_n * alpha_z_generic(0, pi/2, 0) * |z| * A

which is the *inverse* of the one-shot calibration in
``MaterialResponse.from_vertical_plate_intrusion``.  The case therefore
closes the loop between the calibration and the forward solve, and a sign
or factor error in either shows up as a mismatch.

**Zero-speed limit.**  The inertial traction is ``lambda rho v_n^2``, so
it must vanish quadratically as the speed goes to zero and the solver
must degrade exactly to quasi-static RFT.  The same algebra gives an
exact crossing: with ``delta_h = 0`` the two terms are equal when
``v = MaterialResponse.crossover_speed_m_s(|z|)``, which links the
material model and the solver by an identity rather than by a comment.

**Cylinder, for the order of accuracy.**  For a circular cylinder of
radius ``R`` and length ``L`` with its axis across the flow, moving at
``v`` along ``+x``, the inertial traction integrates in closed form.
Leading elements are those with ``cos(theta) >= 0``, so::

    F_x = -lambda rho v^2 R L * integral(cos^3, -pi/2, pi/2)
        = -(4/3) lambda rho v^2 R L

and ``F_z = 0`` by reflection symmetry.  Discretising the side surface
into ``N`` flat facets makes this a pure quadrature-error problem: the
composite midpoint rule lands on an integrand whose first derivative
vanishes at both limits, so the leading error comes from the chord area
``2 R sin(dtheta/2)`` against the arc ``R dtheta`` -- a relative error of
``-dtheta^2/24``.  **Second order**, and that is what the refinement
study should recover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..solvers import (
    VERTICAL_PLATE_ALPHA_Z,
    DRFTSolver,
    IntrusionState,
    MaterialResponse,
    RefusalPolicy,
    SurfaceElements,
    ZeroDepression,
)
from .exceptions import VerificationError

__all__ = [
    "CylinderVerificationCase",
    "FlatPlateVerificationCase",
    "asymmetric_body_elements",
    "cylinder_case",
    "cylinder_inertial_force_n",
    "cylinder_side_elements",
    "flat_plate_depth_force_n",
    "flat_plate_elements",
    "flat_plate_inertial_force_n",
    "leading_edge_fraction",
    "quasi_static_plate_case",
    "quasi_static_solver",
    "spiral_body_elements",
]

_GOLDEN_ANGLE_RAD = math.pi * (3.0 - math.sqrt(5.0))


def quasi_static_solver(material: MaterialResponse) -> DRFTSolver:
    """A solver whose structural correction is exactly zero.

    ``delta_h = 0`` is the *measured* plate limit, which is what makes the
    closed forms in this module exact rather than approximate.  The
    refusal policy is reporting, because every bunker-relevant speed is
    outside the published envelope by design and a code-verification case
    is about arithmetic, not about whether the physics applies.

    Args:
        material: Sand response constants.

    Returns:
        A reporting :class:`~bunkershot3d.solvers.DRFTSolver` with
        :class:`~bunkershot3d.solvers.ZeroDepression`.
    """
    return DRFTSolver(
        material=material,
        structural_correction=ZeroDepression(),
        refusal_policy=RefusalPolicy.REPORT,
    )


def flat_plate_elements(*, area_m2: float, depth_m: float) -> SurfaceElements:
    """One downward-facing horizontal element at ``depth_m`` below the surface.

    Args:
        area_m2: Plate area.
        depth_m: Depth below the free surface, positive downward.

    Returns:
        A single-element surface.

    Raises:
        VerificationError: If either input is not positive and finite.
    """
    for name, value in (("area_m2", area_m2), ("depth_m", depth_m)):
        if not math.isfinite(value) or value <= 0.0:
            raise VerificationError(f"{name} must be positive, got {value!r}")
    return SurfaceElements(
        [[0.0, 0.0, -float(depth_m)]], [[0.0, 0.0, -1.0]], [float(area_m2)]
    )


def flat_plate_depth_force_n(
    material: MaterialResponse, *, area_m2: float, depth_m: float
) -> float:
    """``xi_n * alpha_z_generic(0, pi/2, 0) * |z| * A``, the exact depth force.

    Args:
        material: Sand response constants.
        area_m2: Plate area.
        depth_m: Depth below the free surface, positive downward.

    Returns:
        The upward (``+z``) force magnitude in newtons.
    """
    return (
        material.normal_stress_scale_pa_per_m
        * VERTICAL_PLATE_ALPHA_Z
        * float(depth_m)
        * float(area_m2)
    )


def flat_plate_inertial_force_n(
    material: MaterialResponse, *, area_m2: float, speed_m_s: float
) -> float:
    """``lambda rho v^2 A``, the exact inertial force on the same plate.

    Args:
        material: Sand response constants.
        area_m2: Plate area.
        speed_m_s: Downward speed.

    Returns:
        The upward (``+z``) force magnitude in newtons.
    """
    return (
        material.inertial_stress_scale_pa_s2_per_m2
        * float(speed_m_s) ** 2
        * float(area_m2)
    )


@dataclass(frozen=True)
class FlatPlateVerificationCase:
    """A horizontal plate driven straight down, with its exact answer.

    Attributes:
        elements: The single-element surface.
        area_m2: Plate area.
        depth_m: Depth below the free surface.
        speed_m_s: Downward speed.
        exact_depth_force_n: Exact ``+z`` depth force, ``delta_h = 0``.
        exact_inertial_force_n: Exact ``+z`` inertial force.
    """

    elements: SurfaceElements
    area_m2: float
    depth_m: float
    speed_m_s: float
    exact_depth_force_n: float
    exact_inertial_force_n: float

    def state(self) -> IntrusionState:
        """The intrusion query for this case."""
        return IntrusionState(self.elements, (0.0, 0.0, -self.speed_m_s))


def quasi_static_plate_case(
    material: MaterialResponse,
    *,
    area_m2: float = 20e-3 * 80e-3,
    depth_m: float = 0.040,
    speed_m_s: float = 0.05,
) -> FlatPlateVerificationCase:
    """Build the flat-plate case with both exact forces precomputed.

    The default is the 20 x 80 mm sole footprint at a 40 mm divot depth,
    the geometry the research digest quotes its crossover speed for.

    Args:
        material: Sand response constants.
        area_m2: Plate area.
        depth_m: Depth below the free surface.
        speed_m_s: Downward speed. The default is far below the ~7 m/s
            crossover, which is what makes it the *quasi-static* limit.

    Returns:
        The case.
    """
    return FlatPlateVerificationCase(
        elements=flat_plate_elements(area_m2=area_m2, depth_m=depth_m),
        area_m2=float(area_m2),
        depth_m=float(depth_m),
        speed_m_s=float(speed_m_s),
        exact_depth_force_n=flat_plate_depth_force_n(
            material, area_m2=area_m2, depth_m=depth_m
        ),
        exact_inertial_force_n=flat_plate_inertial_force_n(
            material, area_m2=area_m2, speed_m_s=speed_m_s
        ),
    )


def cylinder_side_elements(
    *,
    n_facets: int,
    radius_m: float = 0.020,
    length_m: float = 0.080,
    centre_depth_m: float = 0.200,
) -> SurfaceElements:
    """Facet the curved side of a cylinder whose axis lies along ``y``.

    The facet count must be a multiple of four so that no facet normal
    lands exactly on ``cos(theta) = 0``.  If one did, the leading-edge
    test ``v . n >= 0`` would include a zero-contribution facet on one
    refinement level and exclude it on the next, and the refinement study
    would be measuring the tie-break rather than the quadrature.

    Args:
        n_facets: Facets around the circumference, a multiple of four.
        radius_m: Cylinder radius.
        length_m: Cylinder length along ``y``.
        centre_depth_m: Depth of the axis below the free surface. The
            default buries the whole body an order of magnitude deeper
            than its radius, so every facet is submerged at every
            refinement level.

    Returns:
        One element per facet, over the full circumference.

    Raises:
        VerificationError: If the facet count is not a positive multiple
            of four, if a dimension is not positive, or if the body would
            break the free surface.
    """
    count = int(n_facets)
    if count < 4 or count % 4 != 0:
        raise VerificationError(
            f"n_facets must be a positive multiple of 4, got {n_facets!r}; "
            "any other count puts a facet normal on cos(theta) = 0, where the "
            "leading-edge test becomes a tie-break and the refinement study "
            "measures the tie-break instead of the quadrature"
        )
    for name, value in (
        ("radius_m", radius_m),
        ("length_m", length_m),
        ("centre_depth_m", centre_depth_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise VerificationError(f"{name} must be positive, got {value!r}")
    if radius_m >= centre_depth_m:
        raise VerificationError(
            f"a cylinder of radius {radius_m!r} m centred {centre_depth_m!r} m "
            "below the surface would break the free surface; the closed form "
            "assumes every facet is submerged"
        )

    step = 2.0 * math.pi / count
    angles = (np.arange(count, dtype=np.float64) + 0.5) * step
    normals = np.stack([np.cos(angles), np.zeros(count), np.sin(angles)], axis=1)
    # The facet centroid sits on the chord, at radius R cos(dtheta/2), and the
    # facet area is the chord width times the length -- both the polyhedral
    # values a real triangulation would produce, not the arc values.
    centroids = normals * (radius_m * math.cos(0.5 * step))
    centroids[:, 2] -= centre_depth_m
    areas = np.full(count, 2.0 * radius_m * math.sin(0.5 * step) * length_m)
    return SurfaceElements(centroids, normals, areas)


def cylinder_inertial_force_n(
    material: MaterialResponse,
    *,
    radius_m: float = 0.020,
    length_m: float = 0.080,
    speed_m_s: float = 25.0,
) -> float:
    """``-(4/3) lambda rho v^2 R L``: the exact ``x`` inertial force.

    Derived by integrating ``-n lambda rho (v . n)^2`` over the leading
    half of the cylinder; see the module docstring.

    Args:
        material: Sand response constants.
        radius_m: Cylinder radius.
        length_m: Cylinder length.
        speed_m_s: Speed along ``+x``.

    Returns:
        The ``x`` component in newtons, negative because the medium
        resists the motion.
    """
    return -(4.0 / 3.0) * (
        material.inertial_stress_scale_pa_s2_per_m2
        * float(speed_m_s) ** 2
        * float(radius_m)
        * float(length_m)
    )


@dataclass(frozen=True)
class CylinderVerificationCase:
    """One refinement level of the cylinder case.

    Attributes:
        elements: The faceted side surface.
        n_facets: Facets around the circumference.
        speed_m_s: Speed along ``+x``.
        exact_inertial_force_x_n: The closed-form answer.
        facet_width_m: Chord width of one facet, ``2 R sin(pi/N)``.
    """

    elements: SurfaceElements
    n_facets: int
    speed_m_s: float
    exact_inertial_force_x_n: float
    facet_width_m: float

    @property
    def cell_size_m(self) -> float:
        """Representative cell size: the facet chord width.

        **Not** the root mean element area.  This refinement varies in one
        direction only -- the traction is uniform along the cylinder axis,
        so subdividing axially changes no digit of the answer -- and
        ``sqrt(A/n)`` would therefore fall as ``1/sqrt(N)`` while the error
        falls as ``1/N^2``, reporting a fourth-order scheme for a
        second-order quadrature.  The representative size of a
        one-directional refinement is that direction's cell size.
        """
        return self.facet_width_m

    def state(self) -> IntrusionState:
        """The intrusion query for this level."""
        return IntrusionState(self.elements, (self.speed_m_s, 0.0, 0.0))


def cylinder_case(
    material: MaterialResponse,
    *,
    n_facets: int,
    radius_m: float = 0.020,
    length_m: float = 0.080,
    speed_m_s: float = 25.0,
    centre_depth_m: float = 0.200,
) -> CylinderVerificationCase:
    """Build one refinement level of the cylinder case.

    Args:
        material: Sand response constants.
        n_facets: Facets around the circumference, a multiple of four.
        radius_m: Cylinder radius.
        length_m: Cylinder length along ``y``.
        speed_m_s: Speed along ``+x``.
        centre_depth_m: Depth of the axis below the free surface.

    Returns:
        The level, with its exact answer attached.
    """
    return CylinderVerificationCase(
        elements=cylinder_side_elements(
            n_facets=n_facets,
            radius_m=radius_m,
            length_m=length_m,
            centre_depth_m=centre_depth_m,
        ),
        n_facets=int(n_facets),
        speed_m_s=float(speed_m_s),
        exact_inertial_force_x_n=cylinder_inertial_force_n(
            material, radius_m=radius_m, length_m=length_m, speed_m_s=speed_m_s
        ),
        facet_width_m=2.0 * float(radius_m) * math.sin(math.pi / int(n_facets)),
    )


def asymmetric_body_elements(
    *, n_elements: int = 64, radius_m: float = 0.030, centre_depth_m: float = 0.150
) -> SurfaceElements:
    """A deliberately lopsided closed-ish body, for the torque tests.

    Normals are laid out on a golden-angle spiral so they span the whole
    sphere of directions, and the areas ramp linearly so that no
    reflection or rotation of the body maps it onto itself.  A *symmetric*
    body would give a torque with vanishing components, and a test on a
    vanishing component cannot see an axis swap -- which is exactly the
    defect class the angular-momentum test exists to catch.

    No random number generator is involved: the layout is a closed-form
    spiral, so the fixture is bit-identical on every platform and no seed
    has to be recorded.

    Args:
        n_elements: Number of elements.
        radius_m: Radius the centroids sit at.
        centre_depth_m: Depth of the body centre below the free surface.

    Returns:
        The element arrays.

    Raises:
        VerificationError: If the body would break the free surface or the
            element count is too small to be lopsided.
    """
    count = int(n_elements)
    if count < 8:
        raise VerificationError(
            f"n_elements must be at least 8 to produce a genuinely "
            f"three-dimensional torque, got {n_elements!r}"
        )
    if radius_m >= centre_depth_m:
        raise VerificationError(
            f"a body of radius {radius_m!r} m centred {centre_depth_m!r} m "
            "below the surface would break the free surface"
        )
    elements = spiral_body_elements(n_elements=count, radius_m=radius_m)
    return elements.translated((0.0, 0.0, -float(centre_depth_m)))


def spiral_body_elements(
    *, n_elements: int = 64, radius_m: float = 0.030
) -> SurfaceElements:
    """The same lopsided body, centred on the **body-frame** origin.

    This is the form :func:`~bunkershot3d.solvers.simulate_shot` wants,
    since it applies its own rigid transform.

    Args:
        n_elements: Number of elements.
        radius_m: Radius the centroids sit at.

    Returns:
        The element arrays, centred on the origin.

    Raises:
        VerificationError: If the element count or radius is unusable.
    """
    count = int(n_elements)
    if count < 8:
        raise VerificationError(
            f"n_elements must be at least 8 to produce a genuinely "
            f"three-dimensional torque, got {n_elements!r}"
        )
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise VerificationError(f"radius_m must be positive, got {radius_m!r}")
    index = np.arange(count, dtype=np.float64)
    cos_polar = 1.0 - 2.0 * (index + 0.5) / count
    sin_polar = np.sqrt(np.maximum(1.0 - cos_polar**2, 0.0))
    azimuth = index * _GOLDEN_ANGLE_RAD
    normals = np.stack(
        [sin_polar * np.cos(azimuth), sin_polar * np.sin(azimuth), cos_polar],
        axis=1,
    )
    base = 4.0 * math.pi * radius_m**2 / count
    return SurfaceElements(
        normals * float(radius_m), normals, base * (1.0 + 0.6 * index / count)
    )


def leading_edge_fraction(
    elements: SurfaceElements, velocity: NDArray[np.float64]
) -> float:
    """Share of the total area that is a leading edge for ``velocity``.

    Reported by the verification tests so a case that engages almost
    nothing is visible rather than silently trivial.

    Args:
        elements: The surface.
        velocity: A ``(3,)`` velocity vector.

    Returns:
        The engaged area fraction.
    """
    leading = (elements.normals @ np.asarray(velocity, dtype=np.float64)) >= 0.0
    total = elements.total_area_m2
    if total <= 0.0:
        return 0.0
    return float(elements.areas_m2[leading].sum() / total)
