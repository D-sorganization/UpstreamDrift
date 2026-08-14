"""Native mass properties by divergence-theorem integration (issue #8609).

Volume, centroid and the full inertia tensor are computed exactly for a
polyhedron by summing signed tetrahedra spanned from the origin to each
triangle - the divergence theorem applied to a closed, outward-wound
surface.  ``trimesh`` is used as an *independent cross-check in tests*
only: per ADR-0032 an OEM tool must be able to verify its own numbers,
and ``trimesh`` is not a dependency of this repo.

For a tetrahedron with one vertex at the origin and edge matrix
``J = [a b c]``, the second-moment integral over the standard simplex
gives

    C = det(J)/120 * [ (a+b+c)(a+b+c)^T + a a^T + b b^T + c c^T ]

which is exact in floating point for polyhedral solids; curved solids
carry only the tessellation error of their mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mesh import TriangleMesh, require_watertight

__all__ = ["MassProperties", "compute_mass_properties"]


@dataclass(frozen=True, eq=False)
class MassProperties:
    """Rigid-body mass properties of a closed mesh, in SI units.

    Attributes:
        volume_m3: Enclosed volume.
        centroid_m: Centre of mass, assuming uniform density.
        inertia_kg_m2: Inertia tensor about the centroid, in the mesh's
            own axes.
        mass_kg: Total mass.
        density_kg_m3: Uniform density used.
    """

    volume_m3: float
    centroid_m: NDArray[np.float64]
    inertia_kg_m2: NDArray[np.float64]
    mass_kg: float
    density_kg_m3: float

    @property
    def inertia_about_origin_kg_m2(self) -> NDArray[np.float64]:
        """Inertia tensor about the mesh-frame origin (parallel axis)."""
        offset = self.centroid_m
        shift = self.mass_kg * (
            float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
        )
        return self.inertia_kg_m2 + shift

    @property
    def principal_moments_kg_m2(self) -> NDArray[np.float64]:
        """Ascending principal moments about the centroid."""
        return np.asarray(np.linalg.eigvalsh(self.inertia_kg_m2), dtype=np.float64)

    def inertia_about_axis_kg_m2(
        self, point_m: ArrayLike, direction: ArrayLike
    ) -> float:
        """Moment of inertia about an arbitrary axis.

        For a wedge this is how the shaft-axis MOI - the head's
        resistance to the sand torquing the face open between entry and
        the ball - is obtained.

        Args:
            point_m: Any point on the axis, in metres.
            direction: Axis direction; normalised internally.

        Returns:
            The moment of inertia in kg.m^2.

        Raises:
            ValueError: If the direction is degenerate or non-finite.
        """
        axis = np.asarray(direction, dtype=np.float64).reshape(-1)
        origin = np.asarray(point_m, dtype=np.float64).reshape(-1)
        if axis.shape != (3,) or origin.shape != (3,):
            raise ValueError("point_m and direction must both be 3-vectors")
        if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(origin)):
            raise ValueError("axis point and direction must be finite")
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError("axis direction must be a non-zero vector")
        unit = axis / norm
        offset = self.centroid_m - origin
        perpendicular_sq = float(offset @ offset) - float(offset @ unit) ** 2
        moment = float(unit @ self.inertia_kg_m2 @ unit) + self.mass_kg * max(
            perpendicular_sq, 0.0
        )
        return moment


def _second_moment_about_origin(
    mesh: TriangleMesh,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(volume, first moment, second moment)`` about the origin."""
    first, second, third = mesh.triangle_corners()
    determinants = np.einsum("ij,ij->i", first, np.cross(second, third))
    volume = float(determinants.sum() / 6.0)

    corner_sum = first + second + third
    first_moment = np.einsum("i,ij->j", determinants, corner_sum) / 24.0

    outer = (
        np.einsum("ij,ik->ijk", corner_sum, corner_sum)
        + np.einsum("ij,ik->ijk", first, first)
        + np.einsum("ij,ik->ijk", second, second)
        + np.einsum("ij,ik->ijk", third, third)
    )
    second_moment = np.einsum("i,ijk->jk", determinants / 120.0, outer)
    return volume, first_moment, second_moment


def compute_mass_properties(
    mesh: TriangleMesh,
    *,
    density_kg_m3: float | None = None,
    mass_kg: float | None = None,
) -> MassProperties:
    """Exact mass properties of a watertight mesh.

    Exactly one of ``density_kg_m3`` or ``mass_kg`` must be supplied;
    supplying both would allow the pair to disagree.

    Args:
        mesh: A closed, outward-wound manifold mesh.
        density_kg_m3: Uniform density.
        mass_kg: Total mass, from which the density is derived.

    Returns:
        The mass properties about the mesh frame and its centroid.

    Raises:
        MeshValidationError: If the mesh is not a watertight solid.
        ValueError: If the mass/density specification is invalid.
    """
    if (density_kg_m3 is None) == (mass_kg is None):
        raise ValueError(
            "supply exactly one of density_kg_m3 or mass_kg "
            f"(got density={density_kg_m3!r}, mass={mass_kg!r})"
        )
    require_watertight(mesh, context="mass-property integration")

    volume, first_moment, second_moment = _second_moment_about_origin(mesh)
    if volume <= 0.0 or not math.isfinite(volume):
        raise ValueError(f"enclosed volume must be positive, got {volume!r}")

    if mass_kg is None:
        density = float(density_kg_m3)  # type: ignore[arg-type]
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError(f"density_kg_m3 must be positive, got {density_kg_m3!r}")
        mass = density * volume
    else:
        mass = float(mass_kg)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"mass_kg must be positive, got {mass_kg!r}")
        density = mass / volume

    centroid = first_moment / volume
    covariance = density * second_moment
    inertia_origin = np.trace(covariance) * np.eye(3) - covariance
    inertia_centroid = inertia_origin - mass * (
        float(centroid @ centroid) * np.eye(3) - np.outer(centroid, centroid)
    )
    inertia_centroid = 0.5 * (inertia_centroid + inertia_centroid.T)

    centroid.flags.writeable = False
    inertia_centroid.flags.writeable = False
    return MassProperties(
        volume_m3=volume,
        centroid_m=centroid,
        inertia_kg_m2=inertia_centroid,
        mass_kg=mass,
        density_kg_m3=density,
    )
