"""
Primitive shape inertia calculations.

This module provides analytical inertia formulas for primitive shapes
(box, cylinder, sphere, capsule) as a fallback when mesh-based
calculation is not available or not desired.

All formulas assume uniform density and return inertia about the
center of mass with principal axes aligned to the shape's natural axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
)


class PrimitiveShape(Enum):
    """Primitive geometry shapes."""

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    ELLIPSOID = "ellipsoid"


@dataclass
class PrimitiveInertiaCalculator:
    """
    Calculate inertia tensors for primitive shapes.

    This class provides analytical formulas for common shapes,
    useful as a fallback when meshes are unavailable or for
    quick approximations.
    """

    @staticmethod
    def compute_box(
        mass: float, size_x: float, size_y: float, size_z: float
    ) -> InertiaResult:
        """
        Compute inertia for a solid box (rectangular parallelepiped).

        The box is centered at the origin with edges aligned to axes.

        Args:
            mass: Mass in kg
            size_x: Width (X dimension) in meters
            size_y: Depth (Y dimension) in meters
            size_z: Height (Z dimension) in meters

        Returns:
            InertiaResult for the box
        """
        # I_xx = (1/12) * m * (y^2 + z^2)
        # I_yy = (1/12) * m * (x^2 + z^2)
        # I_zz = (1/12) * m * (x^2 + y^2)
        factor = mass / 12.0

        ixx = factor * (size_y**2 + size_z**2)
        iyy = factor * (size_x**2 + size_z**2)
        izz = factor * (size_x**2 + size_y**2)

        volume = size_x * size_y * size_z

        return InertiaResult(
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            volume=volume,
            mass=mass,
            mode=InertiaMode.PRIMITIVE_APPROXIMATION,
        )

    @staticmethod
    def compute_cylinder(
        mass: float, radius: float, length: float, axis: str = "z"
    ) -> InertiaResult:
        """
        Compute inertia for a solid cylinder.

        The cylinder is centered at the origin with its longitudinal
        axis along the specified axis.

        Args:
            mass: Mass in kg
            radius: Radius in meters
            length: Length (height) in meters
            axis: Longitudinal axis ('x', 'y', or 'z')

        Returns:
            InertiaResult for the cylinder
        """
        # For cylinder along Z:
        # I_xx = I_yy = (1/12) * m * (3*r^2 + h^2)
        # I_zz = (1/2) * m * r^2

        r2 = radius**2
        h2 = length**2

        i_longitudinal = 0.5 * mass * r2
        i_transverse = (1.0 / 12.0) * mass * (3.0 * r2 + h2)

        volume = math.pi * r2 * length

        # Assign based on axis
        if axis.lower() == "z":
            ixx, iyy, izz = i_transverse, i_transverse, i_longitudinal
        elif axis.lower() == "y":
            ixx, iyy, izz = i_transverse, i_longitudinal, i_transverse
        else:  # x
            ixx, iyy, izz = i_longitudinal, i_transverse, i_transverse

        return InertiaResult(
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            volume=volume,
            mass=mass,
            mode=InertiaMode.PRIMITIVE_APPROXIMATION,
        )

    @staticmethod
    def compute_sphere(mass: float, radius: float) -> InertiaResult:
        """
        Compute inertia for a solid sphere.

        Args:
            mass: Mass in kg
            radius: Radius in meters

        Returns:
            InertiaResult for the sphere
        """
        # I = (2/5) * m * r^2 for all axes
        i_sphere = 0.4 * mass * radius**2
        volume = (4.0 / 3.0) * math.pi * radius**3

        return InertiaResult(
            ixx=i_sphere,
            iyy=i_sphere,
            izz=i_sphere,
            volume=volume,
            mass=mass,
            mode=InertiaMode.PRIMITIVE_APPROXIMATION,
        )

    @staticmethod
    def compute_capsule(
        mass: float, radius: float, length: float, axis: str = "z"
    ) -> InertiaResult:
        """
        Compute inertia for a capsule (cylinder with hemispherical caps).

        The capsule is centered at the origin with its longitudinal
        axis along the specified axis.

        Args:
            mass: Mass in kg
            radius: Radius of cylinder and hemispheres in meters
            length: Length of cylindrical portion in meters
            axis: Longitudinal axis ('x', 'y', or 'z')

        Returns:
            InertiaResult for the capsule
        """
        r = radius
        h = length  # Cylinder length (not including caps)

        # Volumes
        v_cylinder = math.pi * r**2 * h
        v_sphere = (4.0 / 3.0) * math.pi * r**3  # Both hemispheres = one sphere
        v_total = v_cylinder + v_sphere

        # Mass distribution (proportional to volume assuming uniform density)
        m_cylinder = mass * (v_cylinder / v_total)
        m_sphere = mass * (v_sphere / v_total)

        # Cylinder inertia (along Z)
        i_cyl_long = 0.5 * m_cylinder * r**2
        i_cyl_trans = (1.0 / 12.0) * m_cylinder * (3.0 * r**2 + h**2)

        # Sphere inertia at sphere center
        i_sphere_cm = 0.4 * m_sphere * r**2

        # Parallel axis theorem to move hemisphere centers to capsule center
        # Each hemisphere center is at +/- (h/2 + 2r/3) from capsule center
        # But for a capsule, the hemisphere centers are at +/- h/2
        # (the hemisphere center of mass is at 3r/8 from the flat face)
        d_hemi = h / 2.0 + (3.0 * r / 8.0)

        # Transverse inertia of sphere at capsule center (parallel axis)
        i_sphere_trans = i_sphere_cm + m_sphere * d_hemi**2

        # Total inertia
        i_longitudinal = i_cyl_long + i_sphere_cm
        i_transverse = i_cyl_trans + i_sphere_trans

        # Assign based on axis
        if axis.lower() == "z":
            ixx, iyy, izz = i_transverse, i_transverse, i_longitudinal
        elif axis.lower() == "y":
            ixx, iyy, izz = i_transverse, i_longitudinal, i_transverse
        else:  # x
            ixx, iyy, izz = i_longitudinal, i_transverse, i_transverse

        return InertiaResult(
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            volume=v_total,
            mass=mass,
            mode=InertiaMode.PRIMITIVE_APPROXIMATION,
        )

    @staticmethod
    def compute_ellipsoid(
        mass: float, semi_a: float, semi_b: float, semi_c: float
    ) -> InertiaResult:
        """
        Compute inertia for a solid ellipsoid.

        The ellipsoid is centered at origin with semi-axes aligned to
        coordinate axes.

        Args:
            mass: Mass in kg
            semi_a: Semi-axis along X in meters
            semi_b: Semi-axis along Y in meters
            semi_c: Semi-axis along Z in meters

        Returns:
            InertiaResult for the ellipsoid
        """
        # I_xx = (1/5) * m * (b^2 + c^2)
        # I_yy = (1/5) * m * (a^2 + c^2)
        # I_zz = (1/5) * m * (a^2 + b^2)
        factor = mass / 5.0

        ixx = factor * (semi_b**2 + semi_c**2)
        iyy = factor * (semi_a**2 + semi_c**2)
        izz = factor * (semi_a**2 + semi_b**2)

        volume = (4.0 / 3.0) * math.pi * semi_a * semi_b * semi_c

        return InertiaResult(
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            volume=volume,
            mass=mass,
            mode=InertiaMode.PRIMITIVE_APPROXIMATION,
        )

    @classmethod
    def compute(
        cls,
        shape: PrimitiveShape | str,
        mass: float,
        dimensions: dict[str, float] | tuple[float, ...],
        axis: str = "z",
    ) -> InertiaResult:
        """
        Compute inertia for any primitive shape.

        Args:
            shape: PrimitiveShape enum or string name
            mass: Mass in kg
            dimensions: Shape-specific dimensions
                - box: (size_x, size_y, size_z) or {'x': ..., 'y': ..., 'z': ...}
                - cylinder: (radius, length) or {'radius': ..., 'length': ...}
                - sphere: (radius,) or {'radius': ...}
                - capsule: (radius, length) or {'radius': ..., 'length': ...}
                - ellipsoid: (semi_a, semi_b, semi_c) or {'a': ..., 'b': ..., 'c': ...}
            axis: Longitudinal axis for cylinder/capsule

        Returns:
            InertiaResult for the shape
        """
        if isinstance(shape, str):
            shape = PrimitiveShape(shape.lower())

        # Convert tuple to dict if needed
        if isinstance(dimensions, tuple):
            dimensions = cls._tuple_to_dict(shape, dimensions)

        if shape == PrimitiveShape.BOX:
            return cls.compute_box(
                mass,
                dimensions.get("x", dimensions.get("size_x", 0.1)),
                dimensions.get("y", dimensions.get("size_y", 0.1)),
                dimensions.get("z", dimensions.get("size_z", 0.1)),
            )
        elif shape == PrimitiveShape.CYLINDER:
            return cls.compute_cylinder(
                mass,
                dimensions.get("radius", 0.05),
                dimensions.get("length", dimensions.get("height", 0.1)),
                axis,
            )
        elif shape == PrimitiveShape.SPHERE:
            return cls.compute_sphere(mass, dimensions.get("radius", 0.05))
        elif shape == PrimitiveShape.CAPSULE:
            return cls.compute_capsule(
                mass,
                dimensions.get("radius", 0.05),
                dimensions.get("length", dimensions.get("height", 0.1)),
                axis,
            )
        elif shape == PrimitiveShape.ELLIPSOID:
            return cls.compute_ellipsoid(
                mass,
                dimensions.get("a", dimensions.get("semi_a", 0.1)),
                dimensions.get("b", dimensions.get("semi_b", 0.1)),
                dimensions.get("c", dimensions.get("semi_c", 0.1)),
            )
        else:
            raise ValueError(f"Unknown shape: {shape}")

    @staticmethod
    def _tuple_to_dict(
        shape: PrimitiveShape, dims: tuple[float, ...]
    ) -> dict[str, float]:
        """Convert dimension tuple to dictionary."""
        if shape == PrimitiveShape.BOX:
            if len(dims) >= 3:
                return {"x": dims[0], "y": dims[1], "z": dims[2]}
            elif len(dims) == 1:
                return {"x": dims[0], "y": dims[0], "z": dims[0]}
        elif shape in (PrimitiveShape.CYLINDER, PrimitiveShape.CAPSULE):
            if len(dims) >= 2:
                return {"radius": dims[0], "length": dims[1]}
            elif len(dims) == 1:
                return {"radius": dims[0], "length": dims[0] * 2}
        elif shape == PrimitiveShape.SPHERE:
            return {"radius": dims[0]}
        elif shape == PrimitiveShape.ELLIPSOID:
            if len(dims) >= 3:
                return {"a": dims[0], "b": dims[1], "c": dims[2]}
            elif len(dims) == 1:
                return {"a": dims[0], "b": dims[0], "c": dims[0]}

        # Default fallback
        return {"radius": dims[0] if dims else 0.1}


def estimate_segment_primitive(
    segment_type: str,
    length: float,
    width: float | None = None,
    depth: float | None = None,
) -> tuple[PrimitiveShape, dict[str, float]]:
    """
    Estimate best primitive shape for a body segment.

    Args:
        segment_type: Type of segment (head, thigh, forearm, etc.)
        length: Segment length in meters
        width: Segment width in meters (optional)
        depth: Segment depth in meters (optional)

    Returns:
        Tuple of (PrimitiveShape, dimension_dict)
    """
    # Default width/depth as fractions of length if not specified
    if width is None:
        width = length * 0.2
    if depth is None:
        depth = length * 0.15

    segment_lower = segment_type.lower()

    # Head -> sphere
    if "head" in segment_lower:
        radius = length / 2
        return PrimitiveShape.SPHERE, {"radius": radius}

    # Limb segments -> capsule
    if any(
        x in segment_lower for x in ["arm", "forearm", "thigh", "shin", "shank", "leg"]
    ):
        radius = (width + depth) / 4  # Average of width/depth divided by 2
        cyl_length = max(0.01, length - 2 * radius)  # Subtract hemispherical caps
        return PrimitiveShape.CAPSULE, {"radius": radius, "length": cyl_length}

    # Torso segments -> box or ellipsoid
    if any(x in segment_lower for x in ["torso", "thorax", "lumbar", "pelvis"]):
        return PrimitiveShape.BOX, {"x": width, "y": depth, "z": length}

    # Hands/feet -> box
    if any(x in segment_lower for x in ["hand", "foot"]):
        return PrimitiveShape.BOX, {"x": width, "y": length, "z": depth}

    # Neck -> cylinder
    if "neck" in segment_lower:
        radius = (width + depth) / 4
        return PrimitiveShape.CYLINDER, {"radius": radius, "length": length}

    # Default -> capsule
    radius = (width + depth) / 4
    cyl_length = max(0.01, length - 2 * radius)
    return PrimitiveShape.CAPSULE, {"radius": radius, "length": cyl_length}
