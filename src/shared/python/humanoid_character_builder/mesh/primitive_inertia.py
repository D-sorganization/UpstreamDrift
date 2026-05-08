"""
Primitive shape inertia calculations.

This module is a thin re-export of model_generation.inertia.primitives
with humanoid-specific helpers to maintain backward compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
)

from model_generation.inertia.primitives import (
    box_inertia,
    cylinder_inertia,
    sphere_inertia,
    capsule_inertia,
    ellipsoid_inertia,
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
    """Calculate inertia tensors for primitive shapes (compatibility wrapper)."""

    @staticmethod
    def compute_box(
        mass: float, size_x: float, size_y: float, size_z: float
    ) -> InertiaResult:
        if not (mass is not None):
            raise ValueError("mass must be provided")
        i_dict = box_inertia(mass, size_x, size_y, size_z)
        vol = size_x * size_y * size_z
        return InertiaResult(
            ixx=i_dict["ixx"],
            iyy=i_dict["iyy"],
            izz=i_dict["izz"],
            volume=vol,
            mass=mass,
            mode=InertiaMode.PRIMITIVE,
        )

    @staticmethod
    def compute_cylinder(
        mass: float, radius: float, length: float, axis: str = "z"
    ) -> InertiaResult:
        if not (mass is not None):
            raise ValueError("mass must be provided")
        i_dict = cylinder_inertia(mass, radius, length, axis)
        vol = math.pi * radius**2 * length
        return InertiaResult(
            ixx=i_dict["ixx"],
            iyy=i_dict["iyy"],
            izz=i_dict["izz"],
            volume=vol,
            mass=mass,
            mode=InertiaMode.PRIMITIVE,
        )

    @staticmethod
    def compute_sphere(mass: float, radius: float) -> InertiaResult:
        if not (mass is not None):
            raise ValueError("mass must be provided")
        i_dict = sphere_inertia(mass, radius)
        vol = (4.0 / 3.0) * math.pi * radius**3
        return InertiaResult(
            ixx=i_dict["ixx"],
            iyy=i_dict["iyy"],
            izz=i_dict["izz"],
            volume=vol,
            mass=mass,
            mode=InertiaMode.PRIMITIVE,
        )

    @staticmethod
    def compute_capsule(
        mass: float, radius: float, length: float, axis: str = "z"
    ) -> InertiaResult:
        if not (mass is not None):
            raise ValueError("mass must be provided")
        # Degenerate geometry (zero radius/length) collapses to a point;
        # fall back to the legacy small-positive default so downstream
        # physics engines don't crash. See #4600.
        if radius <= 0.0 or length <= 0.0:
            return InertiaResult.create_default(mass)
        i_dict = capsule_inertia(mass, radius, length, axis)
        v_cyl = math.pi * radius**2 * length
        v_sphere = (4.0 / 3.0) * math.pi * radius**3
        return InertiaResult(
            ixx=i_dict["ixx"],
            iyy=i_dict["iyy"],
            izz=i_dict["izz"],
            volume=v_cyl + v_sphere,
            mass=mass,
            mode=InertiaMode.PRIMITIVE,
        )

    @staticmethod
    def compute_ellipsoid(
        mass: float, semi_a: float, semi_b: float, semi_c: float
    ) -> InertiaResult:
        if not (mass is not None):
            raise ValueError("mass must be provided")
        i_dict = ellipsoid_inertia(mass, semi_a, semi_b, semi_c)
        vol = (4.0 / 3.0) * math.pi * semi_a * semi_b * semi_c
        return InertiaResult(
            ixx=i_dict["ixx"],
            iyy=i_dict["iyy"],
            izz=i_dict["izz"],
            volume=vol,
            mass=mass,
            mode=InertiaMode.PRIMITIVE,
        )

    @classmethod
    def compute(
        cls,
        shape: PrimitiveShape | str,
        mass: float,
        dimensions: dict[str, float] | tuple[float, ...],
        axis: str = "z",
    ) -> InertiaResult:
        if isinstance(shape, str):
            shape = PrimitiveShape(shape.lower())
        if isinstance(dimensions, tuple):
            dimensions = cls._tuple_to_dict(shape, dimensions)

        if shape == PrimitiveShape.BOX:
            return cls.compute_box(
                mass,
                dimensions.get("x", dimensions.get("size_x", 0.1)),
                dimensions.get("y", dimensions.get("size_y", 0.1)),
                dimensions.get("z", dimensions.get("size_z", 0.1)),
            )
        if shape == PrimitiveShape.CYLINDER:
            return cls.compute_cylinder(
                mass,
                dimensions.get("radius", 0.05),
                dimensions.get("length", dimensions.get("height", 0.1)),
                axis,
            )
        if shape == PrimitiveShape.SPHERE:
            return cls.compute_sphere(mass, dimensions.get("radius", 0.05))
        if shape == PrimitiveShape.CAPSULE:
            return cls.compute_capsule(
                mass,
                dimensions.get("radius", 0.05),
                dimensions.get("length", dimensions.get("height", 0.1)),
                axis,
            )
        if shape == PrimitiveShape.ELLIPSOID:
            return cls.compute_ellipsoid(
                mass,
                dimensions.get("a", dimensions.get("semi_a", 0.1)),
                dimensions.get("b", dimensions.get("semi_b", 0.1)),
                dimensions.get("c", dimensions.get("semi_c", 0.1)),
            )
        raise ValueError(f"Unknown shape: {shape}")

    @staticmethod
    def _tuple_to_dict(
        shape: PrimitiveShape, dims: tuple[float, ...]
    ) -> dict[str, float]:
        if not (shape is not None):
            raise ValueError("shape must be provided")
        if shape == PrimitiveShape.BOX:
            if len(dims) >= 3:
                return {"x": dims[0], "y": dims[1], "z": dims[2]}
            if len(dims) == 1:
                return {"x": dims[0], "y": dims[0], "z": dims[0]}
        elif shape in (PrimitiveShape.CYLINDER, PrimitiveShape.CAPSULE):
            if len(dims) >= 2:
                return {"radius": dims[0], "length": dims[1]}
            if len(dims) == 1:
                return {"radius": dims[0], "length": dims[0] * 2}
        elif shape == PrimitiveShape.SPHERE:
            return {"radius": dims[0]}
        elif shape == PrimitiveShape.ELLIPSOID:
            if len(dims) >= 3:
                return {"a": dims[0], "b": dims[1], "c": dims[2]}
            if len(dims) == 1:
                return {"a": dims[0], "b": dims[0], "c": dims[0]}
        return {"radius": dims[0] if dims else 0.1}


def estimate_segment_primitive(
    segment_type: str,
    length: float,
    width: float | None = None,
    depth: float | None = None,
) -> tuple[PrimitiveShape, dict[str, float]]:
    if not (segment_type is not None):
        raise ValueError("segment_type must be provided")
    width, depth = _normalize_dimensions(length, width, depth)
    segment_lower = segment_type.lower()
    if _matches_category(segment_lower, ["head"]):
        return _create_sphere_primitive(length)
    if _matches_category(
        segment_lower, ["arm", "forearm", "thigh", "shin", "shank", "leg"]
    ):
        return _create_limb_capsule_primitive(length, width, depth)
    if _matches_category(segment_lower, ["torso", "thorax", "lumbar", "pelvis"]):
        return _create_torso_box_primitive(length, width, depth)
    if _matches_category(segment_lower, ["hand", "foot"]):
        return _create_extremity_box_primitive(length, width, depth)
    if _matches_category(segment_lower, ["neck"]):
        return _create_neck_cylinder_primitive(length, width, depth)
    return _create_limb_capsule_primitive(length, width, depth)


def _normalize_dimensions(
    length: float, width: float | None, depth: float | None
) -> tuple[float, float]:
    if not (length is not None):
        raise ValueError("length must be provided")
    if width is None:
        width = length * 0.2
    if depth is None:
        depth = length * 0.15
    return width, depth


def _matches_category(segment_lower: str, keywords: list[str]) -> bool:
    return any(keyword in segment_lower for keyword in keywords)


def _create_sphere_primitive(length: float) -> tuple[PrimitiveShape, dict[str, float]]:
    return PrimitiveShape.SPHERE, {"radius": length / 2}


def _create_limb_capsule_primitive(
    length: float, width: float, depth: float
) -> tuple[PrimitiveShape, dict[str, float]]:
    if not (length is not None):
        raise ValueError("length must be provided")
    radius = (width + depth) / 4
    cyl_length = max(0.01, length - 2 * radius)
    return PrimitiveShape.CAPSULE, {"radius": radius, "length": cyl_length}


def _create_torso_box_primitive(
    length: float, width: float, depth: float
) -> tuple[PrimitiveShape, dict[str, float]]:
    return PrimitiveShape.BOX, {"x": width, "y": depth, "z": length}


def _create_extremity_box_primitive(
    length: float, width: float, depth: float
) -> tuple[PrimitiveShape, dict[str, float]]:
    return PrimitiveShape.BOX, {"x": width, "y": length, "z": depth}


def _create_neck_cylinder_primitive(
    length: float, width: float, depth: float
) -> tuple[PrimitiveShape, dict[str, float]]:
    if not (length is not None):
        raise ValueError("length must be provided")
    radius = (width + depth) / 4
    return PrimitiveShape.CYLINDER, {"radius": radius, "length": length}
