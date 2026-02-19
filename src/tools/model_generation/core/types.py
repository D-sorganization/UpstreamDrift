"""
Core data types for model generation.

This module defines the fundamental data structures used throughout
the model_generation package for representing URDF elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class GeometryType(Enum):
    """Supported geometry types for visual and collision elements."""

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    MESH = "mesh"


class JointType(Enum):
    """Supported URDF joint types."""

    FIXED = "fixed"
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FLOATING = "floating"
    PLANAR = "planar"
    # Composite types (expanded to multiple revolute joints)
    UNIVERSAL = "universal"  # 2-DOF
    GIMBAL = "gimbal"  # 3-DOF (spherical approximation)


@dataclass
class Origin:
    """
    Position and orientation in 3D space.

    Represents the origin of a link or joint frame relative to its parent.
    """

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Roll, pitch, yaw in radians

    @classmethod
    def from_position(cls, x: float, y: float, z: float) -> Origin:
        """Create origin with position only."""
        return cls(xyz=(x, y, z))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Origin:
        """Create from dictionary."""
        xyz = data.get("xyz", (0.0, 0.0, 0.0))
        rpy = data.get("rpy", (0.0, 0.0, 0.0))
        if isinstance(xyz, list):
            xyz = tuple(xyz)
        if isinstance(rpy, list):
            rpy = tuple(rpy)
        return cls(xyz=xyz, rpy=rpy)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"xyz": list(self.xyz), "rpy": list(self.rpy)}

    def to_urdf_string(self) -> str:
        """Generate URDF origin element string."""
        xyz_str = " ".join(f"{v:.6g}" for v in self.xyz)
        rpy_str = " ".join(f"{v:.6g}" for v in self.rpy)
        return f'<origin xyz="{xyz_str}" rpy="{rpy_str}"/>'


@dataclass
class Inertia:
    """
    Inertia tensor representation.

    Stores the 6 unique elements of a symmetric 3x3 inertia matrix
    and the associated mass and center of mass.
    """

    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0
    mass: float = 1.0
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert to 3x3 inertia matrix."""
        return np.array(
            [
                [self.ixx, self.ixy, self.ixz],
                [self.ixy, self.iyy, self.iyz],
                [self.ixz, self.iyz, self.izz],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: NDArray[np.float64],
        mass: float = 1.0,
        center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Inertia:
        """Create from 3x3 inertia matrix."""
        return cls(
            ixx=float(matrix[0, 0]),
            iyy=float(matrix[1, 1]),
            izz=float(matrix[2, 2]),
            ixy=float(matrix[0, 1]),
            ixz=float(matrix[0, 2]),
            iyz=float(matrix[1, 2]),
            mass=mass,
            center_of_mass=center_of_mass,
        )

    @classmethod
    def from_box(
        cls, mass: float, size_x: float, size_y: float, size_z: float
    ) -> Inertia:
        """
        Create inertia for a solid box.

        Args:
            mass: Mass in kg
            size_x, size_y, size_z: Full dimensions in meters
        """
        ixx = (mass / 12.0) * (size_y**2 + size_z**2)
        iyy = (mass / 12.0) * (size_x**2 + size_z**2)
        izz = (mass / 12.0) * (size_x**2 + size_y**2)
        return cls(ixx=ixx, iyy=iyy, izz=izz, mass=mass)

    @classmethod
    def from_cylinder(
        cls, mass: float, radius: float, length: float, axis: str = "z"
    ) -> Inertia:
        """
        Create inertia for a solid cylinder.

        Args:
            mass: Mass in kg
            radius: Radius in meters
            length: Length in meters
            axis: Cylinder axis ('x', 'y', or 'z')
        """
        # Inertia about cylinder axis
        i_axial = 0.5 * mass * radius**2
        # Inertia about perpendicular axes
        i_perp = (mass / 12.0) * (3 * radius**2 + length**2)

        if axis == "x":
            return cls(ixx=i_axial, iyy=i_perp, izz=i_perp, mass=mass)
        if axis == "y":
            return cls(ixx=i_perp, iyy=i_axial, izz=i_perp, mass=mass)
        # z
        return cls(ixx=i_perp, iyy=i_perp, izz=i_axial, mass=mass)

    @classmethod
    def from_sphere(cls, mass: float, radius: float) -> Inertia:
        """
        Create inertia for a solid sphere.

        Args:
            mass: Mass in kg
            radius: Radius in meters
        """
        i = (2.0 / 5.0) * mass * radius**2
        return cls(ixx=i, iyy=i, izz=i, mass=mass)

    @classmethod
    def from_capsule(
        cls, mass: float, radius: float, length: float, axis: str = "z"
    ) -> Inertia:
        """
        Create inertia for a solid capsule (cylinder with hemispherical caps).

        Args:
            mass: Mass in kg
            radius: Radius in meters
            length: Length of cylindrical portion in meters
            axis: Capsule axis ('x', 'y', or 'z')
        """
        # Volume fractions
        v_cyl = math.pi * radius**2 * length
        v_sphere = (4.0 / 3.0) * math.pi * radius**3
        v_total = v_cyl + v_sphere

        m_cyl = mass * v_cyl / v_total
        m_sphere = mass * v_sphere / v_total

        # Cylinder inertia
        i_cyl_axial = 0.5 * m_cyl * radius**2
        i_cyl_perp = (m_cyl / 12.0) * (3 * radius**2 + length**2)

        # Sphere inertia (two hemispheres at ends)
        i_sphere = (2.0 / 5.0) * m_sphere * radius**2
        # Parallel axis theorem for hemisphere offset
        hemisphere_offset = length / 2.0 + (3.0 / 8.0) * radius
        i_sphere_perp = i_sphere + 0.5 * m_sphere * hemisphere_offset**2

        i_axial = i_cyl_axial + i_sphere
        i_perp = i_cyl_perp + i_sphere_perp

        if axis == "x":
            return cls(ixx=i_axial, iyy=i_perp, izz=i_perp, mass=mass)
        if axis == "y":
            return cls(ixx=i_perp, iyy=i_axial, izz=i_perp, mass=mass)
        # z
        return cls(ixx=i_perp, iyy=i_perp, izz=i_axial, mass=mass)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Inertia:
        """Create from dictionary."""
        return cls(
            ixx=data.get("ixx", 0.1),
            iyy=data.get("iyy", 0.1),
            izz=data.get("izz", 0.1),
            ixy=data.get("ixy", 0.0),
            ixz=data.get("ixz", 0.0),
            iyz=data.get("iyz", 0.0),
            mass=data.get("mass", 1.0),
            center_of_mass=tuple(data.get("center_of_mass", (0.0, 0.0, 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ixx": self.ixx,
            "iyy": self.iyy,
            "izz": self.izz,
            "ixy": self.ixy,
            "ixz": self.ixz,
            "iyz": self.iyz,
            "mass": self.mass,
            "center_of_mass": list(self.center_of_mass),
        }

    def is_positive_definite(self) -> bool:
        """Check if inertia matrix is positive definite."""
        try:
            np.linalg.cholesky(self.to_matrix())
            return True
        except np.linalg.LinAlgError:
            return False

    def is_diagonal(self) -> bool:
        """Check if inertia is diagonal (no off-diagonal elements)."""
        return abs(self.ixy) < 1e-10 and abs(self.ixz) < 1e-10 and abs(self.iyz) < 1e-10

    def satisfies_triangle_inequality(self) -> bool:
        """
        Check triangle inequality for principal moments.

        For a physical rigid body: |Ia - Ib| <= Ic <= Ia + Ib
        """
        values = [self.ixx, self.iyy, self.izz]
        for i in range(3):
            ia, ib, ic = values[i], values[(i + 1) % 3], values[(i + 2) % 3]
            if not (abs(ia - ib) <= ic <= ia + ib):
                return False
        return True

    def scale_to_mass(self, new_mass: float) -> Inertia:
        """Return new inertia scaled to different mass."""
        if self.mass <= 0:
            raise ValueError("Cannot scale from zero or negative mass")
        scale = new_mass / self.mass
        return Inertia(
            ixx=self.ixx * scale,
            iyy=self.iyy * scale,
            izz=self.izz * scale,
            ixy=self.ixy * scale,
            ixz=self.ixz * scale,
            iyz=self.iyz * scale,
            mass=new_mass,
            center_of_mass=self.center_of_mass,
        )

    def to_urdf_string(self) -> str:
        """Generate URDF inertia element string."""
        return (
            f'<inertia ixx="{self.ixx:.6g}" ixy="{self.ixy:.6g}" '
            f'ixz="{self.ixz:.6g}" iyy="{self.iyy:.6g}" '
            f'iyz="{self.iyz:.6g}" izz="{self.izz:.6g}"/>'
        )


@dataclass
class Material:
    """Material definition for visual appearance."""

    name: str
    color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)  # RGBA
    texture: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Material:
        """Create from dictionary."""
        color = data.get("color", (0.8, 0.8, 0.8, 1.0))
        if isinstance(color, list):
            color = tuple(color)
        return cls(
            name=data.get("name", "default"),
            color=color,
            texture=data.get("texture"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name, "color": list(self.color)}
        if self.texture:
            result["texture"] = self.texture
        return result

    def to_urdf_string(self, inline: bool = False) -> str:
        """Generate URDF material element string."""
        rgba_str = " ".join(f"{v:.4g}" for v in self.color)
        if inline:
            return f'<material name="{self.name}"><color rgba="{rgba_str}"/></material>'
        return f'<material name="{self.name}"/>'

    # Common materials
    @classmethod
    def skin(cls) -> Material:
        """Create a skin-colored material preset."""
        return cls("skin", (0.87, 0.72, 0.53, 1.0))

    @classmethod
    def bone(cls) -> Material:
        """Create a bone-colored material preset."""
        return cls("bone", (0.95, 0.95, 0.90, 1.0))

    @classmethod
    def muscle(cls) -> Material:
        """Create a muscle-colored material preset."""
        return cls("muscle", (0.8, 0.3, 0.3, 1.0))

    @classmethod
    def metal(cls) -> Material:
        """Create a metal-colored material preset."""
        return cls("metal", (0.7, 0.7, 0.75, 1.0))

    @classmethod
    def plastic(cls) -> Material:
        """Create a plastic-colored material preset."""
        return cls("plastic", (0.3, 0.3, 0.8, 1.0))


@dataclass
class Geometry:
    """
    Geometry specification for visual or collision elements.

    Supports primitive shapes and mesh references.
    """

    geometry_type: GeometryType
    # Dimensions depend on type:
    # BOX: (size_x, size_y, size_z)
    # CYLINDER: (radius, length)
    # SPHERE: (radius,)
    # CAPSULE: (radius, length)
    # MESH: () - use mesh_filename
    dimensions: tuple[float, ...] = ()
    mesh_filename: str | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def box(cls, size_x: float, size_y: float, size_z: float) -> Geometry:
        """Create box geometry."""
        return cls(GeometryType.BOX, dimensions=(size_x, size_y, size_z))

    @classmethod
    def cylinder(cls, radius: float, length: float) -> Geometry:
        """Create cylinder geometry."""
        return cls(GeometryType.CYLINDER, dimensions=(radius, length))

    @classmethod
    def sphere(cls, radius: float) -> Geometry:
        """Create sphere geometry."""
        return cls(GeometryType.SPHERE, dimensions=(radius,))

    @classmethod
    def capsule(cls, radius: float, length: float) -> Geometry:
        """Create capsule geometry."""
        return cls(GeometryType.CAPSULE, dimensions=(radius, length))

    @classmethod
    def mesh(
        cls,
        filename: str,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Geometry:
        """Create mesh geometry."""
        return cls(GeometryType.MESH, mesh_filename=filename, mesh_scale=scale)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Geometry:
        """Create from dictionary."""
        geom_type = GeometryType(data.get("type", "box"))
        dims = data.get("dimensions", ())
        if isinstance(dims, list):
            dims = tuple(dims)
        return cls(
            geometry_type=geom_type,
            dimensions=dims,
            mesh_filename=data.get("mesh_filename"),
            mesh_scale=tuple(data.get("mesh_scale", (1.0, 1.0, 1.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.geometry_type.value,
            "dimensions": list(self.dimensions),
        }
        if self.mesh_filename:
            result["mesh_filename"] = self.mesh_filename
            result["mesh_scale"] = list(self.mesh_scale)
        return result

    def to_urdf_string(self) -> str:
        """Generate URDF geometry element string."""
        if self.geometry_type == GeometryType.BOX:
            size_str = " ".join(f"{d:.6g}" for d in self.dimensions)
            return f'<geometry><box size="{size_str}"/></geometry>'
        if self.geometry_type == GeometryType.CYLINDER:
            return (
                f'<geometry><cylinder radius="{self.dimensions[0]:.6g}" '
                f'length="{self.dimensions[1]:.6g}"/></geometry>'
            )
        if self.geometry_type == GeometryType.SPHERE:
            return f'<geometry><sphere radius="{self.dimensions[0]:.6g}"/></geometry>'
        if self.geometry_type == GeometryType.CAPSULE:
            # URDF doesn't have capsule, approximate with cylinder
            return (
                f'<geometry><cylinder radius="{self.dimensions[0]:.6g}" '
                f'length="{self.dimensions[1]:.6g}"/></geometry>'
            )
        if self.geometry_type == GeometryType.MESH:
            scale_str = " ".join(f"{s:.6g}" for s in self.mesh_scale)
            return (
                f'<geometry><mesh filename="{self.mesh_filename}" '
                f'scale="{scale_str}"/></geometry>'
            )
        raise ValueError(f"Unknown geometry type: {self.geometry_type}")


@dataclass
class JointLimits:
    """Joint limits specification."""

    lower: float = -math.pi
    upper: float = math.pi
    effort: float = 1000.0
    velocity: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JointLimits:
        """Create from dictionary."""
        return cls(
            lower=data.get("lower", -math.pi),
            upper=data.get("upper", math.pi),
            effort=data.get("effort", 1000.0),
            velocity=data.get("velocity", 10.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "effort": self.effort,
            "velocity": self.velocity,
        }

    def to_urdf_string(self) -> str:
        """Generate URDF limit element string."""
        return (
            f'<limit lower="{self.lower:.6g}" upper="{self.upper:.6g}" '
            f'effort="{self.effort:.6g}" velocity="{self.velocity:.6g}"/>'
        )


@dataclass
class JointDynamics:
    """Joint dynamics parameters."""

    damping: float = 0.5
    friction: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JointDynamics:
        """Create from dictionary."""
        return cls(
            damping=data.get("damping", 0.5),
            friction=data.get("friction", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"damping": self.damping, "friction": self.friction}

    def to_urdf_string(self) -> str:
        """Generate URDF dynamics element string."""
        return (
            f'<dynamics damping="{self.damping:.6g}" friction="{self.friction:.6g}"/>'
        )


@dataclass
class Link:
    """
    URDF link representation.

    A link represents a rigid body with inertial properties,
    visual geometry, and collision geometry.
    """

    name: str
    inertia: Inertia = field(default_factory=lambda: Inertia(0.1, 0.1, 0.1))
    visual_geometry: Geometry | None = None
    visual_origin: Origin = field(default_factory=Origin)
    visual_material: Material | None = None
    collision_geometry: Geometry | None = None
    collision_origin: Origin = field(default_factory=Origin)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Link:
        """Create from dictionary."""
        inertia_data = data.get("inertia", {})
        if "mass" not in inertia_data:
            inertia_data["mass"] = data.get("mass", 1.0)

        visual_geom = None
        if "visual_geometry" in data:
            visual_geom = Geometry.from_dict(data["visual_geometry"])
        elif "visual" in data:
            visual_geom = Geometry.from_dict(data["visual"])

        collision_geom = None
        if "collision_geometry" in data:
            collision_geom = Geometry.from_dict(data["collision_geometry"])
        elif "collision" in data:
            collision_geom = Geometry.from_dict(data["collision"])

        return cls(
            name=data["name"],
            inertia=Inertia.from_dict(inertia_data),
            visual_geometry=visual_geom,
            visual_origin=Origin.from_dict(data.get("visual_origin", {})),
            visual_material=(
                Material.from_dict(data["material"]) if "material" in data else None
            ),
            collision_geometry=collision_geom,
            collision_origin=Origin.from_dict(data.get("collision_origin", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "inertia": self.inertia.to_dict(),
            "visual_origin": self.visual_origin.to_dict(),
            "collision_origin": self.collision_origin.to_dict(),
        }
        if self.visual_geometry:
            result["visual_geometry"] = self.visual_geometry.to_dict()
        if self.visual_material:
            result["material"] = self.visual_material.to_dict()
        if self.collision_geometry:
            result["collision_geometry"] = self.collision_geometry.to_dict()
        return result


@dataclass
class Joint:
    """
    URDF joint representation.

    A joint connects two links and defines their relative motion.
    """

    name: str
    joint_type: JointType
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limits: JointLimits | None = None
    dynamics: JointDynamics = field(default_factory=JointDynamics)
    # For composite joints (gimbal, universal)
    composite_axes: list[tuple[float, float, float]] | None = None
    composite_limits: list[JointLimits] | None = None

    def __post_init__(self) -> None:
        """Ensure limits are set for limited joint types."""
        if self.joint_type == JointType.REVOLUTE and self.limits is None:
            self.limits = JointLimits()
        if self.joint_type == JointType.PRISMATIC and self.limits is None:
            self.limits = JointLimits(lower=-1.0, upper=1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Joint:
        """Create from dictionary."""
        joint_type = JointType(data.get("type", "revolute"))
        axis = data.get("axis", (0.0, 0.0, 1.0))
        if isinstance(axis, list):
            axis = tuple(axis)

        limits = None
        if "limits" in data:
            limits = JointLimits.from_dict(data["limits"])

        dynamics = JointDynamics()
        if "dynamics" in data:
            dynamics = JointDynamics.from_dict(data["dynamics"])

        return cls(
            name=data["name"],
            joint_type=joint_type,
            parent=data["parent"],
            child=data["child"],
            origin=Origin.from_dict(data.get("origin", {})),
            axis=axis,
            limits=limits,
            dynamics=dynamics,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.joint_type.value,
            "parent": self.parent,
            "child": self.child,
            "origin": self.origin.to_dict(),
            "axis": list(self.axis),
            "dynamics": self.dynamics.to_dict(),
        }
        if self.limits:
            result["limits"] = self.limits.to_dict()
        return result

    def is_composite(self) -> bool:
        """Check if this is a composite joint (gimbal or universal)."""
        return self.joint_type in (JointType.GIMBAL, JointType.UNIVERSAL)

    def get_dof_count(self) -> int:
        """Return degrees of freedom for this joint type."""
        dof_map = {
            JointType.FIXED: 0,
            JointType.REVOLUTE: 1,
            JointType.CONTINUOUS: 1,
            JointType.PRISMATIC: 1,
            JointType.FLOATING: 6,
            JointType.PLANAR: 3,
            JointType.UNIVERSAL: 2,
            JointType.GIMBAL: 3,
        }
        return dof_map.get(self.joint_type, 1)
