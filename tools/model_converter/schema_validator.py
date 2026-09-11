"""Schema validator for the canonical golfer biomechanical specification.

Validates that `golfer_canonical.yaml` conforms to structural, topological,
and physical requirements (positive mass, positive definite inertia tensors,
triangle inequality, and connected tree kinematics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a canonical model fails schema or physical validation."""


@dataclass(frozen=True)
class Inertia:
    """Inertia tensor parameters in local frame."""

    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    def matrix(self) -> np.ndarray:
        """Return the 3x3 symmetric inertia matrix."""
        return np.array(
            [
                [self.ixx, -self.ixy, -self.ixz],
                [-self.ixy, self.iyy, -self.iyz],
                [-self.ixz, -self.iyz, self.izz],
            ],
            dtype=np.float64,
        )

    def satisfies_triangle_inequality(self, tol: float = 1e-6) -> bool:
        """Verify principal triangle inequalities: I_a + I_b >= I_c."""
        return (
            (self.ixx + self.iyy >= self.izz - tol)
            and (self.ixx + self.izz >= self.iyy - tol)
            and (self.iyy + self.izz >= self.ixx - tol)
        )

    def is_positive_definite(self) -> bool:
        """Verify that eigenvalues of the inertia matrix are strictly positive."""
        eigvals = np.linalg.eigvalsh(self.matrix())
        return bool(np.all(eigvals > 0))


@dataclass(frozen=True)
class Geometry:
    """Visual/collision geometry specification."""

    geom_type: str
    size: tuple[float, ...]
    visual_rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


@dataclass(frozen=True)
class JointDof:
    """Single degree of freedom specification."""

    axis: tuple[float, float, float]
    limits: tuple[float, float]


@dataclass(frozen=True)
class Joint:
    """Joint articulation specification."""

    joint_type: str
    dofs: tuple[JointDof, ...] = ()
    damping: float = 0.0


@dataclass(frozen=True)
class Transform:
    """Spatial transform offset (translation xyz and rotation rpy)."""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RootBody:
    """Root floating body of the kinematic tree."""

    name: str
    frame: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    mass: float
    inertia: Inertia
    geometry: Geometry


@dataclass(frozen=True)
class Segment:
    """Articulated segment link."""

    name: str
    parent: str
    joint: Joint
    origin: Transform
    mass: float
    inertia: Inertia
    geometry: Geometry


@dataclass(frozen=True)
class CanonicalModel:
    """Validated full canonical biomechanical model specification."""

    description: str
    units: dict[str, str]
    coordinate_system: dict[str, str]
    root: RootBody
    segments: tuple[Segment, ...]
    constraints: tuple[dict[str, Any], ...] = ()
    contacts: tuple[dict[str, Any], ...] = ()
    named_frames: tuple[dict[str, Any], ...] = ()
    scaling: dict[str, Any] | None = None

    @property
    def total_mass(self) -> float:
        """Calculate total model mass in kg."""
        return self.root.mass + sum(s.mass for s in self.segments)

    def get_segment(self, name: str) -> Segment | None:
        """Look up a segment by name."""
        for seg in self.segments:
            if seg.name == name:
                return seg
        return None


def _parse_inertia(data: dict[str, Any], segment_name: str) -> Inertia:
    """Parse and validate an inertia dictionary."""
    for key in ("ixx", "iyy", "izz"):
        if key not in data:
            raise ValidationError(
                f"Segment '{segment_name}' missing inertia property '{key}'"
            )
        val = float(data[key])
        if val <= 0:
            raise ValidationError(
                f"Segment '{segment_name}' inertia '{key}' must be positive, got {val}"
            )

    inertia = Inertia(
        ixx=float(data["ixx"]),
        iyy=float(data["iyy"]),
        izz=float(data["izz"]),
        ixy=float(data.get("ixy", 0.0)),
        ixz=float(data.get("ixz", 0.0)),
        iyz=float(data.get("iyz", 0.0)),
    )

    if not inertia.satisfies_triangle_inequality():
        raise ValidationError(
            f"Segment '{segment_name}' inertia violates triangle inequality: "
            f"ixx={inertia.ixx}, iyy={inertia.iyy}, izz={inertia.izz}"
        )

    if not inertia.is_positive_definite():
        raise ValidationError(
            f"Segment '{segment_name}' inertia tensor is not positive-definite"
        )

    return inertia


def _parse_geometry(data: dict[str, Any], segment_name: str) -> Geometry:
    """Parse a geometry dictionary."""
    geom_type = str(data.get("type", "capsule")).lower()
    raw_size = data.get("size", [0.05, 0.1])
    size = tuple(float(x) for x in raw_size)
    raw_rgba = data.get("visual_rgba", [0.7, 0.7, 0.7, 1.0])
    rgba = tuple(float(x) for x in raw_rgba)
    if len(rgba) != 4:
        rgba = (0.7, 0.7, 0.7, 1.0)
    return Geometry(
        geom_type=geom_type, size=size, visual_rgba=(rgba[0], rgba[1], rgba[2], rgba[3])
    )


def _parse_vec3(
    val: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[float, float, float]:
    """Parse a 3-element numeric vector."""
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        return default
    return (float(val[0]), float(val[1]), float(val[2]))


def _parse_joint(data: dict[str, Any], segment_name: str) -> Joint:
    """Parse and validate joint specification."""
    joint_type = str(data.get("type", "revolute")).lower()
    valid_types = {"revolute", "universal", "gimbal", "fixed", "prismatic", "spherical"}
    if joint_type not in valid_types:
        raise ValidationError(
            f"Segment '{segment_name}' invalid joint type '{joint_type}'. "
            f"Expected one of {sorted(valid_types)}"
        )

    damping = float(data.get("damping", 0.0))
    if damping < 0:
        raise ValidationError(
            f"Segment '{segment_name}' joint damping must be non-negative, got {damping}"
        )

    dofs: list[JointDof] = []
    if joint_type == "revolute":
        if "axis" not in data:
            raise ValidationError(f"Revolute joint in '{segment_name}' missing 'axis'")
        axis = _parse_vec3(data["axis"])
        if all(a == 0 for a in axis):
            raise ValidationError(
                f"Revolute joint in '{segment_name}' has invalid 3D axis {axis}"
            )
        limits_raw = data.get("limits", [-np.pi, np.pi])
        limits = (float(limits_raw[0]), float(limits_raw[1]))
        if limits[0] > limits[1]:
            raise ValidationError(
                f"Revolute joint in '{segment_name}' has lower limit {limits[0]} > upper {limits[1]}"
            )
        dofs.append(JointDof(axis=axis, limits=limits))

    elif joint_type in ("universal", "gimbal"):
        raw_dofs = data.get("dofs", [])
        expected_dofs = 2 if joint_type == "universal" else 3
        if len(raw_dofs) != expected_dofs:
            raise ValidationError(
                f"{joint_type.capitalize()} joint in '{segment_name}' must declare "
                f"{expected_dofs} DOFs, got {len(raw_dofs)}"
            )
        for i, raw_dof in enumerate(raw_dofs):
            if "axis" not in raw_dof:
                raise ValidationError(
                    f"DOF {i} of '{segment_name}' joint missing 'axis'"
                )
            axis = _parse_vec3(raw_dof["axis"])
            if all(a == 0 for a in axis):
                raise ValidationError(
                    f"DOF {i} of '{segment_name}' joint has invalid axis {axis}"
                )
            limits_raw = raw_dof.get("limits", [-np.pi, np.pi])
            limits = (float(limits_raw[0]), float(limits_raw[1]))
            if limits[0] > limits[1]:
                raise ValidationError(
                    f"DOF {i} of '{segment_name}' joint has lower limit {limits[0]} > upper {limits[1]}"
                )
            dofs.append(JointDof(axis=axis, limits=limits))

    return Joint(joint_type=joint_type, dofs=tuple(dofs), damping=damping)


def validate_canonical_model(source: dict[str, Any] | Path | str) -> CanonicalModel:
    """Validate a canonical specification dictionary or YAML file path.

    Raises:
        ValidationError: If schema or physical validity checks fail.
        FileNotFoundError: If path does not exist.
    """
    if isinstance(source, (Path, str)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical specification file not found at: {path}"
            )
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif isinstance(source, dict):
        data = source
    else:
        raise ValidationError(f"Expected dict, Path, or str; got {type(source)}")

    if not isinstance(data, dict):
        raise ValidationError("Top-level YAML structure must be a dictionary")

    # Required top-level keys
    for req in ("root", "segments"):
        if req not in data:
            raise ValidationError(f"Missing required top-level section: '{req}'")

    # Units check
    units = data.get("units", {})
    if not isinstance(units, dict):
        raise ValidationError("'units' section must be a dictionary")

    # Coordinate system check
    coord_sys = data.get("coordinate_system", {})
    if not isinstance(coord_sys, dict):
        raise ValidationError("'coordinate_system' section must be a dictionary")

    # Root validation
    root_data = data["root"]
    if not isinstance(root_data, dict):
        raise ValidationError("'root' section must be a dictionary")
    root_name = root_data.get("name")
    if not root_name or not isinstance(root_name, str):
        raise ValidationError("'root' must have a non-empty string 'name'")
    root_mass = float(root_data.get("mass", 0.0))
    if root_mass <= 0:
        raise ValidationError(
            f"Root '{root_name}' mass must be positive, got {root_mass}"
        )
    root_pos = _parse_vec3(root_data.get("position", [0.0, 0.0, 0.0]))
    root_orient = _parse_vec3(root_data.get("orientation", [0.0, 0.0, 0.0]))
    root_inertia = _parse_inertia(root_data.get("inertia", {}), root_name)
    root_geom = _parse_geometry(root_data.get("geometry", {}), root_name)

    root = RootBody(
        name=root_name,
        frame=str(root_data.get("frame", f"{root_name}_frame")),
        position=root_pos,
        orientation=root_orient,
        mass=root_mass,
        inertia=root_inertia,
        geometry=root_geom,
    )

    # Segments validation
    raw_segments = data["segments"]
    if not isinstance(raw_segments, list) or len(raw_segments) == 0:
        raise ValidationError(
            "'segments' must be a non-empty list of segment definitions"
        )

    parsed_segments: list[Segment] = []
    seen_names: set[str] = {root_name}

    for seg_dict in raw_segments:
        if not isinstance(seg_dict, dict):
            raise ValidationError(
                f"Segment entry must be a dictionary, got {type(seg_dict)}"
            )
        name = seg_dict.get("name")
        if not name or not isinstance(name, str):
            raise ValidationError("Segment missing a non-empty string 'name'")
        if name in seen_names:
            raise ValidationError(f"Duplicate segment name: '{name}'")
        seen_names.add(name)

        parent = seg_dict.get("parent")
        if not parent or not isinstance(parent, str):
            raise ValidationError(f"Segment '{name}' missing a non-empty parent name")

        mass = float(seg_dict.get("mass", 0.0))
        if mass <= 0:
            raise ValidationError(f"Segment '{name}' mass must be positive, got {mass}")

        origin_dict = seg_dict.get("origin", {})
        xyz = _parse_vec3(origin_dict.get("xyz", [0.0, 0.0, 0.0]))
        rpy = _parse_vec3(origin_dict.get("rpy", [0.0, 0.0, 0.0]))
        origin = Transform(xyz=xyz, rpy=rpy)

        joint = _parse_joint(seg_dict.get("joint", {}), name)
        inertia = _parse_inertia(seg_dict.get("inertia", {}), name)
        geometry = _parse_geometry(seg_dict.get("geometry", {}), name)

        parsed_segments.append(
            Segment(
                name=name,
                parent=parent,
                joint=joint,
                origin=origin,
                mass=mass,
                inertia=inertia,
                geometry=geometry,
            )
        )

    # Topological connectivity validation (all parents must exist, no cycles)
    all_names = {s.name for s in parsed_segments} | {root.name}
    for seg in parsed_segments:
        if seg.parent not in all_names:
            raise ValidationError(
                f"Segment '{seg.name}' has nonexistent parent '{seg.parent}'"
            )

    # Cycle detection using depth-first search from root
    adj: dict[str, list[str]] = {name: [] for name in all_names}
    for seg in parsed_segments:
        adj[seg.parent].append(seg.name)

    visited: set[str] = set()
    visiting: set[str] = set()

    def dfs(node: str) -> None:
        if node in visiting:
            raise ValidationError(
                f"Cycle detected in kinematic tree involving node '{node}'"
            )
        if node in visited:
            return
        visiting.add(node)
        for child in adj[node]:
            dfs(child)
        visiting.remove(node)
        visited.add(node)

    dfs(root.name)

    unvisited = all_names - visited
    if unvisited:
        raise ValidationError(
            f"Disconnected segments not reachable from root '{root.name}': {sorted(unvisited)}"
        )

    logger.info(
        "Successfully validated canonical model '%s' with %d segments (total mass %.2f kg)",
        root.name,
        len(parsed_segments),
        root.mass + sum(s.mass for s in parsed_segments),
    )

    return CanonicalModel(
        description=str(data.get("description", "Canonical golfer model")),
        units=units,
        coordinate_system=coord_sys,
        root=root,
        segments=tuple(parsed_segments),
        constraints=tuple(data.get("constraints", [])),
        contacts=tuple(data.get("contacts", [])),
        named_frames=tuple(data.get("named_frames", [])),
        scaling=data.get("scaling"),
    )
