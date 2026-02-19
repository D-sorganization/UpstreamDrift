"""
Manual URDF builder for segment-by-segment construction.

This builder allows adding links and joints one at a time,
with validation at each step.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from model_generation.builders.base_builder import BaseURDFBuilder, BuildResult
from model_generation.core.constants import DEFAULT_INERTIA_KG_M2
from model_generation.core.types import (
    Geometry,
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)
from model_generation.core.validation import Validator

from src.shared.python.core.contracts import postcondition, precondition

logger = logging.getLogger(__name__)


class Handedness(Enum):
    """Handedness for model mirroring."""

    LEFT = "left"
    RIGHT = "right"


class ManualBuilder(BaseURDFBuilder):
    """
    Manual URDF builder with segment-by-segment construction.

    Provides a fluent API for building URDF models step by step,
    with validation at each addition.

    Example:
        builder = ManualBuilder("my_robot")
        builder.add_link(Link(name="base", inertia=Inertia.from_box(10, 1, 1, 0.5)))
        builder.add_link(Link(name="arm", inertia=Inertia.from_cylinder(2, 0.05, 0.5)))
        builder.add_joint(Joint(
            name="base_to_arm",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="arm",
            axis=(1, 0, 0),
        ))
        result = builder.build()
    """

    # Inherited from BaseURDFBuilder but redeclared for mypy
    _links: list[Link]
    _joints: list[Joint]

    def __init__(
        self,
        robot_name: str = "robot",
        handedness: Handedness = Handedness.RIGHT,
        validate_on_add: bool = True,
    ) -> None:
        """
        Initialize manual builder.

        Args:
            robot_name: Name for the robot element
            handedness: Default handedness for the model
            validate_on_add: If True, validate each addition immediately
        """
        super().__init__(robot_name)
        self._handedness = handedness
        self._validate_on_add = validate_on_add

    @property
    def handedness(self) -> Handedness:
        """Get current handedness."""
        return self._handedness

    @handedness.setter
    def handedness(self, value: Handedness) -> None:
        """Set handedness."""
        self._handedness = value

    @precondition(
        lambda self, link: link is not None
        and hasattr(link, "name")
        and len(link.name) > 0,
        "Link must be a valid Link object with a non-empty name",
    )
    def add_link(self, link: Link) -> ManualBuilder:  # type: ignore[override]
        """
        Add a link to the model.

        Args:
            link: Link to add

        Returns:
            Self for method chaining

        Raises:
            ValueError: If validation fails and validate_on_add is True
        """
        if self._validate_on_add:
            result = Validator.validate_link(link)
            if not result.is_valid:
                errors = "; ".join(result.get_error_messages())
                raise ValueError(f"Link validation failed: {errors}")
            for warning in result.warnings:
                logger.warning(str(warning))

        # Check for duplicate name
        if link.name in self.get_link_names():
            raise ValueError(f"Link '{link.name}' already exists")

        self._links.append(link)
        return self

    @precondition(
        lambda self, joint: joint is not None
        and hasattr(joint, "name")
        and len(joint.name) > 0,
        "Joint must be a valid Joint object with a non-empty name",
    )
    def add_joint(self, joint: Joint) -> ManualBuilder:  # type: ignore[override]
        """
        Add a joint to the model.

        Args:
            joint: Joint to add

        Returns:
            Self for method chaining

        Raises:
            ValueError: If validation fails and validate_on_add is True
        """
        link_names = set(self.get_link_names())

        if self._validate_on_add:
            result = Validator.validate_joint(joint, link_names)
            if not result.is_valid:
                errors = "; ".join(result.get_error_messages())
                raise ValueError(f"Joint validation failed: {errors}")
            for warning in result.warnings:
                logger.warning(str(warning))

        # Check for duplicate name
        if joint.name in self.get_joint_names():
            raise ValueError(f"Joint '{joint.name}' already exists")

        self._joints.append(joint)
        return self

    def add_segment(self, segment_data: dict[str, Any]) -> ManualBuilder:
        """
        Add a segment (link + joint) from dictionary data.

        This method provides compatibility with the legacy URDFBuilder API.

        Args:
            segment_data: Dictionary with link and joint properties

        Returns:
            Self for method chaining
        """
        link = self._link_from_dict(segment_data)
        self.add_link(link)

        # Add joint if parent is specified
        if segment_data.get("parent"):
            joint = self._joint_from_dict(segment_data)
            self.add_joint(joint)

        return self

    @precondition(
        lambda self, name, cascade=True: name is not None and len(name.strip()) > 0,
        "Link name must be a non-empty string",
    )
    def remove_link(self, name: str, cascade: bool = True) -> ManualBuilder:
        """
        Remove a link from the model.

        Args:
            name: Name of link to remove
            cascade: If True, also remove child links and joints

        Returns:
            Self for method chaining
        """
        # Find the link
        link_to_remove = None
        for link in self._links:
            if link.name == name:
                link_to_remove = link
                break

        if link_to_remove is None:
            raise ValueError(f"Link '{name}' not found")

        if cascade:
            # Find all descendants
            descendants = self._get_descendants(name)
            names_to_remove = {name} | descendants

            # Remove links
            self._links = [
                link for link in self._links if link.name not in names_to_remove
            ]

            # Remove joints
            self._joints = [
                j
                for j in self._joints
                if j.parent not in names_to_remove and j.child not in names_to_remove
            ]
        else:
            # Just remove this link and joints connecting to it
            self._links = [link for link in self._links if link.name != name]
            self._joints = [
                j for j in self._joints if j.parent != name and j.child != name
            ]

        return self

    def remove_joint(self, name: str) -> ManualBuilder:
        """
        Remove a joint from the model.

        Args:
            name: Name of joint to remove

        Returns:
            Self for method chaining
        """
        self._joints = [j for j in self._joints if j.name != name]
        return self

    def modify_link(self, name: str, **kwargs: Any) -> ManualBuilder:
        """
        Modify an existing link.

        Args:
            name: Name of link to modify
            **kwargs: Properties to update

        Returns:
            Self for method chaining
        """
        for i, link in enumerate(self._links):
            if link.name == name:
                # Create new link with updated properties
                link_dict = link.to_dict()
                link_dict.update(kwargs)
                self._links[i] = Link.from_dict(link_dict)
                return self

        raise ValueError(f"Link '{name}' not found")

    def modify_joint(self, name: str, **kwargs: Any) -> ManualBuilder:
        """
        Modify an existing joint.

        Args:
            name: Name of joint to modify
            **kwargs: Properties to update

        Returns:
            Self for method chaining
        """
        for i, joint in enumerate(self._joints):
            if joint.name == name:
                joint_dict = joint.to_dict()
                joint_dict.update(kwargs)
                self._joints[i] = Joint.from_dict(joint_dict)
                return self

        raise ValueError(f"Joint '{name}' not found")

    @precondition(
        lambda self, axis="y": axis.lower() in ("x", "y", "z"),
        "Mirror axis must be 'x', 'y', or 'z'",
    )
    def mirror(self, axis: str = "y") -> ManualBuilder:
        """
        Mirror the model about an axis.

        Args:
            axis: Axis to mirror about ('x', 'y', or 'z')

        Returns:
            Self for method chaining
        """
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

        # Mirror link origins and geometry
        for link in self._links:
            # Mirror visual origin
            new_xyz = list(link.visual_origin.xyz)
            new_xyz[axis_idx] = -new_xyz[axis_idx]
            link.visual_origin = Origin(
                xyz=(new_xyz[0], new_xyz[1], new_xyz[2]), rpy=link.visual_origin.rpy
            )

            # Mirror collision origin
            new_xyz = list(link.collision_origin.xyz)
            new_xyz[axis_idx] = -new_xyz[axis_idx]
            link.collision_origin = Origin(
                xyz=(new_xyz[0], new_xyz[1], new_xyz[2]), rpy=link.collision_origin.rpy
            )

            # Mirror COM
            new_com = list(link.inertia.center_of_mass)
            new_com[axis_idx] = -new_com[axis_idx]
            link.inertia = Inertia(
                ixx=link.inertia.ixx,
                iyy=link.inertia.iyy,
                izz=link.inertia.izz,
                ixy=-link.inertia.ixy if axis_idx != 2 else link.inertia.ixy,
                ixz=-link.inertia.ixz if axis_idx != 1 else link.inertia.ixz,
                iyz=-link.inertia.iyz if axis_idx != 0 else link.inertia.iyz,
                mass=link.inertia.mass,
                center_of_mass=(new_com[0], new_com[1], new_com[2]),
            )

        # Mirror joint origins and axes
        for joint in self._joints:
            new_xyz = list(joint.origin.xyz)
            new_xyz[axis_idx] = -new_xyz[axis_idx]
            joint.origin = Origin(
                xyz=(new_xyz[0], new_xyz[1], new_xyz[2]), rpy=joint.origin.rpy
            )

            # Mirror axis
            new_axis = list(joint.axis)
            new_axis[axis_idx] = -new_axis[axis_idx]
            joint.axis = (new_axis[0], new_axis[1], new_axis[2])

        # Toggle handedness
        self._handedness = (
            Handedness.LEFT
            if self._handedness == Handedness.RIGHT
            else Handedness.RIGHT
        )

        return self

    def get_mirrored(self, axis: str = "y") -> ManualBuilder:
        """
        Get a mirrored copy of the model.

        Args:
            axis: Axis to mirror about

        Returns:
            New ManualBuilder with mirrored model
        """
        # Create copy
        new_builder = ManualBuilder(
            robot_name=self._robot_name,
            handedness=self._handedness,
            validate_on_add=False,
        )

        # Copy links and joints
        for link in self._links:
            new_builder._links.append(Link.from_dict(link.to_dict()))
        for joint in self._joints:
            new_builder._joints.append(Joint.from_dict(joint.to_dict()))
        new_builder._materials = self._materials.copy()

        # Mirror the copy
        new_builder.mirror(axis)

        return new_builder

    def clear(self) -> None:
        """Clear all links and joints."""
        self._links.clear()
        self._joints.clear()
        self._materials.clear()

    @postcondition(
        lambda result: result is not None and isinstance(result.success, bool),
        "Build must return a valid BuildResult with success status",
    )
    def build(self, **kwargs: Any) -> BuildResult:
        """
        Build the URDF model.

        Returns:
            BuildResult with generated URDF
        """
        # Validate complete model
        validation = self.validate()

        if not validation.is_valid:
            return BuildResult(
                success=False,
                validation=validation,
                error_message="; ".join(validation.get_error_messages()),
                links=self._links.copy(),
                joints=self._joints.copy(),
            )

        # Generate URDF
        urdf_xml = self.to_urdf(pretty_print=kwargs.get("pretty_print", True))

        return BuildResult(
            success=True,
            urdf_xml=urdf_xml,
            links=self._links.copy(),
            joints=self._joints.copy(),
            validation=validation,
            metadata={
                "handedness": self._handedness.value,
                "robot_name": self._robot_name,
            },
        )

    def _get_descendants(self, link_name: str) -> set[str]:
        """Get all descendant link names."""
        descendants: set[str] = set()
        queue = [link_name]

        while queue:
            current = queue.pop(0)
            for joint in self._joints:
                if joint.parent == current and joint.child not in descendants:
                    descendants.add(joint.child)
                    queue.append(joint.child)

        return descendants

    def _link_from_dict(self, data: dict[str, Any]) -> Link:
        """Create Link from dictionary data."""
        name = data["name"]

        # Handle inertia
        physics = data.get("physics", {})
        inertia_data = physics.get("inertia", {})
        mass = physics.get("mass", 1.0)

        inertia = Inertia(
            ixx=inertia_data.get("ixx", DEFAULT_INERTIA_KG_M2),
            iyy=inertia_data.get("iyy", DEFAULT_INERTIA_KG_M2),
            izz=inertia_data.get("izz", DEFAULT_INERTIA_KG_M2),
            ixy=inertia_data.get("ixy", 0.0),
            ixz=inertia_data.get("ixz", 0.0),
            iyz=inertia_data.get("iyz", 0.0),
            mass=mass,
        )

        # Handle geometry
        geom_data = data.get("geometry", {})
        visual_geom = self._geometry_from_dict(geom_data) if geom_data else None

        # Handle origin
        origin_data = geom_data.get("position", {}) if geom_data else {}
        orientation_data = geom_data.get("orientation", {}) if geom_data else {}
        import math

        origin = Origin(
            xyz=(
                origin_data.get("x", 0.0),
                origin_data.get("y", 0.0),
                origin_data.get("z", 0.0),
            ),
            rpy=(
                math.radians(orientation_data.get("roll", 0.0)),
                math.radians(orientation_data.get("pitch", 0.0)),
                math.radians(orientation_data.get("yaw", 0.0)),
            ),
        )

        # Handle material
        mat_data = physics.get("material", {})
        material = None
        if mat_data:
            color_data = mat_data.get("color", {})
            material = Material(
                name=mat_data.get("name", f"{name}_material"),
                color=(
                    color_data.get("r", 0.8),
                    color_data.get("g", 0.8),
                    color_data.get("b", 0.8),
                    color_data.get("a", 1.0),
                ),
            )

        return Link(
            name=name,
            inertia=inertia,
            visual_geometry=visual_geom,
            visual_origin=origin,
            visual_material=material,
            collision_geometry=visual_geom,  # Use same for collision
            collision_origin=origin,
        )

    def _joint_from_dict(self, data: dict[str, Any]) -> Joint:
        """Create Joint from dictionary data."""
        name = f"{data['parent']}_to_{data['name']}"
        joint_data = data.get("joint", {})
        geom_data = data.get("geometry", {})

        # Get origin from geometry
        pos = geom_data.get("position", {})
        orient = geom_data.get("orientation", {})
        import math

        origin = Origin(
            xyz=(pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)),
            rpy=(
                math.radians(orient.get("roll", 0.0)),
                math.radians(orient.get("pitch", 0.0)),
                math.radians(orient.get("yaw", 0.0)),
            ),
        )

        # Get joint type
        joint_type_str = joint_data.get("type", "fixed").lower()
        joint_type = JointType(joint_type_str)

        # Get axis
        axis_data = joint_data.get("axis", {"x": 0, "y": 0, "z": 1})
        axis = (axis_data.get("x", 0), axis_data.get("y", 0), axis_data.get("z", 1))

        # Get limits
        limits_data = joint_data.get("limits", {})
        limits = None
        if limits_data:
            limits = JointLimits(
                lower=math.radians(limits_data.get("lower", -180)),
                upper=math.radians(limits_data.get("upper", 180)),
                effort=limits_data.get("effort", 1000),
                velocity=limits_data.get("velocity", 10),
            )

        return Joint(
            name=name,
            joint_type=joint_type,
            parent=data["parent"],
            child=data["name"],
            origin=origin,
            axis=axis,
            limits=limits,
        )

    def _geometry_from_dict(self, data: dict[str, Any]) -> Geometry:
        """Create Geometry from dictionary data."""
        shape = data.get("shape", "box").lower()
        dims = data.get("dimensions", {})

        from model_generation.core.types import GeometryType

        if shape == "box":
            return Geometry(
                geometry_type=GeometryType.BOX,
                dimensions=(
                    dims.get("width", 0.1),
                    dims.get("height", 0.1),
                    dims.get("length", 0.1),
                ),
            )
        elif shape == "cylinder":
            return Geometry(
                geometry_type=GeometryType.CYLINDER,
                dimensions=(dims.get("radius", 0.05), dims.get("length", 0.1)),
            )
        elif shape == "sphere":
            return Geometry(
                geometry_type=GeometryType.SPHERE,
                dimensions=(dims.get("radius", 0.05),),
            )
        elif shape == "capsule":
            return Geometry(
                geometry_type=GeometryType.CAPSULE,
                dimensions=(dims.get("radius", 0.05), dims.get("length", 0.1)),
            )
        else:
            # Default box
            return Geometry(
                geometry_type=GeometryType.BOX,
                dimensions=(0.1, 0.1, 0.1),
            )
