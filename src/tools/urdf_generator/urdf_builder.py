"""URDF builder for creating and managing URDF content."""

from src.shared.python.logging_config import get_logger
import math
from enum import Enum
from xml.etree.ElementTree import Element, SubElement, tostring

import defusedxml.minidom as minidom
import numpy as np

logger = get_logger(__name__)

# Default moment of inertia value (kg·m²) used when inertia parameters are not specified.
# This value (0.1 kg·m²) represents a reasonable default for small-to-medium rigid bodies
# in golf biomechanics applications (e.g., club segments, body parts).
# Reference: Typical inertia values from ergonomic and sports biomechanics literature.
DEFAULT_INERTIA_MOMENT = 0.1


class Handedness(Enum):
    """Golfer handedness for model configuration.

    Golf models can be mirrored for left-handed vs right-handed golfers.
    The default is right-handed.
    """

    RIGHT = "right"
    LEFT = "left"


class URDFBuilder:
    """Builder class for creating URDF files with support for parallel configurations."""

    def __init__(self) -> None:
        """Initialize the URDF builder."""
        self.segments: list[dict] = []
        self.materials: dict[str, dict] = {}
        self.robot_name = "golf_robot"
        self.handedness: Handedness = Handedness.RIGHT

    def _validate_physical_parameters(self, segment_data: dict) -> None:
        """Validate physical parameters for correctness.

        Args:
            segment_data: Segment data dictionary.

        Raises:
            ValueError: If parameters violate physics constraints.
        """
        physics = segment_data.get("physics", {})
        segment_name = segment_data.get("name", "unknown")

        # Validate mass
        mass = physics.get("mass", 1.0)
        if mass <= 0:
            raise ValueError(
                f"Mass must be positive\\n"
                f"Segment: {segment_name}\\n"
                f"Got: {mass} kg\\n"
                f"Hint: Check the mass value in the physics properties."
            )

        # Validate inertia matrix if provided
        inertia = physics.get("inertia", {})
        if inertia:
            # Extract inertia components (full 3x3 symmetric matrix)
            ixx = inertia.get("ixx", DEFAULT_INERTIA_MOMENT)
            iyy = inertia.get("iyy", DEFAULT_INERTIA_MOMENT)
            izz = inertia.get("izz", DEFAULT_INERTIA_MOMENT)
            ixy = inertia.get("ixy", 0.0)
            ixz = inertia.get("ixz", 0.0)
            iyz = inertia.get("iyz", 0.0)

            # Build 3x3 inertia matrix
            inertia_matrix = np.array(
                [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
            )

            # Check 1: Positive diagonal elements
            if ixx <= 0 or iyy <= 0 or izz <= 0:
                raise ValueError(
                    f"Inertia diagonal elements must be positive\\n"
                    f"Segment: {segment_name}\\n"
                    f"Ixx={ixx:.6f}, Iyy={iyy:.6f}, Izz={izz:.6f}\\n"
                    f"Hint: Check for negative signs or zero values."
                )

            # Check 2: Positive-definite via Cholesky decomposition
            try:
                np.linalg.cholesky(inertia_matrix)
            except np.linalg.LinAlgError as e:
                # Preserve exception chain for debugging eigenvalue issues
                raise ValueError(
                    f"Inertia matrix must be positive-definite\\n"
                    f"Segment: {segment_name}\\n"
                    f"Inertia matrix:\\n{inertia_matrix}\\n"
                    f"Hint: Check off-diagonal elements (ixy, ixz, iyz) for consistency.\\n"
                    f"The matrix must be symmetric and all eigenvalues positive."
                ) from e

            # Check 3: Triangle inequality (parallel axis theorem bounds)
            # For any rigid body: |I_a - I_b| <= I_c <= I_a + I_b
            if not (abs(ixx - iyy) <= izz <= ixx + iyy):
                logger.warning(
                    f"Inertia values may violate triangle inequality\\n"
                    f"Segment: {segment_name}\\n"
                    f"Ixx={ixx:.6f}, Iyy={iyy:.6f}, Izz={izz:.6f}\\n"
                    f"This may indicate an unusual mass distribution."
                )
            if not (abs(iyy - izz) <= ixx <= iyy + izz):
                logger.warning(
                    f"Inertia values may violate triangle inequality (YZ plane)\\n"
                    f"Segment: {segment_name}"
                )
            if not (abs(ixx - izz) <= iyy <= ixx + izz):
                logger.warning(
                    f"Inertia values may violate triangle inequality (XZ plane)\\n"
                    f"Segment: {segment_name}"
                )

    def add_segment(self, segment_data: dict) -> None:
        """Add a segment to the URDF.

        Args:
            segment_data: Dictionary containing segment information.

        Raises:
            ValueError: If segment data is invalid or physically incorrect.
        """
        # Validate segment data
        if not segment_data.get("name"):
            raise ValueError("Segment must have a name")

        # Check for duplicate names
        if any(seg["name"] == segment_data["name"] for seg in self.segments):
            raise ValueError(
                f"Segment with name '{segment_data['name']}' already exists"
            )

        # Validate physical parameters
        self._validate_physical_parameters(segment_data)

        self.segments.append(segment_data.copy())

        # Add material if it has a name
        material = segment_data.get("physics", {}).get("material", {})
        if material.get("name"):
            self.materials[material["name"]] = material

        logger.info(f"Added segment: {segment_data['name']}")

    def remove_segment(self, segment_name: str) -> None:
        """Remove a segment from the URDF.

        Args:
            segment_name: Name of the segment to remove.
        """
        # Find and remove the segment
        original_count = len(self.segments)
        self.segments = [seg for seg in self.segments if seg["name"] != segment_name]

        if len(self.segments) == original_count:
            raise ValueError(f"Segment '{segment_name}' not found")

        # Remove any child segments
        children_to_remove = []
        for seg in self.segments:
            if seg.get("parent") == segment_name:
                children_to_remove.append(seg["name"])

        for child_name in children_to_remove:
            self.remove_segment(child_name)

        logger.info(f"Removed segment: {segment_name}")

    def modify_segment(self, segment_data: dict) -> None:
        """Modify an existing segment.

        Args:
            segment_data: Dictionary containing updated segment information.
        """
        segment_name = segment_data.get("name")
        if not segment_name:
            raise ValueError("Segment must have a name")

        # Find and update the segment
        for i, seg in enumerate(self.segments):
            if seg["name"] == segment_name:
                self.segments[i] = segment_data.copy()

                # Update material
                material = segment_data.get("physics", {}).get("material", {})
                if material.get("name"):
                    self.materials[material["name"]] = material

                logger.info(f"Modified segment: {segment_name}")
                return

        raise ValueError(f"Segment '{segment_name}' not found")

    def get_urdf(self) -> str:
        """Generate the URDF XML content.

        Returns:
            URDF XML content as a string.
        """
        if not self.segments:
            return self._create_empty_urdf()

        # Create root element
        robot = Element("robot", name=self.robot_name)

        # Add materials
        self._add_materials(robot)

        # Add links and joints
        self._add_links_and_joints(robot)

        # Pretty print the XML
        rough_string = tostring(robot, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return str(reparsed.toprettyxml(indent="  "))

    def _create_empty_urdf(self) -> str:
        """Create an empty URDF structure.

        Returns:
            Empty URDF XML content.
        """
        robot = Element("robot", name=self.robot_name)

        # Add a base link for empty URDF
        base_link = SubElement(robot, "link", name="base_link")
        visual = SubElement(base_link, "visual")
        geometry = SubElement(visual, "geometry")
        SubElement(geometry, "box", size="0.1 0.1 0.1")

        rough_string = tostring(robot, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return str(reparsed.toprettyxml(indent="  "))

    def _add_materials(self, robot: Element) -> None:
        """Add material definitions to the URDF.

        Args:
            robot: Root robot element.
        """
        for material_name, material_data in self.materials.items():
            material_elem = SubElement(robot, "material", name=material_name)

            color = material_data.get("color", {})
            if color:
                color_str = f"{color.get('r', 0.8)} {color.get('g', 0.8)} {color.get('b', 0.8)} {color.get('a', 1.0)}"
                SubElement(material_elem, "color", rgba=color_str)

    def _add_links_and_joints(self, robot: Element) -> None:
        """Add links and joints to the URDF.

        Args:
            robot: Root robot element.
        """
        # Sort segments to ensure parents are processed before children
        sorted_segments = self._sort_segments_by_hierarchy()

        for segment in sorted_segments:
            self._add_link(robot, segment)

            # Add joint if this segment has a parent
            if segment.get("parent"):
                self._add_joint(robot, segment)

    def _sort_segments_by_hierarchy(self) -> list[dict]:
        """Sort segments by hierarchy (parents before children).

        Returns:
            List of segments sorted by hierarchy.
        """
        sorted_segments = []
        processed = set()

        def add_segment_and_children(segment_name: str | None) -> None:
            """Recursively add segment and its children."""
            # Find segments with this parent
            children = [
                seg
                for seg in self.segments
                if seg.get("parent") == segment_name and seg["name"] not in processed
            ]

            for child in children:
                processed.add(child["name"])
                sorted_segments.append(child)
                add_segment_and_children(child["name"])

        # Start with root segments (no parent)
        add_segment_and_children(None)

        return sorted_segments

    def _add_link(self, robot: Element, segment: dict) -> None:
        """Add a link element to the URDF.

        Args:
            robot: Root robot element.
            segment: Segment data.
        """
        link = SubElement(robot, "link", name=segment["name"])

        # Add visual
        self._add_visual(link, segment)

        # Add collision (same as visual for now)
        self._add_collision(link, segment)

        # Add inertial
        self._add_inertial(link, segment)

    def _add_visual(self, link: Element, segment: dict) -> None:
        """Add visual element to a link.

        Args:
            link: Link element.
            segment: Segment data.
        """
        visual = SubElement(link, "visual")

        # Add origin
        self._add_origin(visual, segment["geometry"])

        # Add geometry
        geometry = SubElement(visual, "geometry")
        self._add_geometry(geometry, segment["geometry"])

        # Add material
        material = segment.get("physics", {}).get("material", {})
        if material.get("name"):
            SubElement(visual, "material", name=material["name"])

    def _add_collision(self, link: Element, segment: dict) -> None:
        """Add collision element to a link.

        Args:
            link: Link element.
            segment: Segment data.
        """
        collision = SubElement(link, "collision")

        # Add origin
        self._add_origin(collision, segment["geometry"])

        # Add geometry
        geometry = SubElement(collision, "geometry")
        self._add_geometry(geometry, segment["geometry"])

    def _add_inertial(self, link: Element, segment: dict) -> None:
        """Add inertial element to a link.

        Args:
            link: Link element.
            segment: Segment data.
        """
        inertial = SubElement(link, "inertial")

        # Add origin (center of mass)
        origin = SubElement(inertial, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")

        # Add mass
        physics = segment.get("physics", {})
        mass_value = physics.get("mass", 1.0)
        SubElement(inertial, "mass", value=str(mass_value))

        # Add inertia
        inertia_data = physics.get("inertia", {})
        inertia = SubElement(inertial, "inertia")
        inertia.set("ixx", str(inertia_data.get("ixx", 0.1)))
        inertia.set("ixy", "0")
        inertia.set("ixz", "0")
        inertia.set("iyy", str(inertia_data.get("iyy", 0.1)))
        inertia.set("iyz", "0")
        inertia.set("izz", str(inertia_data.get("izz", 0.1)))

    def _add_origin(self, parent: Element, geometry: dict) -> None:
        """Add origin element.

        Args:
            parent: Parent element.
            geometry: Geometry data containing position and orientation.
        """
        position = geometry.get("position", {})
        orientation = geometry.get("orientation", {})

        xyz = f"{position.get('x', 0)} {position.get('y', 0)} {position.get('z', 0)}"

        # Convert degrees to radians
        roll = math.radians(orientation.get("roll", 0))
        pitch = math.radians(orientation.get("pitch", 0))
        yaw = math.radians(orientation.get("yaw", 0))
        rpy = f"{roll} {pitch} {yaw}"

        origin = SubElement(parent, "origin")
        origin.set("xyz", xyz)
        origin.set("rpy", rpy)

    def _add_geometry(self, geometry: Element, geom_data: dict) -> None:
        """Add geometry element.

        Args:
            geometry: Geometry element.
            geom_data: Geometry data.
        """
        shape = geom_data.get("shape", "Box").lower()
        dimensions = geom_data.get("dimensions", {})

        if shape == "box":
            size = f"{dimensions.get('length', 1.0)} {dimensions.get('width', 0.1)} {dimensions.get('height', 0.1)}"
            SubElement(geometry, "box", size=size)
        elif shape == "cylinder":
            radius = dimensions.get("width", 0.1) / 2  # Use width as diameter
            length = dimensions.get("length", 1.0)
            cylinder = SubElement(geometry, "cylinder")
            cylinder.set("radius", str(radius))
            cylinder.set("length", str(length))
        elif shape == "sphere":
            radius = dimensions.get("length", 0.1) / 2  # Use length as diameter
            SubElement(geometry, "sphere", radius=str(radius))
        elif shape == "capsule":
            radius = dimensions.get("width", 0.1) / 2
            length = dimensions.get("length", 1.0)
            capsule = SubElement(geometry, "capsule")
            capsule.set("radius", str(radius))
            capsule.set("length", str(length))
        else:
            # Default to box
            size = f"{dimensions.get('length', 1.0)} {dimensions.get('width', 0.1)} {dimensions.get('height', 0.1)}"
            SubElement(geometry, "box", size=size)

    def _add_joint(self, robot: Element, segment: dict) -> None:
        """Add a joint element to the URDF.

        Args:
            robot: Root robot element.
            segment: Segment data.
        """
        joint_name = f"{segment['parent']}_to_{segment['name']}"
        joint_data = segment.get("joint", {})
        joint_type = joint_data.get("type", "fixed")

        joint = SubElement(robot, "joint", name=joint_name, type=joint_type)

        # Add parent and child
        SubElement(joint, "parent", link=segment["parent"])
        SubElement(joint, "child", link=segment["name"])

        # Add origin
        self._add_origin(joint, segment["geometry"])

        # Add axis for revolute/prismatic joints
        if joint_type in ["revolute", "prismatic", "continuous"]:
            axis_data = joint_data.get("axis", {})
            axis_str = f"{axis_data.get('x', 0)} {axis_data.get('y', 0)} {axis_data.get('z', 1)}"
            SubElement(joint, "axis", xyz=axis_str)

        # Add limits for revolute/prismatic joints
        if joint_type in ["revolute", "prismatic"]:
            limits = joint_data.get("limits", {})
            limit_elem = SubElement(joint, "limit")

            if joint_type == "revolute":
                # Convert degrees to radians
                lower = math.radians(limits.get("lower", -180))
                upper = math.radians(limits.get("upper", 180))
            else:  # prismatic
                lower = limits.get("lower", -1.0)
                upper = limits.get("upper", 1.0)

            limit_elem.set("lower", str(lower))
            limit_elem.set("upper", str(upper))
            limit_elem.set("velocity", str(limits.get("velocity", 10.0)))
            limit_elem.set("effort", str(limits.get("effort", 100.0)))

    def clear(self) -> None:
        """Clear all segments and materials."""
        self.segments.clear()
        self.materials.clear()
        logger.info("URDF builder cleared")

    def set_robot_name(self, name: str) -> None:
        """Set the robot name.

        Args:
            name: New robot name.
        """
        self.robot_name = name
        logger.info(f"Robot name set to: {name}")

    def get_segment_count(self) -> int:
        """Get the number of segments.

        Returns:
            Number of segments.
        """
        return len(self.segments)

    def get_segment_names(self) -> list[str]:
        """Get list of segment names.

        Returns:
            List of segment names.
        """
        return [seg["name"] for seg in self.segments]

    def validate_urdf(self) -> list[str]:
        """Validate the URDF structure.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Check for circular dependencies
        def has_circular_dependency(segment_name: str, visited: set) -> bool:
            """Check for circular dependencies in the segment hierarchy.

            Args:
                segment_name: Name of the current segment.
                visited: Set of visited segment names.

            Returns:
                True if a circular dependency is detected.
            """
            if segment_name in visited:
                return True

            visited.add(segment_name)

            # Find children
            children = [
                seg for seg in self.segments if seg.get("parent") == segment_name
            ]
            for child in children:
                if has_circular_dependency(child["name"], visited.copy()):
                    return True

            return False

        for segment in self.segments:
            if has_circular_dependency(segment["name"], set()):
                errors.append(
                    f"Circular dependency detected involving segment: {segment['name']}"
                )

        # Check for orphaned segments (parent doesn't exist)
        segment_names = {seg["name"] for seg in self.segments}
        for segment in self.segments:
            parent = segment.get("parent")
            if parent and parent not in segment_names:
                errors.append(
                    f"Segment '{segment['name']}' has non-existent parent: '{parent}'"
                )

        return errors

    def set_handedness(self, handedness: Handedness) -> None:
        """Set the handedness for the model.

        Args:
            handedness: Handedness.LEFT or Handedness.RIGHT
        """
        self.handedness = handedness
        logger.info(f"Handedness set to: {handedness.value}")

    def get_handedness(self) -> Handedness:
        """Get the current handedness setting.

        Returns:
            Current Handedness value.
        """
        return self.handedness

    def mirror_for_handedness(self) -> None:
        """Mirror all segments for opposite handedness (Task 3.4).

        Mirrors the Y-axis of all segment positions to convert between
        left-handed and right-handed configurations.

        This method:
        - Negates the Y position of all segments
        - Adjusts joint axes as needed
        - Renames segments with 'left_'/'right_' prefix swap
        """
        for segment in self.segments:
            geometry = segment.get("geometry", {})
            position = geometry.get("position", {})

            # Mirror Y-axis position
            if "y" in position:
                position["y"] = -position["y"]

            # Update geometry
            segment["geometry"]["position"] = position

            # Mirror joint axes if applicable
            joint = segment.get("joint", {})
            axis = joint.get("axis", {})
            if "y" in axis:
                axis["y"] = -axis.get("y", 0)
            joint["axis"] = axis
            segment["joint"] = joint

            # Handle segment naming for body parts
            name = segment.get("name", "")
            if name.startswith("left_"):
                segment["name"] = name.replace("left_", "right_", 1)
            elif name.startswith("right_"):
                segment["name"] = name.replace("right_", "left_", 1)

            # Handle parent naming
            parent = segment.get("parent", "")
            if parent and parent.startswith("left_"):
                segment["parent"] = parent.replace("left_", "right_", 1)
            elif parent and parent.startswith("right_"):
                segment["parent"] = parent.replace("right_", "left_", 1)

        # Toggle handedness state
        if self.handedness == Handedness.RIGHT:
            self.handedness = Handedness.LEFT
        else:
            self.handedness = Handedness.RIGHT

        logger.info(f"Mirrored model for {self.handedness.value}-handed golfer")

    def get_mirrored_urdf(self, target_handedness: Handedness) -> str:
        """Get URDF mirrored for specified handedness.

        Does not modify the internal state. Returns a mirrored copy if needed.

        Args:
            target_handedness: Desired handedness for the output.

        Returns:
            URDF XML string configured for the target handedness.
        """
        if target_handedness == self.handedness:
            return self.get_urdf()

        # Create a copy, mirror it, generate URDF, then restore
        import copy

        original_segments = copy.deepcopy(self.segments)
        original_handedness = self.handedness

        try:
            self.mirror_for_handedness()
            return self.get_urdf()
        finally:
            self.segments = original_segments
            self.handedness = original_handedness
