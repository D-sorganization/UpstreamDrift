"""
Base builder interface for URDF generation.

Defines the abstract interface that all builders implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts import InvariantError
from model_generation.core.contracts import precondition
from model_generation.core.types import Joint, Link
from model_generation.core.validation import ValidationResult


@dataclass
class BuildResult:
    """Result of a build operation."""

    # Whether build was successful
    success: bool

    # Generated URDF XML string
    urdf_xml: str | None = None

    # All links in the model
    links: list[Link] = field(default_factory=list)

    # All joints in the model
    joints: list[Joint] = field(default_factory=list)

    # Validation result
    validation: ValidationResult | None = None

    # Error message if build failed
    error_message: str | None = None

    # Output path if saved to file
    output_path: Path | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_link(self, name: str) -> Link | None:
        """Get a link by name."""
        assert name is not None, "name must be provided"
        for link in self.links:
            if link.name == name:
                return link
        return None

    def get_joint(self, name: str) -> Joint | None:
        """Get a joint by name."""
        assert name is not None, "name must be provided"
        for joint in self.joints:
            if joint.name == name:
                return joint
        return None

    def get_root_link(self) -> Link | None:
        """Get the root (base) link."""
        child_names = {j.child for j in self.joints}
        for link in self.links:
            if link.name not in child_names:
                return link
        return self.links[0] if self.links else None

    def get_children(self, link_name: str) -> list[str]:
        """Get names of child links."""
        return [j.child for j in self.joints if j.parent == link_name]

    def get_total_mass(self) -> float:
        """Get total mass of all links."""
        return sum(link.inertia.mass for link in self.links)

    def get_total_dof(self) -> int:
        """Get total degrees of freedom."""
        return sum(j.get_dof_count() for j in self.joints)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "link_count": len(self.links),
            "joint_count": len(self.joints),
            "total_mass": self.get_total_mass(),
            "total_dof": self.get_total_dof(),
            "error_message": self.error_message,
            "output_path": str(self.output_path) if self.output_path else None,
            "metadata": self.metadata,
        }


class BaseURDFBuilder(ABC):
    """
    Abstract base class for URDF builders.

    All builders (manual, parametric, composite) implement this interface.
    """

    def __init__(self, robot_name: str = "robot"):
        """
        Initialize builder.

        Args:
            robot_name: Name for the robot element
        """
        assert robot_name is not None, "robot_name must be provided"
        self._robot_name = robot_name
        self._links: list[Link] = []
        self._joints: list[Joint] = []
        self._materials: dict[str, Any] = {}

    @property
    def robot_name(self) -> str:
        """Get robot name."""
        return self._robot_name

    @robot_name.setter
    def robot_name(self, name: str) -> None:
        """Set robot name."""
        self._robot_name = name

    @property
    def links(self) -> list[Link]:
        """Get all links."""
        return self._links.copy()

    @property
    def joints(self) -> list[Joint]:
        """Get all joints."""
        return self._joints.copy()

    @property
    def link_count(self) -> int:
        """Get number of links."""
        return len(self._links)

    @property
    def joint_count(self) -> int:
        """Get number of joints."""
        return len(self._joints)

    def get_link_names(self) -> list[str]:
        """Get all link names."""
        return [link.name for link in self._links]

    def get_joint_names(self) -> list[str]:
        """Get all joint names."""
        return [joint.name for joint in self._joints]

    @precondition(lambda link: link is not None, "Link cannot be None")
    @precondition(lambda link: link.name, "Link must have a name")
    def add_link(self, link: Link) -> None:
        """
        Add a link to the model.

        Args:
            link: Link to add

        Raises:
            ContractViolationError: If link is None or has no name
        """
        self._links.append(link)

    @precondition(lambda joint: joint is not None, "Joint cannot be None")
    @precondition(lambda joint: joint.name, "Joint must have a name")
    @precondition(lambda joint: joint.parent, "Joint must have a parent")
    @precondition(lambda joint: joint.child, "Joint must have a child")
    def add_joint(self, joint: Joint) -> None:
        """
        Add a joint to the model.

        Args:
            joint: Joint to add

        Raises:
            ContractViolationError: If joint is None or missing required fields
        """
        self._joints.append(joint)

    @abstractmethod
    def build(self, **kwargs: Any) -> BuildResult:
        """
        Build the URDF model.

        Returns:
            BuildResult with generated URDF and metadata
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all links and joints."""
        ...

    def _check_invariants(self) -> None:
        """Validate that _links and _joints remain consistent.

        Checks:
        - No duplicate link names
        - No duplicate joint names
        - Every joint parent references an existing link
        - Every joint child references an existing link

        Raises:
            InvariantError: If any invariant is violated.
        """
        # Check for duplicate link names
        link_names = [link.name for link in self._links]
        if len(link_names) != len(set(link_names)):
            seen: set[str] = set()
            for name in link_names:
                if name in seen:
                    raise InvariantError(f"Duplicate link name detected: '{name}'")
                seen.add(name)

        # Check for duplicate joint names
        joint_names = [joint.name for joint in self._joints]
        if len(joint_names) != len(set(joint_names)):
            seen_joints: set[str] = set()
            for name in joint_names:
                if name in seen_joints:
                    raise InvariantError(f"Duplicate joint name detected: '{name}'")
                seen_joints.add(name)

        # Check joint references
        link_name_set = set(link_names)
        for joint in self._joints:
            if joint.parent not in link_name_set:
                raise InvariantError(
                    f"Joint '{joint.name}' references non-existent "
                    f"parent link '{joint.parent}'"
                )
            if joint.child not in link_name_set:
                raise InvariantError(
                    f"Joint '{joint.name}' references non-existent "
                    f"child link '{joint.child}'"
                )

    def validate(self) -> ValidationResult:
        """
        Validate the current model.

        Returns:
            ValidationResult with any errors/warnings
        """
        from model_generation.core.validation import Validator

        return Validator.validate_model(self._links, self._joints)

    def to_urdf(self, pretty_print: bool = True) -> str:
        """
        Generate URDF XML string.

        Args:
            pretty_print: If True, format with indentation

        Returns:
            URDF XML string
        """
        assert pretty_print is not None, "pretty_print must be provided"
        from model_generation.builders.urdf_writer import URDFWriter

        writer = URDFWriter(pretty_print=pretty_print)
        return writer.write(
            self._robot_name, self._links, self._joints, self._materials
        )

    def save(self, path: str | Path, pretty_print: bool = True) -> Path:
        """
        Save URDF to file.

        Args:
            path: Output file path
            pretty_print: If True, format with indentation

        Returns:
            Path to saved file
        """
        assert path is not None, "path must be provided"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        urdf_xml = self.to_urdf(pretty_print=pretty_print)
        path.write_text(urdf_xml)

        return path
