"""
Centralized validation for model generation.

This module provides comprehensive validation for all model components,
ensuring physical correctness and URDF compliance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from model_generation.core.constants import (
    FLOAT_TOLERANCE,
    MIN_INERTIA_KG_M2,
    MIN_MASS_KG,
)

if TYPE_CHECKING:
    from model_generation.core.types import Inertia, Joint, Link

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error."""

    code: str
    message: str
    component: str | None = None
    details: dict | None = None

    def __str__(self) -> str:
        prefix = f"[{self.component}] " if self.component else ""
        return f"{prefix}{self.code}: {self.message}"


@dataclass
class ValidationWarning:
    """Represents a validation warning (non-fatal)."""

    code: str
    message: str
    component: str | None = None
    details: dict | None = None

    def __str__(self) -> str:
        prefix = f"[{self.component}] " if self.component else ""
        return f"{prefix}Warning: {self.message}"


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid

    def add_error(
        self,
        code: str,
        message: str,
        component: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Add an error to the result."""
        self.errors.append(ValidationError(code, message, component, details))
        self.is_valid = False

    def add_warning(
        self,
        code: str,
        message: str,
        component: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Add a warning to the result."""
        self.warnings.append(ValidationWarning(code, message, component, details))

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False

    def get_error_messages(self) -> list[str]:
        """Return list of error message strings."""
        return [str(e) for e in self.errors]

    def get_warning_messages(self) -> list[str]:
        """Return list of warning message strings."""
        return [str(w) for w in self.warnings]


class Validator:
    """
    Centralized validation for model components.

    Provides comprehensive validation for:
    - Inertia tensors (positive-definite, triangle inequality)
    - Mass values (positive)
    - Link hierarchy (no cycles, valid parents)
    - Joint configurations (valid types, axes)
    - Complete models (connectivity, physics)
    """

    # Validation codes
    MASS_ZERO_OR_NEGATIVE = "MASS_001"
    MASS_TOO_SMALL = "MASS_002"
    INERTIA_NOT_POSITIVE_DEFINITE = "INERTIA_001"
    INERTIA_DIAGONAL_NEGATIVE = "INERTIA_002"
    INERTIA_TRIANGLE_INEQUALITY = "INERTIA_003"
    HIERARCHY_CIRCULAR = "HIERARCHY_001"
    HIERARCHY_ORPHAN = "HIERARCHY_002"
    HIERARCHY_DUPLICATE = "HIERARCHY_003"
    JOINT_INVALID_AXIS = "JOINT_001"
    JOINT_INVALID_LIMITS = "JOINT_002"
    JOINT_MISSING_PARENT = "JOINT_003"
    JOINT_MISSING_CHILD = "JOINT_004"

    @classmethod
    def validate_mass(
        cls, mass: float, component: str | None = None
    ) -> ValidationResult:
        """
        Validate mass value.

        Args:
            mass: Mass in kg
            component: Optional component name for error messages

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if mass <= 0:
            result.add_error(
                cls.MASS_ZERO_OR_NEGATIVE,
                f"Mass must be positive (got {mass})",
                component,
            )
        elif mass < MIN_MASS_KG:
            result.add_warning(
                cls.MASS_TOO_SMALL,
                f"Mass {mass} is very small, may cause numerical issues",
                component,
            )

        return result

    @classmethod
    def validate_inertia(
        cls, inertia: Inertia, component: str | None = None, strict: bool = True
    ) -> ValidationResult:
        """
        Validate inertia tensor.

        Args:
            inertia: Inertia object to validate
            component: Optional component name for error messages
            strict: If True, triangle inequality violations are errors

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check mass
        mass_result = cls.validate_mass(inertia.mass, component)
        result.merge(mass_result)

        # Check diagonal elements are positive
        if inertia.ixx <= 0 or inertia.iyy <= 0 or inertia.izz <= 0:
            result.add_error(
                cls.INERTIA_DIAGONAL_NEGATIVE,
                f"Diagonal inertia elements must be positive "
                f"(ixx={inertia.ixx}, iyy={inertia.iyy}, izz={inertia.izz})",
                component,
            )
            return result  # Can't proceed with positive-definite check

        # Check very small inertia values
        min_inertia = min(inertia.ixx, inertia.iyy, inertia.izz)
        if min_inertia < MIN_INERTIA_KG_M2:
            result.add_warning(
                "INERTIA_SMALL",
                f"Minimum inertia {min_inertia} is very small",
                component,
            )

        # Check positive-definite via Cholesky
        if not inertia.is_positive_definite():
            result.add_error(
                cls.INERTIA_NOT_POSITIVE_DEFINITE,
                "Inertia matrix is not positive-definite. "
                "This indicates off-diagonal elements may be too large.",
                component,
                details={"matrix": inertia.to_matrix().tolist()},
            )

        # Check triangle inequality
        if not inertia.satisfies_triangle_inequality():
            msg = (
                "Inertia does not satisfy triangle inequality. "
                "For physical rigid bodies: |Ia - Ib| <= Ic <= Ia + Ib"
            )
            if strict:
                result.add_error(cls.INERTIA_TRIANGLE_INEQUALITY, msg, component)
            else:
                result.add_warning(cls.INERTIA_TRIANGLE_INEQUALITY, msg, component)

        return result

    @classmethod
    def validate_link(cls, link: Link, strict: bool = True) -> ValidationResult:
        """
        Validate a link definition.

        Args:
            link: Link to validate
            strict: If True, more stringent validation

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Validate inertia
        inertia_result = cls.validate_inertia(link.inertia, link.name, strict)
        result.merge(inertia_result)

        # Validate name
        if not link.name or not link.name.strip():
            result.add_error("LINK_NAME_EMPTY", "Link name cannot be empty", link.name)

        return result

    @classmethod
    def validate_joint(cls, joint: Joint, link_names: set[str]) -> ValidationResult:
        """
        Validate a joint definition.

        Args:
            joint: Joint to validate
            link_names: Set of valid link names

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check parent exists
        if joint.parent not in link_names:
            result.add_error(
                cls.JOINT_MISSING_PARENT,
                f"Parent link '{joint.parent}' does not exist",
                joint.name,
            )

        # Check child exists
        if joint.child not in link_names:
            result.add_error(
                cls.JOINT_MISSING_CHILD,
                f"Child link '{joint.child}' does not exist",
                joint.name,
            )

        # Check axis is normalized (for revolute/prismatic)
        from model_generation.core.types import JointType

        if joint.joint_type in (
            JointType.REVOLUTE,
            JointType.CONTINUOUS,
            JointType.PRISMATIC,
        ):
            axis = np.array(joint.axis)
            norm = np.linalg.norm(axis)
            if abs(norm - 1.0) > FLOAT_TOLERANCE:
                if norm < FLOAT_TOLERANCE:
                    result.add_error(
                        cls.JOINT_INVALID_AXIS,
                        "Joint axis cannot be zero vector",
                        joint.name,
                    )
                else:
                    result.add_warning(
                        cls.JOINT_INVALID_AXIS,
                        f"Joint axis is not normalized (norm={norm:.6f})",
                        joint.name,
                    )

        # Check limits for limited joints
        if joint.joint_type == JointType.REVOLUTE:
            if joint.limits and joint.limits.lower >= joint.limits.upper:
                result.add_error(
                    cls.JOINT_INVALID_LIMITS,
                    f"Lower limit ({joint.limits.lower}) must be less than "
                    f"upper limit ({joint.limits.upper})",
                    joint.name,
                )

        return result

    @classmethod
    def validate_hierarchy(
        cls, links: list[Link], joints: list[Joint]
    ) -> ValidationResult:
        """
        Validate link-joint hierarchy.

        Checks for:
        - Circular dependencies
        - Orphaned links
        - Duplicate names
        - Single root

        Args:
            links: List of links
            joints: List of joints

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check for duplicate link names
        link_names = [link.name for link in links]
        seen = set()
        for name in link_names:
            if name in seen:
                result.add_error(
                    cls.HIERARCHY_DUPLICATE,
                    f"Duplicate link name: {name}",
                )
            seen.add(name)

        # Check for duplicate joint names
        joint_names = [joint.name for joint in joints]
        seen = set()
        for name in joint_names:
            if name in seen:
                result.add_error(
                    cls.HIERARCHY_DUPLICATE,
                    f"Duplicate joint name: {name}",
                )
            seen.add(name)

        # Build parent map
        link_name_set = set(link_names)
        parent_map: dict[str, str | None] = {name: None for name in link_names}
        for joint in joints:
            if joint.child in parent_map:
                parent_map[joint.child] = joint.parent

        # Check for orphaned links (except root)
        children = {j.child for j in joints}
        roots = link_name_set - children
        if len(roots) == 0:
            result.add_error(
                cls.HIERARCHY_CIRCULAR,
                "No root link found - possible circular dependency",
            )
        elif len(roots) > 1:
            result.add_warning(
                "HIERARCHY_MULTIPLE_ROOTS",
                f"Multiple root links found: {roots}",
            )

        # Check for circular dependencies using DFS
        def has_cycle(start: str, visited: set[str], path: set[str]) -> bool:
            """Detect cycles via depth-first search from start."""
            if start in path:
                return True
            if start in visited:
                return False
            visited.add(start)
            path.add(start)
            for joint in joints:
                if joint.parent == start:
                    if has_cycle(joint.child, visited, path):
                        return True
            path.remove(start)
            return False

        visited: set[str] = set()
        for link_name in link_names:
            if has_cycle(link_name, visited, set()):
                result.add_error(
                    cls.HIERARCHY_CIRCULAR,
                    f"Circular dependency detected involving {link_name}",
                )
                break

        return result

    @classmethod
    def validate_model(
        cls,
        links: list[Link],
        joints: list[Joint],
        strict: bool = True,
    ) -> ValidationResult:
        """
        Validate complete model.

        Args:
            links: All links in model
            joints: All joints in model
            strict: If True, more stringent validation

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Validate hierarchy
        hierarchy_result = cls.validate_hierarchy(links, joints)
        result.merge(hierarchy_result)

        # Validate each link
        for link in links:
            link_result = cls.validate_link(link, strict)
            result.merge(link_result)

        # Validate each joint
        link_names = {link.name for link in links}
        for joint in joints:
            joint_result = cls.validate_joint(joint, link_names)
            result.merge(joint_result)

        return result
