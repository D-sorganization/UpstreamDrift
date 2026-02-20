"""
URDF validation mixin for the URDFTextEditor.

Extracted from URDFTextEditor to respect SRP:
validation logic is independent of editing, history, and diff concerns.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Protocol, TYPE_CHECKING, cast, Any

import defusedxml.ElementTree as DefusedET

from .text_editor_types import ValidationMessage, ValidationSeverity

if TYPE_CHECKING:
    class ValidationProtocol(Protocol):
        _content: str

logger = logging.getLogger(__name__)


class ValidationMixin:
    """URDF validation operations for the URDFTextEditor.

    Requires host class to provide:
        _content: str
    """

    def validate(self) -> list[ValidationMessage]:
        """
        Validate current URDF content.

        Returns:
            List of validation messages
        """
        messages: list[ValidationMessage] = []

        # XML validation
        messages.extend(self._validate_xml())

        if not any(m.severity == ValidationSeverity.ERROR for m in messages):
            # URDF-specific validation
            messages.extend(self._validate_urdf())

        return messages

    def _validate_xml(self) -> list[ValidationMessage]:
        """Validate XML syntax."""
        messages: list[ValidationMessage] = []
        host = cast("ValidationProtocol", self)

        try:
            DefusedET.fromstring(host._content)
        except ET.ParseError as e:
            # Parse error message for line/column
            error_str = str(e)
            line, col = 1, 0

            # Try to extract line number
            match = re.search(r"line (\d+)", error_str)
            if match:
                line = int(match.group(1))

            match = re.search(r"column (\d+)", error_str)
            if match:
                col = int(match.group(1))

            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=line,
                    column=col,
                    message=f"XML syntax error: {error_str}",
                )
            )

        return messages

    def _validate_urdf(self) -> list[ValidationMessage]:
        """Validate URDF-specific rules."""
        messages: list[ValidationMessage] = []
        host = cast("ValidationProtocol", self)

        try:
            root = DefusedET.fromstring(host._content)
        except ET.ParseError:
            return messages  # Already reported in XML validation

        # Check root element
        if root.tag != "robot":
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=1,
                    column=0,
                    message=f"Root element should be 'robot', got '{root.tag}'",
                )
            )
            return messages

        # Check robot name
        if not root.get("name"):
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    line=1,
                    column=0,
                    message="Robot element missing 'name' attribute",
                    element="robot",
                )
            )

        # Collect and validate links and joints
        links = self._validate_urdf_links(root, messages)
        joints = self._validate_urdf_joints(root, links, messages)

        # Check for orphan links (no joint connection)
        self._validate_urdf_orphan_links(root, links, joints, messages)

        return messages

    def _validate_urdf_links(
        self,
        root: ET.Element,
        messages: list[ValidationMessage],
    ) -> dict[str, ET.Element]:
        """Validate all link elements and return a map of name to element."""
        links: dict[str, ET.Element] = {}

        for _idx, link_elem in enumerate(root.findall("link")):
            name = link_elem.get("name")
            if not name:
                line = self._find_element_line(link_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message="Link element missing 'name' attribute",
                        element="link",
                    )
                )
            elif name in links:
                line = self._find_element_line(link_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Duplicate link name: '{name}'",
                        element=name,
                    )
                )
            else:
                links[name] = link_elem

            # Check inertial mass
            self._validate_link_inertial(link_elem, name, messages)

        return links

    def _validate_link_inertial(
        self,
        link_elem: ET.Element,
        link_name: str | None,
        messages: list[ValidationMessage],
    ) -> None:
        """Validate the inertial/mass properties of a single link element."""
        inertial = link_elem.find("inertial")
        if inertial is None:
            return

        mass_elem = inertial.find("mass")
        if mass_elem is None:
            return

        mass = mass_elem.get("value")
        if mass is None:
            return

        try:
            mass_val = float(mass)
            if mass_val < 0:
                line = self._find_element_line(mass_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Negative mass value: {mass_val}",
                        element=link_name,
                    )
                )
            elif mass_val == 0:
                line = self._find_element_line(mass_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        line=line,
                        column=0,
                        message="Zero mass value",
                        element=link_name,
                    )
                )
        except ValueError:
            line = self._find_element_line(mass_elem)
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=line,
                    column=0,
                    message=f"Invalid mass value: '{mass}'",
                    element=link_name,
                )
            )

    def _validate_urdf_joints(
        self,
        root: ET.Element,
        links: dict[str, ET.Element],
        messages: list[ValidationMessage],
    ) -> dict[str, ET.Element]:
        """Validate all joint elements and return a map of name to element."""
        joints: dict[str, ET.Element] = {}

        for _idx, joint_elem in enumerate(root.findall("joint")):
            name = joint_elem.get("name")
            if not name:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message="Joint element missing 'name' attribute",
                        element="joint",
                    )
                )
            elif name in joints:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Duplicate joint name: '{name}'",
                        element=name,
                    )
                )
            else:
                joints[name] = joint_elem

            # Check joint type
            joint_type = joint_elem.get("type")
            valid_types = {
                "revolute",
                "continuous",
                "prismatic",
                "fixed",
                "floating",
                "planar",
            }
            if joint_type not in valid_types:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Invalid joint type: '{joint_type}'",
                        element=name,
                    )
                )

            # Check parent/child references and limits
            self._validate_joint_references(
                joint_elem, name, joint_type, links, messages
            )

        return joints

    def _validate_joint_references(
        self,
        joint_elem: ET.Element,
        name: str | None,
        joint_type: str | None,
        links: dict[str, ET.Element],
        messages: list[ValidationMessage],
    ) -> None:
        """Validate parent/child references and limits for a single joint."""
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")

        if parent_elem is None:
            line = self._find_element_line(joint_elem)
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=line,
                    column=0,
                    message="Joint missing parent element",
                    element=name,
                )
            )
        else:
            parent_link = parent_elem.get("link")
            if parent_link and parent_link not in links:
                line = self._find_element_line(parent_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Parent link not found: '{parent_link}'",
                        element=name,
                    )
                )

        if child_elem is None:
            line = self._find_element_line(joint_elem)
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=line,
                    column=0,
                    message="Joint missing child element",
                    element=name,
                )
            )
        else:
            child_link = child_elem.get("link")
            if child_link and child_link not in links:
                line = self._find_element_line(child_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Child link not found: '{child_link}'",
                        element=name,
                    )
                )

        # Check limits for revolute/prismatic
        if joint_type in {"revolute", "prismatic"}:
            limit_elem = joint_elem.find("limit")
            if limit_elem is None:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        line=line,
                        column=0,
                        message=f"{joint_type} joint missing limit element",
                        element=name,
                    )
                )

    @staticmethod
    def _validate_urdf_orphan_links(
        root: ET.Element,
        links: dict[str, ET.Element],
        joints: dict[str, ET.Element],
        messages: list[ValidationMessage],
    ) -> None:
        """Check for orphan links that have no joint connection."""
        child_links: set[str | None] = set()
        for joint_elem in root.findall("joint"):
            child_elem = joint_elem.find("child")
            if child_elem is not None:
                child_links.add(child_elem.get("link"))

        for link_name in links:
            if link_name not in child_links:
                # This might be the root link
                is_parent = any(
                    (parent := j.find("parent")) is not None
                    and parent.get("link") == link_name
                    for j in root.findall("joint")
                )
                if not is_parent and len(links) > 1:
                    messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.WARNING,
                            line=1,
                            column=0,
                            message=f"Link '{link_name}' is not connected to any joint",
                            element=link_name,
                        )
                    )

    def _find_element_line(self, elem: ET.Element) -> int:
        """Find the line number of an element (approximate)."""
        host = cast("ValidationProtocol", self)
        ET.tostring(elem, encoding="unicode")
        tag_start = f"<{elem.tag}"

        # Find in content
        lines = host._content.split("\n")
        for idx, line in enumerate(lines, 1):
            if tag_start in line:
                # Check if attributes match
                name = elem.get("name")
                if name is None or f'name="{name}"' in line or f"name='{name}'" in line:
                    return idx

        return 1

    def get_structure_summary(self) -> dict[str, Any]:
        """
        Get high-level statistics about the URDF.

        Returns:
            Dictionary with counts
        """
        host = cast("ValidationProtocol", self)
        try:
            from model_generation.converters.urdf_parser import URDFParser

            parser = URDFParser()
            model = parser.parse_urdf_string(host._content)

            return {
                "valid": True,
                "robot_name": model.robot_name,
                "links": len(model.links),
                "joints": len(model.joints),
                "materials": len(model.materials),
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "links": 0,
                "joints": 0,
                "materials": 0,
            }
