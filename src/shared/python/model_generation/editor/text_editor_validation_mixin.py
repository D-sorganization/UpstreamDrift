"""URDF validation mixin for URDFTextEditor.

Extracts XML/URDF validation logic from the main editor class to improve
single-responsibility adherence.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import defusedxml.ElementTree as DefusedET

if TYPE_CHECKING:
    from .text_editor import ValidationMessage

logger = logging.getLogger(__name__)


class URDFValidationMixin:
    """Mixin providing XML and URDF validation for URDFTextEditor."""

    _content: str  # provided by URDFTextEditor

    _VALID_JOINT_TYPES = frozenset(
        {
            "revolute",
            "continuous",
            "prismatic",
            "fixed",
            "floating",
            "planar",
        }
    )

    def validate(self) -> list[ValidationMessage]:
        """
        Validate current URDF content.

        Returns:
            List of validation messages
        """
        messages = []

        # XML validation
        messages.extend(self._validate_xml())

        from .text_editor import ValidationSeverity

        if not any(m.severity == ValidationSeverity.ERROR for m in messages):
            # URDF-specific validation
            messages.extend(self._validate_urdf())

        return messages

    def _validate_xml(self) -> list[ValidationMessage]:
        """Validate XML syntax."""
        from .text_editor import ValidationMessage, ValidationSeverity

        messages = []

        try:
            DefusedET.fromstring(self._content)
        except DefusedET.ParseError as e:
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

        try:
            root = DefusedET.fromstring(self._content)
        except DefusedET.ParseError:
            return messages  # Already reported in XML validation

        if not self._validate_root_element(root, messages):
            return messages

        links = self._validate_links(root, messages)
        self._validate_joints(root, links, messages)
        self._validate_orphan_links(root, links, messages)
        return messages

    def _validate_root_element(
        self,
        root: ET.Element,
        messages: list[ValidationMessage],
    ) -> bool:
        """Check root is <robot> with a name. Return False to abort."""
        from .text_editor import ValidationMessage, ValidationSeverity

        if root is None:
            raise ValueError("root must be provided")
        if root.tag != "robot":
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=1,
                    column=0,
                    message=(f"Root element should be 'robot', got '{root.tag}'"),
                )
            )
            return False

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
        return True

    def _validate_links(
        self,
        root: ET.Element,
        messages: list[ValidationMessage],
    ) -> dict[str, ET.Element]:
        """Validate link elements and return name→element map."""
        from .text_editor import ValidationMessage, ValidationSeverity

        if root is None:
            raise ValueError("root must be provided")
        links: dict[str, ET.Element] = {}

        for link_elem in root.findall("link"):
            name = link_elem.get("name")
            if not name:
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=self._find_element_line(link_elem),
                        column=0,
                        message="Link element missing 'name' attribute",
                        element="link",
                    )
                )
            elif name in links:
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=self._find_element_line(link_elem),
                        column=0,
                        message=f"Duplicate link name: '{name}'",
                        element=name,
                    )
                )
            else:
                links[name] = link_elem

            self._validate_link_inertial(
                link_elem,
                name,
                messages,
            )

        return links

    def _validate_link_inertial(
        self,
        link_elem: ET.Element,
        name: str | None,
        messages: list[ValidationMessage],
    ) -> None:
        """Validate inertial/mass properties of a link."""
        from .text_editor import ValidationMessage, ValidationSeverity

        if link_elem is None:
            raise ValueError("link_elem must be provided")
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
        except ValueError:
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=self._find_element_line(mass_elem),
                    column=0,
                    message=f"Invalid mass value: '{mass}'",
                    element=name,
                )
            )
            return

        if mass_val < 0:
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=self._find_element_line(mass_elem),
                    column=0,
                    message=f"Negative mass value: {mass_val}",
                    element=name,
                )
            )
        elif mass_val == 0:
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    line=self._find_element_line(mass_elem),
                    column=0,
                    message="Zero mass value",
                    element=name,
                )
            )

    def _validate_joints(
        self,
        root: ET.Element,
        links: dict[str, ET.Element],
        messages: list[ValidationMessage],
    ) -> None:
        """Validate joint elements (type, parent/child, limits)."""
        from .text_editor import ValidationMessage, ValidationSeverity

        if root is None:
            raise ValueError("root must be provided")
        seen: dict[str, ET.Element] = {}

        for joint_elem in root.findall("joint"):
            name = joint_elem.get("name")
            if not name:
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=self._find_element_line(joint_elem),
                        column=0,
                        message="Joint element missing 'name' attribute",
                        element="joint",
                    )
                )
            elif name in seen:
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=self._find_element_line(joint_elem),
                        column=0,
                        message=f"Duplicate joint name: '{name}'",
                        element=name,
                    )
                )
            else:
                seen[name] = joint_elem

            joint_type = joint_elem.get("type")
            if joint_type not in self._VALID_JOINT_TYPES:
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=self._find_element_line(joint_elem),
                        column=0,
                        message=f"Invalid joint type: '{joint_type}'",
                        element=name,
                    )
                )

            self._validate_joint_refs(
                joint_elem,
                name,
                links,
                messages,
            )

            if (
                joint_type in {"revolute", "prismatic"}
                and joint_elem.find("limit") is None
            ):
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        line=self._find_element_line(joint_elem),
                        column=0,
                        message=(f"{joint_type} joint missing limit element"),
                        element=name,
                    )
                )

    def _validate_joint_refs(
        self,
        joint_elem: ET.Element,
        name: str | None,
        links: dict[str, ET.Element],
        messages: list[ValidationMessage],
    ) -> None:
        """Validate parent/child link references for a joint."""
        from .text_editor import ValidationMessage, ValidationSeverity

        for role in ("parent", "child"):
            ref_elem = joint_elem.find(role)
            if ref_elem is None:
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=self._find_element_line(joint_elem),
                        column=0,
                        message=f"Joint missing {role} element",
                        element=name,
                    )
                )
            else:
                link_name = ref_elem.get("link")
                if link_name and link_name not in links:
                    messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.ERROR,
                            line=self._find_element_line(ref_elem),
                            column=0,
                            message=(f"{role.title()} link not found: '{link_name}'"),
                            element=name,
                        )
                    )

    def _validate_orphan_links(
        self,
        root: ET.Element,
        links: dict[str, ET.Element],
        messages: list[ValidationMessage],
    ) -> None:
        """Detect links that are not connected to any joint."""
        from .text_editor import ValidationMessage, ValidationSeverity

        if root is None:
            raise ValueError("root must be provided")
        child_links = set()
        for joint_elem in root.findall("joint"):
            child_elem = joint_elem.find("child")
            if child_elem is not None:
                child_links.add(child_elem.get("link"))

        for link_name in links:
            if link_name not in child_links:
                is_parent = any(
                    (pe := j.find("parent")) is not None and pe.get("link") == link_name
                    for j in root.findall("joint")
                )
                if not is_parent and len(links) > 1:
                    messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.WARNING,
                            line=1,
                            column=0,
                            message=(
                                f"Link '{link_name}' is not connected to any joint"
                            ),
                            element=link_name,
                        )
                    )

    def _find_element_line(self, elem: ET.Element) -> int:
        """Find the line number of an element (approximate)."""
        # This is a simple heuristic - search for element in content
        if elem is None:
            raise ValueError("elem must be provided")
        ET.tostring(elem, encoding="unicode")
        tag_start = f"<{elem.tag}"

        # Find in content
        lines = self._content.split("\n")
        for idx, line in enumerate(lines, 1):
            if tag_start in line:
                # Check if attributes match
                name = elem.get("name")
                if name is None or f'name="{name}"' in line or f"name='{name}'" in line:
                    return idx

        return 1
