from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EndEffector:
    """Represents an end effector configuration."""

    name: str
    link_element: ET.Element
    joint_element: ET.Element | None  # Joint connecting to parent
    child_links: list[ET.Element]  # Links that are part of this end effector
    child_joints: list[ET.Element]  # Joints within the end effector
    source_file: Path | None = None

    def get_all_link_names(self) -> list[str]:
        """Get names of all links in this end effector."""
        names = [self.link_element.get("name", "")]
        for link in self.child_links:
            names.append(link.get("name", ""))
        return names

    def get_attachment_joint_type(self) -> str:
        """Get the joint type for attaching to parent."""
        if self.joint_element is not None:
            return self.joint_element.get("type", "fixed")
        return "fixed"

    def to_xml_elements(self) -> tuple[list[ET.Element], list[ET.Element]]:
        """Convert to XML elements (links, joints)."""
        links = [copy.deepcopy(self.link_element)]
        links.extend(copy.deepcopy(link) for link in self.child_links)

        joints = []
        if self.joint_element is not None:
            joints.append(copy.deepcopy(self.joint_element))
        joints.extend(copy.deepcopy(joint) for joint in self.child_joints)

        return links, joints
