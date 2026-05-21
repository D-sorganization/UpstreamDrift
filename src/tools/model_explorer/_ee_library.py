from __future__ import annotations

from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DefusedET
import defusedxml.ElementTree as ET  # noqa: S314  # Security: defusedxml prevents XML attacks

from src.tools.model_explorer._ee_model import EndEffector


class EndEffectorLibrary:
    """Library of available end effectors."""

    def __init__(self) -> None:
        """Initialize the library."""
        self.end_effectors: dict[str, EndEffector] = {}
        self._builtin_definitions = self._create_builtin_definitions()

    def _create_builtin_definitions(self) -> dict[str, dict[str, Any]]:
        """Create built-in end effector definitions."""
        return {
            "simple_gripper": {
                "name": "Simple Gripper",
                "description": "Basic two-finger parallel gripper",
                "link_xml": """
                    <link name="gripper_base">
                        <inertial>
                            <mass value="0.5"/>
                            <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
                        </inertial>
                        <visual>
                            <geometry><box size="0.08 0.08 0.02"/></geometry>
                            <material name="gray"><color rgba="0.5 0.5 0.5 1"/></material>
                        </visual>
                        <collision><geometry><box size="0.08 0.08 0.02"/></geometry></collision>
                    </link>
                """,
                "child_links": [
                    """<link name="left_finger">
                        <inertial><mass value="0.1"/>
                        <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
                        <visual><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry>
                        <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
                        <collision><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry></collision>
                    </link>""",
                    """<link name="right_finger">
                        <inertial><mass value="0.1"/>
                        <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
                        <visual><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry>
                        <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
                        <collision><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry></collision>
                    </link>""",
                ],
                "child_joints": [
                    """<joint name="left_finger_joint" type="prismatic">
                        <parent link="gripper_base"/><child link="left_finger"/>
                        <origin xyz="0.02 0 0.01" rpy="0 0 0"/>
                        <axis xyz="1 0 0"/>
                        <limit lower="-0.02" upper="0.02" effort="10" velocity="0.5"/>
                    </joint>""",
                    """<joint name="right_finger_joint" type="prismatic">
                        <parent link="gripper_base"/><child link="right_finger"/>
                        <origin xyz="-0.02 0 0.01" rpy="0 0 0"/>
                        <axis xyz="1 0 0"/>
                        <limit lower="-0.02" upper="0.02" effort="10" velocity="0.5"/>
                    </joint>""",
                ],
            },
            "tool_flange": {
                "name": "Tool Flange",
                "description": "Simple tool attachment flange",
                "link_xml": """
                    <link name="tool_flange">
                        <inertial>
                            <mass value="0.2"/>
                            <inertia ixx="0.0005" iyy="0.0005" izz="0.0005" ixy="0" ixz="0" iyz="0"/>
                        </inertial>
                        <visual>
                            <geometry><cylinder radius="0.04" length="0.02"/></geometry>
                            <material name="metal"><color rgba="0.7 0.7 0.7 1"/></material>
                        </visual>
                        <collision><geometry><cylinder radius="0.04" length="0.02"/></geometry></collision>
                    </link>
                """,
                "child_links": [],
                "child_joints": [],
            },
            "golf_club_attachment": {
                "name": "Golf Club Attachment",
                "description": "Attachment point for golf club grip",
                "link_xml": """
                    <link name="club_mount">
                        <inertial>
                            <mass value="0.1"/>
                            <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
                        </inertial>
                        <visual>
                            <geometry><cylinder radius="0.015" length="0.05"/></geometry>
                            <material name="rubber"><color rgba="0.1 0.1 0.1 1"/></material>
                        </visual>
                        <collision><geometry><cylinder radius="0.015" length="0.05"/></geometry></collision>
                    </link>
                """,
                "child_links": [],
                "child_joints": [],
            },
        }

    def get_builtin(self, key: str) -> EndEffector | None:
        """Get a built-in end effector definition."""
        if key is None:
            raise ValueError("key must be provided")
        if key not in self._builtin_definitions:
            return None

        definition = self._builtin_definitions[key]

        # Parse link XML
        link_elem = DefusedET.fromstring(definition["link_xml"].strip())

        # Parse child links
        child_links = []
        for link_xml in definition["child_links"]:
            child_links.append(DefusedET.fromstring(link_xml.strip()))

        # Parse child joints
        child_joints = []
        for joint_xml in definition["child_joints"]:
            child_joints.append(DefusedET.fromstring(joint_xml.strip()))

        return EndEffector(
            name=definition["name"],
            link_element=link_elem,
            joint_element=None,  # Will be created on attachment
            child_links=child_links,
            child_joints=child_joints,
        )

    def get_builtin_names(self) -> list[str]:
        """Get list of built-in end effector names."""
        return list(self._builtin_definitions.keys())

    def get_builtin_info(self, key: str) -> dict[str, str] | None:
        """Get info about a built-in end effector."""
        if key is None:
            raise ValueError("key must be provided")
        if key in self._builtin_definitions:
            return {
                "name": self._builtin_definitions[key]["name"],
                "description": self._builtin_definitions[key]["description"],
            }
        return None

    def extract_from_urdf(  # noqa: C901
        self,
        urdf_content: str,
        end_effector_link: str,
        source_file: Path | None = None,
    ) -> EndEffector | None:
        """Extract an end effector and its subtree from a URDF.

        Args:
            urdf_content: URDF XML content
            end_effector_link: Name of the root link of the end effector
            source_file: Source file path for reference

        Returns:
            Extracted end effector, or None if not found
        """
        if urdf_content is None:
            raise ValueError("urdf_content must be provided")
        try:
            root = DefusedET.fromstring(urdf_content)
        except ET.ParseError:
            return None

        # Find the end effector link
        ee_link = None
        for link in root.findall("link"):
            if link.get("name") == end_effector_link:
                ee_link = link
                break

        if ee_link is None:
            return None

        # Find the joint connecting to this link (as child)
        ee_joint = None
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None and child.get("link") == end_effector_link:
                ee_joint = joint
                break

        # Recursively find all child links and joints
        child_links = []
        child_joints = []

        def collect_children(parent_name: str) -> None:
            """Recursively gather child links and joints under the parent."""
            for joint in root.findall("joint"):
                parent = joint.find("parent")
                child = joint.find("child")
                if parent is not None and parent.get("link") == parent_name:
                    child_name = child.get("link") if child is not None else None
                    if child_name:
                        # Find the child link
                        for link in root.findall("link"):
                            if link.get("name") == child_name:
                                child_links.append(link)
                                child_joints.append(joint)
                                collect_children(child_name)
                                break

        collect_children(end_effector_link)

        return EndEffector(
            name=end_effector_link,
            link_element=ee_link,
            joint_element=ee_joint,
            child_links=child_links,
            child_joints=child_joints,
            source_file=source_file,
        )

    def add_to_library(self, key: str, end_effector: EndEffector) -> None:
        """Add an end effector to the library."""
        self.end_effectors[key] = end_effector

    def remove_from_library(self, key: str) -> bool:
        """Remove an end effector from the library."""
        if key is None:
            raise ValueError("key must be provided")
        if key in self.end_effectors:
            del self.end_effectors[key]
            return True
        return False
