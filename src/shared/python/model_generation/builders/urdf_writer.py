"""
URDF XML writer.

This module handles the generation of URDF XML from Link and Joint objects,
including proper formatting, material definitions, and composite joint expansion.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from model_generation.core.composite_joints import (
    expand_gimbal_joint,
    expand_universal_joint,
)
from model_generation.core.constants import (
    URDF_INDENT,
    URDF_XML_DECLARATION,
)
from model_generation.core.types import (
    Geometry,
    Joint,
    JointType,
    Link,
    Material,
)

logger = logging.getLogger(__name__)


@dataclass
class URDFWriter:
    """
    Write URDF XML from model components.

    Features:
    - Pretty printing with configurable indentation
    - Material definition and reference
    - Composite joint expansion (gimbal → 3 revolute)
    - Safe XML generation (no injection)
    """

    pretty_print: bool = True
    indent: str = URDF_INDENT
    expand_composite_joints: bool = True
    include_comments: bool = False

    def write(
        self,
        robot_name: str,
        links: list[Link],
        joints: list[Joint],
        materials: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate URDF XML string.

        Args:
            robot_name: Name for robot element
            links: List of Link objects
            joints: List of Joint objects
            materials: Optional material definitions

        Returns:
            URDF XML string
        """
        if not robot_name or not robot_name.strip():
            raise ValueError("robot_name must be a non-empty string")
        if not links:
            raise ValueError("At least one link is required")

        logger.debug(
            "Writing URDF '%s' with %d links and %d joints",
            robot_name,
            len(links),
            len(joints),
        )
        lines: list[str] = []

        # XML declaration
        lines.append(URDF_XML_DECLARATION)

        # Robot element
        lines.append(f'<robot name="{self._escape(robot_name)}">')

        # Collect and write unique materials
        all_materials = self._collect_materials(links, materials or {})
        for mat in all_materials.values():
            lines.extend(self._write_material_definition(mat, 1))

        # Sort links by hierarchy (parents before children)
        sorted_links = self._sort_links_by_hierarchy(links, joints)

        # Write links
        for link in sorted_links:
            lines.extend(self._write_link(link, 1))

        # Expand composite joints and write
        expanded_links, expanded_joints = self._expand_composite_joints(links, joints)

        # Write any intermediate links from joint expansion
        for link in expanded_links:
            if link.name not in {link.name for link in sorted_links}:
                lines.extend(self._write_link(link, 1))

        # Write joints
        for joint in expanded_joints:
            lines.extend(self._write_joint(joint, 1))

        lines.append("</robot>")

        return "\n".join(lines) if self.pretty_print else "".join(lines)

    def _write_link(self, link: Link, level: int) -> list[str]:
        """Generate XML for a link."""
        if not (link is not None):
            raise ValueError("link must be provided")
        if not (link is not None):
            raise ValueError("link must be provided")
        lines: list[str] = []
        indent = self.indent * level
        indent2 = self.indent * (level + 1)
        indent3 = self.indent * (level + 2)

        lines.append(f'{indent}<link name="{self._escape(link.name)}">')

        # Inertial
        lines.append(f"{indent2}<inertial>")
        com = link.inertia.center_of_mass
        lines.append(
            f'{indent3}<origin xyz="{com[0]:.6g} {com[1]:.6g} {com[2]:.6g}" '
            f'rpy="0 0 0"/>'
        )
        lines.append(f'{indent3}<mass value="{link.inertia.mass:.6g}"/>')
        lines.append(
            f'{indent3}<inertia ixx="{link.inertia.ixx:.6g}" '
            f'ixy="{link.inertia.ixy:.6g}" ixz="{link.inertia.ixz:.6g}" '
            f'iyy="{link.inertia.iyy:.6g}" iyz="{link.inertia.iyz:.6g}" '
            f'izz="{link.inertia.izz:.6g}"/>'
        )
        lines.append(f"{indent2}</inertial>")

        # Visual
        if link.visual_geometry:
            lines.append(f"{indent2}<visual>")
            lines.append(
                f'{indent3}<origin xyz="{link.visual_origin.xyz[0]:.6g} '
                f'{link.visual_origin.xyz[1]:.6g} {link.visual_origin.xyz[2]:.6g}" '
                f'rpy="{link.visual_origin.rpy[0]:.6g} '
                f'{link.visual_origin.rpy[1]:.6g} {link.visual_origin.rpy[2]:.6g}"/>'
            )
            lines.extend(self._write_geometry(link.visual_geometry, level + 2))
            if link.visual_material:
                lines.append(
                    f'{indent3}<material name="{self._escape(link.visual_material.name)}"/>'
                )
            lines.append(f"{indent2}</visual>")

        # Collision
        if link.collision_geometry:
            lines.append(f"{indent2}<collision>")
            lines.append(
                f'{indent3}<origin xyz="{link.collision_origin.xyz[0]:.6g} '
                f'{link.collision_origin.xyz[1]:.6g} {link.collision_origin.xyz[2]:.6g}" '
                f'rpy="{link.collision_origin.rpy[0]:.6g} '
                f'{link.collision_origin.rpy[1]:.6g} {link.collision_origin.rpy[2]:.6g}"/>'
            )
            lines.extend(self._write_geometry(link.collision_geometry, level + 2))
            lines.append(f"{indent2}</collision>")

        lines.append(f"{indent}</link>")

        return lines

    def _write_joint(self, joint: Joint, level: int) -> list[str]:
        """Generate XML for a joint."""
        if not (joint is not None):
            raise ValueError("joint must be provided")
        if not (joint is not None):
            raise ValueError("joint must be provided")
        lines: list[str] = []
        indent = self.indent * level
        indent2 = self.indent * (level + 1)

        joint_type = joint.joint_type.value
        # Map composite types to revolute (they should be expanded already)
        if joint_type in ("gimbal", "universal"):
            joint_type = "revolute"

        lines.append(
            f'{indent}<joint name="{self._escape(joint.name)}" type="{joint_type}">'
        )

        # Parent and child
        lines.append(f'{indent2}<parent link="{self._escape(joint.parent)}"/>')
        lines.append(f'{indent2}<child link="{self._escape(joint.child)}"/>')

        # Origin
        lines.append(
            f'{indent2}<origin xyz="{joint.origin.xyz[0]:.6g} '
            f'{joint.origin.xyz[1]:.6g} {joint.origin.xyz[2]:.6g}" '
            f'rpy="{joint.origin.rpy[0]:.6g} '
            f'{joint.origin.rpy[1]:.6g} {joint.origin.rpy[2]:.6g}"/>'
        )

        # Axis (not for fixed joints)
        if joint.joint_type != JointType.FIXED:
            lines.append(
                f'{indent2}<axis xyz="{joint.axis[0]:.6g} '
                f'{joint.axis[1]:.6g} {joint.axis[2]:.6g}"/>'
            )

        # Limits (for revolute and prismatic)
        if joint.limits and joint.joint_type in (
            JointType.REVOLUTE,
            JointType.PRISMATIC,
        ):
            lines.append(
                f'{indent2}<limit lower="{joint.limits.lower:.6g}" '
                f'upper="{joint.limits.upper:.6g}" '
                f'effort="{joint.limits.effort:.6g}" '
                f'velocity="{joint.limits.velocity:.6g}"/>'
            )

        # Dynamics
        if joint.dynamics and joint.joint_type != JointType.FIXED:
            lines.append(
                f'{indent2}<dynamics damping="{joint.dynamics.damping:.6g}" '
                f'friction="{joint.dynamics.friction:.6g}"/>'
            )

        lines.append(f"{indent}</joint>")

        return lines

    def _write_geometry(self, geometry: Geometry, level: int) -> list[str]:
        """Generate XML for geometry."""
        if not (geometry is not None):
            raise ValueError("geometry must be provided")
        if not (geometry is not None):
            raise ValueError("geometry must be provided")
        lines: list[str] = []
        indent = self.indent * level
        indent2 = self.indent * (level + 1)

        lines.append(f"{indent}<geometry>")

        from model_generation.core.types import GeometryType

        if geometry.geometry_type == GeometryType.BOX:
            size = geometry.dimensions
            lines.append(
                f'{indent2}<box size="{size[0]:.6g} {size[1]:.6g} {size[2]:.6g}"/>'
            )
        elif geometry.geometry_type == GeometryType.CYLINDER:
            lines.append(
                f'{indent2}<cylinder radius="{geometry.dimensions[0]:.6g}" '
                f'length="{geometry.dimensions[1]:.6g}"/>'
            )
        elif geometry.geometry_type == GeometryType.SPHERE:
            lines.append(f'{indent2}<sphere radius="{geometry.dimensions[0]:.6g}"/>')
        elif geometry.geometry_type == GeometryType.CAPSULE:
            # URDF doesn't have capsule, use cylinder approximation
            logger.warning(
                "Capsule geometry approximated as cylinder"
                " (URDF has no native capsule support)"
            )
            lines.append(
                f'{indent2}<cylinder radius="{geometry.dimensions[0]:.6g}" '
                f'length="{geometry.dimensions[1]:.6g}"/>'
            )
        elif geometry.geometry_type == GeometryType.MESH:
            scale = geometry.mesh_scale
            lines.append(
                f'{indent2}<mesh filename="{self._escape(geometry.mesh_filename or "")}" '
                f'scale="{scale[0]:.6g} {scale[1]:.6g} {scale[2]:.6g}"/>'
            )

        lines.append(f"{indent}</geometry>")

        return lines

    def _write_material_definition(self, material: Material, level: int) -> list[str]:
        """Generate XML for material definition."""
        if not (material is not None):
            raise ValueError("material must be provided")
        if not (material is not None):
            raise ValueError("material must be provided")
        lines: list[str] = []
        indent = self.indent * level
        indent2 = self.indent * (level + 1)

        lines.append(f'{indent}<material name="{self._escape(material.name)}">')
        rgba = material.color
        lines.append(
            f'{indent2}<color rgba="{rgba[0]:.4g} {rgba[1]:.4g} '
            f'{rgba[2]:.4g} {rgba[3]:.4g}"/>'
        )
        if material.texture:
            lines.append(
                f'{indent2}<texture filename="{self._escape(material.texture)}"/>'
            )
        lines.append(f"{indent}</material>")

        return lines

    def _collect_materials(
        self, links: list[Link], extra_materials: dict[str, Any]
    ) -> dict[str, Material]:
        """Collect all unique materials from links."""
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        materials: dict[str, Material] = {}

        # Add materials from links
        for link in links:
            if link.visual_material and link.visual_material.name not in materials:
                materials[link.visual_material.name] = link.visual_material

        # Add extra materials
        for name, mat_data in extra_materials.items():
            if name not in materials:
                if isinstance(mat_data, Material):
                    materials[name] = mat_data
                elif isinstance(mat_data, dict):
                    materials[name] = Material.from_dict(mat_data)

        return materials

    def _sort_links_by_hierarchy(
        self, links: list[Link], joints: list[Joint]
    ) -> list[Link]:
        """Sort links so parents come before children."""
        # Build parent map
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        parent_map: dict[str, str | None] = {}
        for joint in joints:
            parent_map[joint.child] = joint.parent

        # Find root(s)
        all_children = set(parent_map.keys())
        roots = [link for link in links if link.name not in all_children]

        # BFS to order links
        ordered: list[Link] = []
        link_by_name = {link.name: link for link in links}
        queue: deque[str] = deque(link.name for link in roots)
        visited: set[str] = set()

        while queue:
            name = queue.popleft()
            if name in visited or name not in link_by_name:
                continue
            visited.add(name)
            ordered.append(link_by_name[name])

            # Add children
            for joint in joints:
                if joint.parent == name and joint.child not in visited:
                    queue.append(joint.child)

        # Add any unvisited links (shouldn't happen with valid hierarchy)
        for link in links:
            if link not in ordered:
                ordered.append(link)

        return ordered

    def _expand_composite_joints(
        self, links: list[Link], joints: list[Joint]
    ) -> tuple[list[Link], list[Joint]]:
        """Expand composite joints (gimbal, universal) to multiple revolute joints.

        Delegates to the shared utilities in
        ``model_generation.core.composite_joints``.
        """
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        if not self.expand_composite_joints:
            return links, joints

        new_links: list[Link] = []
        new_joints: list[Joint] = []

        for joint in joints:
            if joint.joint_type == JointType.GIMBAL:
                intermediate_links, revolute_joints = expand_gimbal_joint(joint)
                new_links.extend(intermediate_links)
                new_joints.extend(revolute_joints)
            elif joint.joint_type == JointType.UNIVERSAL:
                intermediate_links, revolute_joints = expand_universal_joint(joint)
                new_links.extend(intermediate_links)
                new_joints.extend(revolute_joints)
            else:
                new_joints.append(joint)

        return new_links, new_joints

    def _escape(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
