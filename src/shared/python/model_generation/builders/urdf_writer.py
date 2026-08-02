# mypy: ignore-errors
# ruff: noqa: E501
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
URDF XML writer.

This module handles the generation of URDF XML from Link and Joint objects,
including proper formatting, material definitions, and composite joint expansion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from src.shared.python.model_generation.core.constants import (
    INTERMEDIATE_LINK_MASS,
    URDF_INDENT,
    URDF_XML_DECLARATION,
)
from src.shared.python.model_generation.core.types import (
    Geometry,
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
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

        # Validate link-joint graph structure
        self._validate_graph(links, joints)

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
        if link is None:
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
        if joint is None:
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
        if geometry is None:
            raise ValueError("geometry must be provided")
        lines: list[str] = []
        indent = self.indent * level
        indent2 = self.indent * (level + 1)

        lines.append(f"{indent}<geometry>")

        from shared.python.model_generation.core.types import GeometryType

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
            lines.append(
                f'{indent2}<cylinder radius="{geometry.dimensions[0]:.6g}" '
                f'length="{geometry.dimensions[1]:.6g}"/>'
            )
        elif geometry.geometry_type == GeometryType.MESH:
            mesh_filename = self._validate_mesh_filename(geometry.mesh_filename or "")
            scale = geometry.mesh_scale
            lines.append(
                f'{indent2}<mesh filename="{self._escape(mesh_filename)}" '
                f'scale="{scale[0]:.6g} {scale[1]:.6g} {scale[2]:.6g}"/>'
            )

        lines.append(f"{indent}</geometry>")

        return lines

    def _write_material_definition(self, material: Material, level: int) -> list[str]:
        """Generate XML for material definition."""
        if material is None:
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
        """Collect all unique materials from links.

        Detects and warns about material name collisions where two links
        define different RGBA colors under the same material name.  The
        first definition wins; subsequent conflicting definitions are
        logged at WARNING level.
        """
        if links is None:
            raise ValueError("links must be provided")
        materials: dict[str, Material] = {}

        # Add materials from links — detect color collisions
        for link in links:
            mat = link.visual_material
            if mat is None:
                continue
            if mat.name in materials:
                existing = materials[mat.name]
                if existing.color != mat.color:
                    logger.warning(
                        "Material name collision: '%s' defined with color "
                        "%s on link '%s' but already defined with color %s. "
                        "The first definition will be used in the URDF.",
                        mat.name,
                        mat.color,
                        link.name,
                        existing.color,
                    )
            else:
                materials[mat.name] = mat

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
        if links is None:
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
        queue = [link.name for link in roots]
        visited: set[str] = set()

        while queue:
            name = queue.pop(0)
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
        """Expand composite joints (gimbal, universal) to multiple revolute joints."""
        if links is None:
            raise ValueError("links must be provided")
        if not self.expand_composite_joints:
            return links, joints

        new_links: list[Link] = []
        new_joints: list[Joint] = []

        for joint in joints:
            if joint.joint_type == JointType.GIMBAL:
                # Expand to 3 revolute joints (Z-Y-X Euler)
                intermediate_links, revolute_joints = self._expand_gimbal_joint(joint)
                new_links.extend(intermediate_links)
                new_joints.extend(revolute_joints)
            elif joint.joint_type == JointType.UNIVERSAL:
                # Expand to 2 revolute joints
                intermediate_links, revolute_joints = self._expand_universal_joint(
                    joint
                )
                new_links.extend(intermediate_links)
                new_joints.extend(revolute_joints)
            else:
                new_joints.append(joint)

        return new_links, new_joints

    def _expand_gimbal_joint(self, joint: Joint) -> tuple[list[Link], list[Joint]]:
        """Expand gimbal joint to 3 revolute joints."""
        # Default axes: Z-Y-X Euler sequence
        if joint is None:
            raise ValueError("joint must be provided")
        default_axes = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        axes = self._normalize_composite_axes(
            joint.name, joint.composite_axes, default_axes
        )
        limits = self._normalize_composite_limits(
            joint.composite_limits, joint.limits, 3
        )

        intermediate_links: list[Link] = []
        revolute_joints: list[Joint] = []

        # Create intermediate links
        for i in range(2):
            link_name = f"{joint.name}_intermediate_{i + 1}"
            intermediate_links.append(
                Link(
                    name=link_name,
                    inertia=Inertia(
                        ixx=1e-6,
                        iyy=1e-6,
                        izz=1e-6,
                        mass=INTERMEDIATE_LINK_MASS,
                    ),
                )
            )

        # Create 3 revolute joints
        parents = [
            joint.parent,
            f"{joint.name}_intermediate_1",
            f"{joint.name}_intermediate_2",
        ]
        children = [
            f"{joint.name}_intermediate_1",
            f"{joint.name}_intermediate_2",
            joint.child,
        ]

        revolute_joints.extend(
            [
                Joint(
                    name=f"{joint.name}_dof{i + 1}",
                    joint_type=JointType.REVOLUTE,
                    parent=parents[i],
                    child=children[i],
                    origin=joint.origin if i == 0 else Origin(),
                    axis=axes[i] if i < len(axes) else (0, 0, 1),
                    limits=limits[i] if limits and i < len(limits) else JointLimits(),
                    dynamics=joint.dynamics,
                )
                for i in range(3)
            ]
        )

        return intermediate_links, revolute_joints

    def _expand_universal_joint(self, joint: Joint) -> tuple[list[Link], list[Joint]]:
        """Expand universal joint to 2 revolute joints."""
        # Default axes: perpendicular
        if joint is None:
            raise ValueError("joint must be provided")
        default_axes = [(1, 0, 0), (0, 1, 0)]
        axes = self._normalize_composite_axes(
            joint.name, joint.composite_axes, default_axes
        )
        limits = self._normalize_composite_limits(
            joint.composite_limits, joint.limits, 2
        )

        intermediate_links: list[Link] = []
        revolute_joints: list[Joint] = []

        # Create one intermediate link
        link_name = f"{joint.name}_intermediate"
        intermediate_links.append(
            Link(
                name=link_name,
                inertia=Inertia(
                    ixx=1e-6,
                    iyy=1e-6,
                    izz=1e-6,
                    mass=INTERMEDIATE_LINK_MASS,
                ),
            )
        )

        # Create 2 revolute joints
        for i in range(2):
            parent = joint.parent if i == 0 else link_name
            child = link_name if i == 0 else joint.child

            revolute_joints.append(
                Joint(
                    name=f"{joint.name}_dof{i + 1}",
                    joint_type=JointType.REVOLUTE,
                    parent=parent,
                    child=child,
                    origin=joint.origin if i == 0 else Origin(),
                    axis=axes[i] if i < len(axes) else (0, 0, 1),
                    limits=limits[i] if limits and i < len(limits) else JointLimits(),
                    dynamics=joint.dynamics,
                )
            )

        return intermediate_links, revolute_joints

    @staticmethod
    def _validate_graph(links: list[Link], joints: list[Joint]) -> None:
        """Validate that the link-joint graph forms a proper tree.

        A valid URDF has exactly one root link (not a child of any joint)
        and every link is reachable from the root via parent→child edges.
        Cycles or disconnected sub-trees are rejected.

        Raises:
            ValueError: If the graph is cyclic, disconnected, or has
                multiple roots.
        """
        link_names = {link.name for link in links}
        for joint in joints:
            if not isinstance(joint.parent, str) or not joint.parent.strip():
                raise ValueError(
                    f"Joint '{joint.name}' must define a non-empty parent link"
                )
            if not isinstance(joint.child, str) or not joint.child.strip():
                raise ValueError(
                    f"Joint '{joint.name}' must define a non-empty child link"
                )
            if joint.parent not in link_names:
                raise ValueError(
                    f"Joint '{joint.name}' references unknown parent link '{joint.parent}'"
                )
            if joint.child not in link_names:
                raise ValueError(
                    f"Joint '{joint.name}' references unknown child link '{joint.child}'"
                )

        children_set = {joint.child for joint in joints}
        roots = link_names - children_set

        if not roots:
            raise ValueError(
                "URDF graph has no root link (every link is a child of some "
                "joint, indicating a cycle)."
            )

        if len(roots) > 1:
            raise ValueError(
                "URDF graph must have exactly one root link; "
                f"found {len(roots)} roots: {sorted(roots)}"
            )

        root = next(iter(roots))

        # BFS reachability check from the single root
        adjacency: dict[str, list[str]] = {name: [] for name in link_names}
        for joint in joints:
            if joint.parent in adjacency:
                adjacency[joint.parent].append(joint.child)

        visited: set[str] = set()
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adjacency.get(node, []))

        unreachable = link_names - visited
        if unreachable:
            raise ValueError(
                "URDF graph contains unreachable links from root "
                f"'{root}': {sorted(unreachable)}"
            )

    @staticmethod
    def _normalize_composite_axes(
        joint_name: str,
        axes: list[tuple[float, float, float]] | None,
        defaults: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        """Return a full axis list, replacing missing entries with defaults."""
        if joint_name is None:
            raise ValueError("joint_name must be provided")
        normalized: list[tuple[float, float, float]] = []
        source = list(axes or [])

        for index, default_axis in enumerate(defaults):
            axis = source[index] if index < len(source) else default_axis
            if axis is None:
                axis = default_axis
            if len(axis) != 3:
                raise ValueError(
                    f"Joint '{joint_name}' has invalid composite axis at DOF {index + 1}"
                )
            normalized.append((float(axis[0]), float(axis[1]), float(axis[2])))

        return normalized

    @staticmethod
    def _normalize_composite_limits(
        composite_limits: list[JointLimits] | None,
        fallback_limit: JointLimits | None,
        dof_count: int,
    ) -> list[JointLimits]:
        """Return a full limit list, replacing missing entries with defaults."""
        if dof_count is None:
            raise ValueError("dof_count must be provided")
        normalized: list[JointLimits] = []
        source = list(composite_limits or [])

        for index in range(dof_count):
            limit = source[index] if index < len(source) else fallback_limit
            normalized.append(limit or JointLimits())

        return normalized

    @staticmethod
    def _validate_mesh_filename(mesh_filename: str) -> str:
        """Validate mesh references against traversal and unsupported URIs."""
        if not mesh_filename:
            raise ValueError("Mesh geometry requires a non-empty mesh filename")

        normalized = mesh_filename.replace("\\", "/")
        if normalized.startswith("package://"):
            candidate = normalized[len("package://") :]
            if not candidate or candidate.startswith("/"):
                raise ValueError(
                    f"Mesh filename '{mesh_filename}' must reference a package-relative asset"
                )
        else:
            if "://" in normalized:
                raise ValueError(
                    f"Mesh filename '{mesh_filename}' uses an unsupported URI scheme"
                )
            if normalized.startswith("/") or URDFWriter._has_windows_drive_prefix(
                normalized
            ):
                raise ValueError(
                    f"Mesh filename '{mesh_filename}' must be relative or package://"
                )
            first_segment = normalized.split("/", 1)[0]
            if ":" in first_segment:
                raise ValueError(
                    f"Mesh filename '{mesh_filename}' uses an unsupported URI scheme"
                )
            candidate = normalized

        if any(part == ".." for part in PurePosixPath(candidate).parts):
            raise ValueError(f"Mesh filename '{mesh_filename}' contains path traversal")

        return mesh_filename

    @staticmethod
    def _has_windows_drive_prefix(path: str) -> bool:
        """Return True when the path starts with a Windows drive prefix."""
        return (
            len(path) >= 3 and path[0].isalpha() and path[1] == ":" and path[2] == "/"
        )

    def _escape(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
