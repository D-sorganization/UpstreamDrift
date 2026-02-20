"""
Batch transform operations mixin for the Frankenstein Editor.

Extracted from FrankensteinEditor to respect SRP:
prefix application and mirroring are batch transforms orthogonal to
the core copy/paste and model management logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, cast

from model_generation.core.types import Joint, JointType, Link, Material, Origin

if TYPE_CHECKING:
    from model_generation.converters.urdf_parser import ParsedModel

    from .frankenstein_types import ComponentType

    class TransformProtocol(Protocol):
        _models: dict[str, ParsedModel]
        _clipboard: list[
            tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]
        ]

        def _save_state(self) -> None: ...
        def _generate_unique_name(
            self, base_name: str, existing_names: set[str]
        ) -> str: ...
        def copy_subtree(self, model_id: str, root_link: str) -> bool: ...

logger = logging.getLogger(__name__)


def _mirror_name(name: str, replacements: dict[str, str]) -> str:
    """Apply name replacements for mirroring (e.g., left->right)."""
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    if result == name:
        result = name + "_mirrored"
    return result


def _flip_origin(origin: Origin, axis_idx: int) -> Origin:
    """Flip an origin across the given axis."""
    xyz = list(origin.xyz)
    xyz[axis_idx] = -xyz[axis_idx]
    return Origin(xyz=(xyz[0], xyz[1], xyz[2]), rpy=origin.rpy)


class TransformMixin:
    """Batch transform operations for the Frankenstein Editor.

    Requires host class to provide:
        _models: dict[str, ParsedModel]
        _clipboard: list
        _save_state() -> None
        _generate_unique_name(base_name, existing_names) -> str
        copy_subtree(model_id, root_link) -> bool
    """

    def apply_prefix(
        self,
        model_id: str,
        prefix: str,
        include_links: bool = True,
        include_joints: bool = True,
        include_materials: bool = True,
    ) -> bool:
        """
        Add a prefix to all names in a model.

        Args:
            model_id: Target model
            prefix: Prefix to add
            include_links: Rename links
            include_joints: Rename joints
            include_materials: Rename materials

        Returns:
            True if applied
        """
        host = cast("TransformProtocol", self)
        model = host._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        host._save_state()

        # Build name maps
        link_map: dict[str, str] = {}
        joint_map: dict[str, str] = {}
        material_map: dict[str, str] = {}

        if include_links:
            for link in model.links:
                link_map[link.name] = prefix + link.name

        if include_joints:
            for joint in model.joints:
                joint_map[joint.name] = prefix + joint.name

        if include_materials:
            for mat_name in model.materials:
                material_map[mat_name] = prefix + mat_name

        # Apply renames
        _apply_link_renames(model, link_map, material_map)
        _apply_joint_renames(model, joint_map, link_map)
        _apply_material_renames(model, material_map)

        logger.info(f"Applied prefix '{prefix}' to model '{model_id}'")
        return True

    def mirror_subtree(
        self,
        model_id: str,
        root_link: str,
        mirror_axis: str = "y",
        name_replacements: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Create a mirrored copy of a subtree.

        Useful for creating symmetric limbs (left/right).

        Args:
            model_id: Target model
            root_link: Root of subtree to mirror
            mirror_axis: Axis to mirror across ('x', 'y', or 'z')
            name_replacements: Name substitutions (e.g., {"left": "right"})

        Returns:
            List of created link names
        """
        host = cast("TransformProtocol", self)
        if not host.copy_subtree(model_id, root_link):
            return []

        model = host._models.get(model_id)
        if not model:
            return []

        parent = model.get_parent(root_link)
        if not parent:
            logger.error("Cannot mirror root link")
            return []

        if name_replacements is None:
            name_replacements = {
                "left": "right",
                "right": "left",
                "Left": "Right",
                "Right": "Left",
                "_l_": "_r_",
                "_r_": "_l_",
                "_L_": "_R_",
                "_R_": "_L_",
            }

        host._save_state()

        _comp_type, links, joints, _materials = host._clipboard[0]
        axis_idx = {"x": 0, "y": 1, "z": 2}[mirror_axis]

        name_map = self._build_mirror_name_map(model, links, name_replacements)
        created_links = self._create_mirrored_links(model, links, name_map, axis_idx)
        self._create_mirrored_joints(
            model, links, joints, name_map, parent, name_replacements, axis_idx
        )

        logger.info(f"Created mirrored subtree with {len(created_links)} links")
        return created_links

    def _build_mirror_name_map(
        self,
        model: ParsedModel,
        links: list[Link],
        name_replacements: dict[str, str],
    ) -> dict[str, str]:
        host = cast("TransformProtocol", self)
        name_map: dict[str, str] = {}
        existing_links = {link.name for link in model.links}

        for link in links:
            new_name = _mirror_name(link.name, name_replacements)
            new_name = host._generate_unique_name(new_name, existing_links)
            name_map[link.name] = new_name
            existing_links.add(new_name)

        return name_map

    @staticmethod
    def _create_mirrored_links(
        model: ParsedModel,
        links: list[Link],
        name_map: dict[str, str],
        axis_idx: int,
    ) -> list[str]:
        created_links: list[str] = []

        for link in links:
            new_link = Link.from_dict(link.to_dict())
            new_link.name = name_map[link.name]

            if new_link.visual_origin:
                new_link.visual_origin = _flip_origin(new_link.visual_origin, axis_idx)

            if new_link.collision_origin:
                new_link.collision_origin = _flip_origin(
                    new_link.collision_origin, axis_idx
                )

            model.links.append(new_link)
            created_links.append(new_link.name)

        return created_links

    @staticmethod
    def _create_mirrored_joints(
        model: ParsedModel,
        links: list[Link],
        joints: list[Joint],
        name_map: dict[str, str],
        parent: str,
        name_replacements: dict[str, str],
        axis_idx: int,
    ) -> None:
        for joint in joints:
            new_joint = Joint.from_dict(joint.to_dict())
            new_joint.name = _mirror_name(joint.name, name_replacements)

            if joint.parent in name_map:
                new_joint.parent = name_map[joint.parent]
            elif joint.child == links[0].name:
                new_joint.parent = parent

            if joint.child in name_map:
                new_joint.child = name_map[joint.child]

            new_joint.origin = _flip_origin(new_joint.origin, axis_idx)

            if new_joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                axis = list(new_joint.axis)
                axis[axis_idx] = -axis[axis_idx]
                new_joint.axis = (axis[0], axis[1], axis[2])

            model.joints.append(new_joint)


def _apply_link_renames(
    model: ParsedModel,
    link_map: dict[str, str],
    material_map: dict[str, str],
) -> None:
    """Apply link and material name renames to model links."""
    for link in model.links:
        if link.name in link_map:
            link.name = link_map[link.name]
        if link.visual_material and link.visual_material.name in material_map:
            link.visual_material.name = material_map[link.visual_material.name]


def _apply_joint_renames(
    model: ParsedModel,
    joint_map: dict[str, str],
    link_map: dict[str, str],
) -> None:
    """Apply joint and parent/child name renames to model joints."""
    for joint in model.joints:
        if joint.name in joint_map:
            joint.name = joint_map[joint.name]
        if joint.parent in link_map:
            joint.parent = link_map[joint.parent]
        if joint.child in link_map:
            joint.child = link_map[joint.child]


def _apply_material_renames(
    model: ParsedModel,
    material_map: dict[str, str],
) -> None:
    """Rename materials in the model's material dictionary."""
    new_materials: dict[str, Material] = {}
    for old_name, mat in model.materials.items():
        new_name = material_map.get(old_name, old_name)
        mat.name = new_name
        new_materials[new_name] = mat
    model.materials = new_materials
