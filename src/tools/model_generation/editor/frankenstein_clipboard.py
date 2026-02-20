"""
Clipboard operations mixin for the Frankenstein Editor.

Extracted from FrankensteinEditor to respect SRP:
clipboard copy/paste logic is independent of model management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

from model_generation.core.types import Joint, JointType, Link, Material, Origin

from .frankenstein_types import ComponentType

if TYPE_CHECKING:
    from model_generation.converters.urdf_parser import ParsedModel

    class ClipboardProtocol(Protocol):
        _models: dict[str, ParsedModel]
        _clipboard: list[
            tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]
        ]

        def _save_state(self) -> None: ...
        def get_connecting_joint(
            self, model_id: str, link_name: str
        ) -> Joint | None: ...
        def _generate_unique_name(
            self, base_name: str, existing_names: set[str]
        ) -> str: ...


logger = logging.getLogger(__name__)


class ClipboardMixin:
    """Clipboard copy/paste operations for the Frankenstein Editor.

    Requires host class to provide:
        _models: dict[str, ParsedModel]
        _clipboard: list[tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]]
        _save_state() -> None
        get_connecting_joint(model_id, link_name) -> Joint | None
        _generate_unique_name(base_name, existing_names) -> str
    """

    # ============================================================
    # Copy Operations
    # ============================================================

    def copy_link(
        self,
        model_id: str,
        link_name: str,
        include_joint: bool = True,
    ) -> bool:
        """
        Copy a single link to clipboard.

        Args:
            model_id: Source model
            link_name: Link to copy
            include_joint: Include the connecting joint

        Returns:
            True if copied
        """
        host = cast("ClipboardProtocol", self)
        model = host._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        link = model.get_link(link_name)
        if not link:
            logger.error(f"Link '{link_name}' not found in '{model_id}'")
            return False

        links = [Link.from_dict(link.to_dict())]
        joints = []
        materials: dict[str, Material] = {}

        if include_joint:
            joint = host.get_connecting_joint(model_id, link_name)
            if joint:
                joints.append(Joint.from_dict(joint.to_dict()))

        # Include materials
        if link.visual_material:
            materials[link.visual_material.name] = Material.from_dict(
                link.visual_material.to_dict()
            )

        host._clipboard = [(ComponentType.LINK, links, joints, materials)]
        logger.info(f"Copied link '{link_name}' to clipboard")
        return True

    def copy_subtree(
        self,
        model_id: str,
        root_link: str,
    ) -> bool:
        """
        Copy a subtree (link and all descendants) to clipboard.

        Args:
            model_id: Source model
            root_link: Root link of subtree

        Returns:
            True if copied
        """
        host = cast("ClipboardProtocol", self)
        model = host._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        subtree_names = model.get_subtree(root_link)
        if not subtree_names:
            logger.error(f"Link '{root_link}' not found in '{model_id}'")
            return False

        # Copy all links in subtree
        links = []
        for name in subtree_names:
            link = model.get_link(name)
            if link:
                links.append(Link.from_dict(link.to_dict()))

        # Copy all joints within subtree
        joints = []
        for joint in model.joints:
            if joint.parent in subtree_names and joint.child in subtree_names:
                joints.append(Joint.from_dict(joint.to_dict()))

        # Also copy the connecting joint to the subtree root
        root_joint = host.get_connecting_joint(model_id, root_link)
        if root_joint:
            joints.insert(0, Joint.from_dict(root_joint.to_dict()))

        # Collect materials
        materials: dict[str, Material] = {}
        for link in links:
            if link.visual_material:
                materials[link.visual_material.name] = Material.from_dict(
                    link.visual_material.to_dict()
                )

        host._clipboard = [(ComponentType.SUBTREE, links, joints, materials)]
        logger.info(
            f"Copied subtree '{root_link}' ({len(links)} links, {len(joints)} joints) to clipboard"
        )
        return True

    def copy_material(self, model_id: str, material_name: str) -> bool:
        """
        Copy a material definition to clipboard.

        Args:
            model_id: Source model
            material_name: Material to copy

        Returns:
            True if copied
        """
        host = cast("ClipboardProtocol", self)
        model = host._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        material = model.materials.get(material_name)
        if not material:
            logger.error(f"Material '{material_name}' not found in '{model_id}'")
            return False

        materials = {material_name: Material.from_dict(material.to_dict())}
        host._clipboard = [(ComponentType.MATERIAL, [], [], materials)]
        logger.info(f"Copied material '{material_name}' to clipboard")
        return True

    def get_clipboard_info(self) -> dict[str, Any]:
        """Get information about clipboard contents."""
        host = cast("ClipboardProtocol", self)
        if not host._clipboard:
            return {"empty": True}

        comp_type, links, joints, materials = host._clipboard[0]
        return {
            "empty": False,
            "type": comp_type.value,
            "link_count": len(links),
            "joint_count": len(joints),
            "material_count": len(materials),
            "link_names": [link.name for link in links],
        }

    def clear_clipboard(self) -> None:
        """Clear the clipboard."""
        host = cast("ClipboardProtocol", self)
        host._clipboard = []

    # ============================================================
    # Paste Operations
    # ============================================================

    def paste(
        self,
        target_model_id: str,
        attach_to: str | None = None,
        attachment_origin: Origin | None = None,
        prefix: str = "",
        suffix: str = "",
        joint_type: JointType = JointType.FIXED,
    ) -> list[str]:
        """
        Paste clipboard contents to a model.

        Args:
            target_model_id: Target model
            attach_to: Link to attach to (None for root)
            attachment_origin: Origin for attachment joint
            prefix: Prefix to add to all names
            suffix: Suffix to add to all names
            joint_type: Type for the attachment joint

        Returns:
            List of created link names
        """
        host = cast("ClipboardProtocol", self)
        if not host._clipboard:
            logger.error("Clipboard is empty")
            return []

        model = host._models.get(target_model_id)
        if not model:
            logger.error(f"Model '{target_model_id}' not found")
            return []

        if model.read_only:
            logger.error(f"Model '{target_model_id}' is read-only")
            return []

        host._save_state()

        comp_type, links, joints, materials = host._clipboard[0]

        name_map = self._build_paste_name_map(model, links, joints, prefix, suffix)
        self._paste_materials(model, materials, prefix, suffix)
        created_links = self._paste_links(model, links, name_map, prefix, suffix)

        first_link = name_map.get(links[0].name) if links else None
        attachment_created = self._paste_joints(
            model, links, joints, name_map, attach_to, attachment_origin, joint_type
        )

        if attach_to and first_link and not attachment_created:
            attach_joint = Joint(
                name=host._generate_unique_name(
                    f"{attach_to}_to_{first_link}_joint",
                    {j.name for j in model.joints},
                ),
                joint_type=joint_type,
                parent=attach_to,
                child=first_link,
                origin=attachment_origin or Origin(),
            )
            model.joints.append(attach_joint)

        logger.info(f"Pasted {len(created_links)} links to '{target_model_id}'")
        return created_links

    def _build_paste_name_map(
        self,
        model: ParsedModel,
        links: list[Link],
        joints: list[Joint],
        prefix: str,
        suffix: str,
    ) -> dict[str, str]:
        host = cast("ClipboardProtocol", self)
        name_map: dict[str, str] = {}
        existing_links = {link.name for link in model.links}
        existing_joints = {j.name for j in model.joints}

        for link in links:
            new_name = host._generate_unique_name(
                prefix + link.name + suffix, existing_links
            )
            name_map[link.name] = new_name
            existing_links.add(new_name)

        for joint in joints:
            new_name = host._generate_unique_name(
                prefix + joint.name + suffix, existing_joints
            )
            name_map[joint.name] = new_name
            existing_joints.add(new_name)

        return name_map

    @staticmethod
    def _paste_materials(
        model: ParsedModel,
        materials: dict[str, Material],
        prefix: str,
        suffix: str,
    ) -> None:
        for mat_name, mat in materials.items():
            new_mat_name = prefix + mat_name + suffix
            if new_mat_name not in model.materials:
                new_mat = Material.from_dict(mat.to_dict())
                new_mat.name = new_mat_name
                model.materials[new_mat_name] = new_mat

    @staticmethod
    def _paste_links(
        model: ParsedModel,
        links: list[Link],
        name_map: dict[str, str],
        prefix: str,
        suffix: str,
    ) -> list[str]:
        created_links = []
        for link in links:
            new_link = Link.from_dict(link.to_dict())
            new_link.name = name_map[link.name]
            if new_link.visual_material:
                new_link.visual_material.name = (
                    prefix + new_link.visual_material.name + suffix
                )
            model.links.append(new_link)
            created_links.append(new_link.name)
        return created_links

    @staticmethod
    def _paste_joints(
        model: ParsedModel,
        links: list[Link],
        joints: list[Joint],
        name_map: dict[str, str],
        attach_to: str | None,
        attachment_origin: Origin | None,
        joint_type: JointType,
    ) -> bool:
        attachment_created = False
        for joint in joints:
            new_joint = Joint.from_dict(joint.to_dict())
            new_joint.name = name_map.get(joint.name, joint.name)

            if joint.parent in name_map:
                new_joint.parent = name_map[joint.parent]
            elif joint.child == links[0].name if links else None:
                if attach_to:
                    new_joint.parent = attach_to
                    new_joint.joint_type = joint_type
                    if attachment_origin:
                        new_joint.origin = attachment_origin
                    attachment_created = True
                else:
                    continue

            if joint.child in name_map:
                new_joint.child = name_map[joint.child]

            model.joints.append(new_joint)
        return attachment_created

    def paste_subtree(
        self,
        target_model_id: str,
        attach_to: str,
        attachment_origin: Origin | None = None,
        prefix: str = "",
        suffix: str = "",
        joint_type: JointType = JointType.FIXED,
    ) -> list[str]:
        """
        Convenience method for pasting subtree with attachment.

        Same as paste() but requires attach_to parameter.
        """
        return self.paste(
            target_model_id,
            attach_to=attach_to,
            attachment_origin=attachment_origin,
            prefix=prefix,
            suffix=suffix,
            joint_type=joint_type,
        )
