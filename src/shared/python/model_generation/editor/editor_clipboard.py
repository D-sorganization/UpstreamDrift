"""Clipboard operations mixin for the Frankenstein Editor.

Provides copy operations for links, subtrees, and materials,
plus clipboard inspection and clearing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from model_generation.core.contracts import precondition
from model_generation.core.types import Joint, Link, Material

from .editor_types import ComponentType

if TYPE_CHECKING:
    from model_generation.converters.urdf_parser import ParsedModel

logger = logging.getLogger(__name__)


def _is_non_empty_str(value: object) -> bool:
    """Return True if *value* is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


class ClipboardMixin:
    """Clipboard operations for the FrankensteinEditor.

    Expects the host class to provide:
    - self._models: dict[str, ParsedModel]
    - self._clipboard: list[tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]]
    - self.get_connecting_joint(model_id, link_name) -> Joint | None
    """

    # Declare expected attributes from the host class (for mypy)
    _models: dict[str, ParsedModel]
    _clipboard: list[tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]]

    def get_connecting_joint(self, model_id: str, link_name: str) -> Joint | None: ...

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda link_name: _is_non_empty_str(link_name),
        "link_name must be a non-empty string",
    )
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
        assert model_id is not None, "model_id must be provided"
        assert model_id is not None, "model_id must be provided"
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        link = model.get_link(link_name)
        if not link:
            logger.error(f"Link '{link_name}' not found in '{model_id}'")
            return False

        links = [Link.from_dict(link.to_dict())]
        joints: list[Joint] = []
        materials: dict[str, Material] = {}

        if include_joint:
            joint = self.get_connecting_joint(model_id, link_name)
            if joint:
                joints.append(Joint.from_dict(joint.to_dict()))

        # Include materials
        if link.visual_material:
            materials[link.visual_material.name] = Material.from_dict(
                link.visual_material.to_dict()
            )

        self._clipboard = [(ComponentType.LINK, links, joints, materials)]
        logger.info(f"Copied link '{link_name}' to clipboard")
        return True

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda root_link: _is_non_empty_str(root_link),
        "root_link must be a non-empty string",
    )
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
        assert model_id is not None, "model_id must be provided"
        assert model_id is not None, "model_id must be provided"
        model = self._models.get(model_id)
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
        joints: list[Joint] = []
        for joint in model.joints:
            if joint.parent in subtree_names and joint.child in subtree_names:
                joints.append(Joint.from_dict(joint.to_dict()))

        # Also copy the connecting joint to the subtree root
        root_joint = self.get_connecting_joint(model_id, root_link)
        if root_joint:
            joints.insert(0, Joint.from_dict(root_joint.to_dict()))

        # Collect materials
        materials: dict[str, Material] = {}
        for link in links:
            if link.visual_material:
                materials[link.visual_material.name] = Material.from_dict(
                    link.visual_material.to_dict()
                )

        self._clipboard = [(ComponentType.SUBTREE, links, joints, materials)]
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
        assert model_id is not None, "model_id must be provided"
        assert model_id is not None, "model_id must be provided"
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        material = model.materials.get(material_name)
        if not material:
            logger.error(f"Material '{material_name}' not found in '{model_id}'")
            return False

        materials = {material_name: Material.from_dict(material.to_dict())}
        self._clipboard = [(ComponentType.MATERIAL, [], [], materials)]
        logger.info(f"Copied material '{material_name}' to clipboard")
        return True

    def get_clipboard_info(self) -> dict[str, Any]:
        """Get information about clipboard contents."""
        if not self._clipboard:
            return {"empty": True}

        comp_type, links, joints, materials = self._clipboard[0]
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
        self._clipboard = []
