"""
Frankenstein Editor for component composition.

Allows combining parts from multiple URDF models into a single composite model,
like building a video game character from different pieces.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.types import Joint, JointType, Link, Material, Origin

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of components that can be copied."""

    LINK = "link"
    SUBTREE = "subtree"
    JOINT = "joint"
    MATERIAL = "material"


@dataclass
class ComponentReference:
    """Reference to a component in a model."""

    model_id: str
    component_type: ComponentType
    component_name: str
    # For subtree: the root link name
    subtree_root: str | None = None

    def __str__(self) -> str:
        if self.component_type == ComponentType.SUBTREE:
            return f"{self.model_id}:{self.subtree_root}/*"
        return f"{self.model_id}:{self.component_name}"


@dataclass
class PendingOperation:
    """A pending copy/paste operation."""

    operation_type: str  # 'copy_link', 'copy_subtree', 'attach', 'rename', 'delete'
    source_ref: ComponentReference | None
    target_model_id: str | None
    parameters: dict[str, Any] = field(default_factory=dict)
    preview_links: list[Link] = field(default_factory=list)
    preview_joints: list[Joint] = field(default_factory=list)


@dataclass
class EditorState:
    """State of the Frankenstein editor for undo/redo."""

    models: dict[str, ParsedModel]
    clipboard: list[tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]]
    operation_history: list[PendingOperation]
    timestamp: float = 0.0


class FrankensteinEditor:
    """
    Editor for composing URDF models from multiple sources.

    Features:
    - Load multiple models side-by-side
    - Copy components (links, subtrees, materials) between models
    - Automatic rename handling for conflicts
    - Preview changes before applying
    - Undo/redo support
    - Export composed model

    Example:
        editor = FrankensteinEditor()

        # Load source models
        editor.load_model("humanoid", "/path/to/humanoid.urdf")
        editor.load_model("robot_arm", "/path/to/arm.urdf")

        # Create a new composite model
        editor.create_model("cyborg")

        # Copy humanoid body to cyborg
        editor.copy_subtree("humanoid", "torso")
        editor.paste_subtree("cyborg", attach_to="base_link")

        # Replace right arm with robot arm
        editor.delete_subtree("cyborg", "right_arm")
        editor.copy_subtree("robot_arm", "arm_base")
        editor.paste_subtree("cyborg", attach_to="right_shoulder", prefix="robot_")

        # Export
        urdf_string = editor.export_model("cyborg")
    """

    def __init__(self):
        """Initialize the Frankenstein editor."""
        self._models: dict[str, ParsedModel] = {}
        self._parser = URDFParser()
        self._clipboard: list[
            tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]
        ] = []
        self._undo_stack: list[EditorState] = []
        self._redo_stack: list[EditorState] = []
        self._max_history = 50
        self._rename_callbacks: list[Callable[[str, str, str], None]] = []

    # ============================================================
    # Model Loading/Creation
    # ============================================================

    def load_model(
        self,
        model_id: str,
        source: str | Path,
        read_only: bool = False,
    ) -> ParsedModel:
        """
        Load a URDF model into the editor.

        Args:
            model_id: Identifier for this model in the editor
            source: Path to URDF file or XML string
            read_only: If True, model cannot be modified

        Returns:
            The loaded ParsedModel
        """
        self._save_state()

        model = self._parser.parse(source, read_only=read_only)
        self._models[model_id] = model

        logger.info(f"Loaded model '{model_id}' with {len(model.links)} links")
        return model

    def create_model(
        self,
        model_id: str,
        name: str | None = None,
        base_link_name: str = "base_link",
    ) -> ParsedModel:
        """
        Create a new empty model.

        Args:
            model_id: Identifier for this model
            name: Robot name (defaults to model_id)
            base_link_name: Name for the base link

        Returns:
            The created ParsedModel
        """
        self._save_state()

        # Create base link with minimal inertia
        from model_generation.core.types import Inertia

        base_link = Link(
            name=base_link_name,
            inertia=Inertia(ixx=0.001, iyy=0.001, izz=0.001, mass=0.001),
        )

        model = ParsedModel(
            name=name or model_id,
            links=[base_link],
            joints=[],
            materials={},
            read_only=False,
        )

        self._models[model_id] = model
        logger.info(f"Created new model '{model_id}'")
        return model

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from the editor.

        Args:
            model_id: Model to unload

        Returns:
            True if unloaded
        """
        if model_id in self._models:
            self._save_state()
            del self._models[model_id]
            logger.info(f"Unloaded model '{model_id}'")
            return True
        return False

    def get_model(self, model_id: str) -> ParsedModel | None:
        """Get a loaded model by ID."""
        return self._models.get(model_id)

    def list_models(self) -> list[str]:
        """List all loaded model IDs."""
        return list(self._models.keys())

    def duplicate_model(self, source_id: str, new_id: str) -> ParsedModel | None:
        """
        Create a copy of an existing model.

        Args:
            source_id: Model to copy
            new_id: ID for the new copy

        Returns:
            The copied model
        """
        source = self._models.get(source_id)
        if not source:
            logger.error(f"Model '{source_id}' not found")
            return None

        self._save_state()

        new_model = source.copy()
        new_model.read_only = False
        self._models[new_id] = new_model

        logger.info(f"Duplicated model '{source_id}' as '{new_id}'")
        return new_model

    # ============================================================
    # Component Inspection
    # ============================================================

    def get_link_tree(self, model_id: str) -> dict[str, Any]:
        """
        Get the link hierarchy as a nested dict.

        Args:
            model_id: Model to inspect

        Returns:
            Dict with link names and children
        """
        model = self._models.get(model_id)
        if not model:
            return {}

        def build_tree(link_name: str) -> dict[str, Any]:
            children = model.get_children(link_name)
            return {
                "name": link_name,
                "children": [build_tree(c) for c in children],
            }

        root = model.get_root_link()
        if root:
            return build_tree(root.name)
        return {}

    def get_subtree_links(self, model_id: str, root_link: str) -> list[str]:
        """
        Get all link names in a subtree.

        Args:
            model_id: Model to inspect
            root_link: Root of the subtree

        Returns:
            List of link names in the subtree
        """
        model = self._models.get(model_id)
        if not model:
            return []
        return model.get_subtree(root_link)

    def get_connecting_joint(self, model_id: str, link_name: str) -> Joint | None:
        """
        Get the joint connecting a link to its parent.

        Args:
            model_id: Model to inspect
            link_name: Link name

        Returns:
            The connecting Joint or None
        """
        model = self._models.get(model_id)
        if not model:
            return None

        for joint in model.joints:
            if joint.child == link_name:
                return joint
        return None

    # ============================================================
    # Clipboard Operations
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
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        link = model.get_link(link_name)
        if not link:
            logger.error(f"Link '{link_name}' not found in '{model_id}'")
            return False

        links = [Link.from_dict(link.to_dict())]
        joints = []
        materials = {}

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
        joints = []
        for joint in model.joints:
            if joint.parent in subtree_names and joint.child in subtree_names:
                joints.append(Joint.from_dict(joint.to_dict()))

        # Also copy the connecting joint to the subtree root
        root_joint = self.get_connecting_joint(model_id, root_link)
        if root_joint:
            joints.insert(0, Joint.from_dict(root_joint.to_dict()))

        # Collect materials
        materials = {}
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
        if not self._clipboard:
            logger.error("Clipboard is empty")
            return []

        model = self._models.get(target_model_id)
        if not model:
            logger.error(f"Model '{target_model_id}' not found")
            return []

        if model.read_only:
            logger.error(f"Model '{target_model_id}' is read-only")
            return []

        self._save_state()

        comp_type, links, joints, materials = self._clipboard[0]
        created_links = []

        # Build name mapping for renames
        name_map = {}
        existing_links = {link.name for link in model.links}
        existing_joints = {j.name for j in model.joints}

        # Generate unique names
        for link in links:
            new_name = self._generate_unique_name(
                prefix + link.name + suffix,
                existing_links,
            )
            name_map[link.name] = new_name
            existing_links.add(new_name)

        for joint in joints:
            new_name = self._generate_unique_name(
                prefix + joint.name + suffix,
                existing_joints,
            )
            name_map[joint.name] = new_name
            existing_joints.add(new_name)

        # Copy materials (with conflict handling)
        for mat_name, mat in materials.items():
            new_mat_name = prefix + mat_name + suffix
            if new_mat_name not in model.materials:
                new_mat = Material.from_dict(mat.to_dict())
                new_mat.name = new_mat_name
                model.materials[new_mat_name] = new_mat

        # Create renamed copies of links
        for link in links:
            new_link = Link.from_dict(link.to_dict())
            new_link.name = name_map[link.name]

            # Update material reference
            if new_link.visual_material:
                new_link.visual_material.name = (
                    prefix + new_link.visual_material.name + suffix
                )

            model.links.append(new_link)
            created_links.append(new_link.name)

        # Create renamed copies of joints
        first_link = name_map.get(links[0].name) if links else None
        attachment_created = False

        for joint in joints:
            new_joint = Joint.from_dict(joint.to_dict())
            new_joint.name = name_map.get(joint.name, joint.name)

            # Update parent/child references
            if joint.parent in name_map:
                new_joint.parent = name_map[joint.parent]
            elif joint.child == links[0].name if links else None:
                # This is the root attachment joint
                if attach_to:
                    new_joint.parent = attach_to
                    new_joint.joint_type = joint_type
                    if attachment_origin:
                        new_joint.origin = attachment_origin
                    attachment_created = True
                else:
                    continue  # Skip root joint if no attachment point

            if joint.child in name_map:
                new_joint.child = name_map[joint.child]

            model.joints.append(new_joint)

        # Create attachment joint if needed
        if attach_to and first_link and not attachment_created:
            attach_joint = Joint(
                name=self._generate_unique_name(
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

    # ============================================================
    # Direct Modifications
    # ============================================================

    def delete_link(
        self,
        model_id: str,
        link_name: str,
        reparent_children: bool = True,
    ) -> bool:
        """
        Delete a link from a model.

        Args:
            model_id: Target model
            link_name: Link to delete
            reparent_children: If True, attach children to grandparent

        Returns:
            True if deleted
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        link = model.get_link(link_name)
        if not link:
            logger.error(f"Link '{link_name}' not found")
            return False

        self._save_state()

        # Get parent
        parent_name = model.get_parent(link_name)
        parent_joint = self.get_connecting_joint(model_id, link_name)

        # Get children
        children = model.get_children(link_name)

        # Reparent children if requested
        if reparent_children and parent_name:
            for child in children:
                child_joint = self.get_connecting_joint(model_id, child)
                if child_joint:
                    child_joint.parent = parent_name
                    # Adjust origin if we have parent joint info
                    if parent_joint:
                        # Combine transforms (simplified - just add positions)
                        px, py, pz = parent_joint.origin.xyz
                        cx, cy, cz = child_joint.origin.xyz
                        child_joint.origin = Origin(
                            xyz=(px + cx, py + cy, pz + cz),
                            rpy=child_joint.origin.rpy,
                        )
        elif not reparent_children:
            # Delete children recursively
            for child in children:
                self.delete_subtree(model_id, child)

        # Remove the link
        model.links = [link for link in model.links if link.name != link_name]

        # Remove connecting joint
        model.joints = [j for j in model.joints if j.child != link_name]

        logger.info(f"Deleted link '{link_name}' from '{model_id}'")
        return True

    def delete_subtree(self, model_id: str, root_link: str) -> bool:
        """
        Delete a subtree (link and all descendants).

        Args:
            model_id: Target model
            root_link: Root link of subtree to delete

        Returns:
            True if deleted
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        subtree = model.get_subtree(root_link)
        if not subtree:
            logger.error(f"Link '{root_link}' not found")
            return False

        self._save_state()

        # Remove all links in subtree
        model.links = [link for link in model.links if link.name not in subtree]

        # Remove all joints connected to subtree
        model.joints = [
            j
            for j in model.joints
            if j.parent not in subtree and j.child not in subtree
        ]

        # Also remove joint connecting subtree to parent
        model.joints = [j for j in model.joints if j.child != root_link]

        logger.info(
            f"Deleted subtree '{root_link}' ({len(subtree)} links) from '{model_id}'"
        )
        return True

    def rename_link(
        self,
        model_id: str,
        old_name: str,
        new_name: str,
    ) -> bool:
        """
        Rename a link.

        Args:
            model_id: Target model
            old_name: Current link name
            new_name: New link name

        Returns:
            True if renamed
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        # Check for conflicts
        if model.get_link(new_name):
            logger.error(f"Link '{new_name}' already exists")
            return False

        link = model.get_link(old_name)
        if not link:
            logger.error(f"Link '{old_name}' not found")
            return False

        self._save_state()

        # Rename link
        link.name = new_name

        # Update joint references
        for joint in model.joints:
            if joint.parent == old_name:
                joint.parent = new_name
            if joint.child == old_name:
                joint.child = new_name

        # Notify callbacks
        for callback in self._rename_callbacks:
            callback(model_id, old_name, new_name)

        logger.info(f"Renamed link '{old_name}' to '{new_name}'")
        return True

    def rename_joint(
        self,
        model_id: str,
        old_name: str,
        new_name: str,
    ) -> bool:
        """
        Rename a joint.

        Args:
            model_id: Target model
            old_name: Current joint name
            new_name: New joint name

        Returns:
            True if renamed
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        # Check for conflicts
        if model.get_joint(new_name):
            logger.error(f"Joint '{new_name}' already exists")
            return False

        joint = model.get_joint(old_name)
        if not joint:
            logger.error(f"Joint '{old_name}' not found")
            return False

        self._save_state()
        joint.name = new_name

        logger.info(f"Renamed joint '{old_name}' to '{new_name}'")
        return True

    def modify_joint(
        self,
        model_id: str,
        joint_name: str,
        **kwargs: Any,
    ) -> bool:
        """
        Modify joint properties.

        Args:
            model_id: Target model
            joint_name: Joint to modify
            **kwargs: Properties to update (origin, axis, limits, dynamics, joint_type)

        Returns:
            True if modified
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        joint = model.get_joint(joint_name)
        if not joint:
            logger.error(f"Joint '{joint_name}' not found")
            return False

        self._save_state()

        # Update properties
        if "origin" in kwargs:
            joint.origin = kwargs["origin"]
        if "axis" in kwargs:
            joint.axis = kwargs["axis"]
        if "limits" in kwargs:
            joint.limits = kwargs["limits"]
        if "dynamics" in kwargs:
            joint.dynamics = kwargs["dynamics"]
        if "joint_type" in kwargs:
            joint.joint_type = kwargs["joint_type"]

        logger.info(f"Modified joint '{joint_name}'")
        return True

    def attach_link(
        self,
        model_id: str,
        parent_link: str,
        child_link: str,
        joint_name: str | None = None,
        joint_type: JointType = JointType.FIXED,
        origin: Origin | None = None,
    ) -> bool:
        """
        Create a joint attaching two existing links.

        Args:
            model_id: Target model
            parent_link: Parent link name
            child_link: Child link name
            joint_name: Optional joint name
            joint_type: Type of joint
            origin: Joint origin

        Returns:
            True if attached
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        # Verify links exist
        if not model.get_link(parent_link):
            logger.error(f"Parent link '{parent_link}' not found")
            return False
        if not model.get_link(child_link):
            logger.error(f"Child link '{child_link}' not found")
            return False

        # Check child doesn't already have a parent
        if model.get_parent(child_link):
            logger.error(f"Link '{child_link}' already has a parent")
            return False

        self._save_state()

        # Generate joint name
        if not joint_name:
            joint_name = self._generate_unique_name(
                f"{parent_link}_to_{child_link}_joint",
                {j.name for j in model.joints},
            )

        joint = Joint(
            name=joint_name,
            joint_type=joint_type,
            parent=parent_link,
            child=child_link,
            origin=origin or Origin(),
        )
        model.joints.append(joint)

        logger.info(f"Attached '{child_link}' to '{parent_link}'")
        return True

    def detach_link(
        self,
        model_id: str,
        link_name: str,
    ) -> bool:
        """
        Detach a link from its parent (remove connecting joint).

        The link becomes a floating root.

        Args:
            model_id: Target model
            link_name: Link to detach

        Returns:
            True if detached
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        joint = self.get_connecting_joint(model_id, link_name)
        if not joint:
            logger.warning(f"Link '{link_name}' has no parent joint")
            return False

        self._save_state()

        model.joints = [j for j in model.joints if j.child != link_name]
        logger.info(f"Detached link '{link_name}'")
        return True

    # ============================================================
    # Batch Operations
    # ============================================================

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
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return False

        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return False

        self._save_state()

        # Build name maps
        link_map = {}
        joint_map = {}
        material_map = {}

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
        for link in model.links:
            if link.name in link_map:
                link.name = link_map[link.name]
            if link.visual_material and link.visual_material.name in material_map:
                link.visual_material.name = material_map[link.visual_material.name]

        for joint in model.joints:
            if joint.name in joint_map:
                joint.name = joint_map[joint.name]
            if joint.parent in link_map:
                joint.parent = link_map[joint.parent]
            if joint.child in link_map:
                joint.child = link_map[joint.child]

        # Rename materials in dict
        new_materials = {}
        for old_name, mat in model.materials.items():
            new_name = material_map.get(old_name, old_name)
            mat.name = new_name
            new_materials[new_name] = mat
        model.materials = new_materials

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
        # Copy subtree to clipboard
        if not self.copy_subtree(model_id, root_link):
            return []

        # Get parent for attachment
        model = self._models.get(model_id)
        if not model:
            return []

        parent = model.get_parent(root_link)
        if not parent:
            logger.error("Cannot mirror root link")
            return []

        # Default replacements for left/right
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

        # Paste with mirrored positions
        self._save_state()

        comp_type, links, joints, materials = self._clipboard[0]

        # Generate mirrored names
        def mirror_name(name: str) -> str:
            result = name
            for old, new in name_replacements.items():
                result = result.replace(old, new)
            if result == name:
                # No replacement found, add suffix
                result = name + "_mirrored"
            return result

        # Build name map
        name_map = {}
        existing_links = {link.name for link in model.links}

        for link in links:
            new_name = mirror_name(link.name)
            new_name = self._generate_unique_name(new_name, existing_links)
            name_map[link.name] = new_name
            existing_links.add(new_name)

        # Mirror positions
        axis_idx = {"x": 0, "y": 1, "z": 2}[mirror_axis]

        created_links = []

        for link in links:
            new_link = Link.from_dict(link.to_dict())
            new_link.name = name_map[link.name]

            # Mirror visual/collision origins
            if new_link.visual_origin:
                xyz = list(new_link.visual_origin.xyz)
                xyz[axis_idx] = -xyz[axis_idx]
                new_link.visual_origin = Origin(
                    xyz=(xyz[0], xyz[1], xyz[2]), rpy=new_link.visual_origin.rpy
                )

            if new_link.collision_origin:
                xyz = list(new_link.collision_origin.xyz)
                xyz[axis_idx] = -xyz[axis_idx]
                new_link.collision_origin = Origin(
                    xyz=(xyz[0], xyz[1], xyz[2]), rpy=new_link.collision_origin.rpy
                )

            model.links.append(new_link)
            created_links.append(new_link.name)

        # Copy and mirror joints
        for joint in joints:
            new_joint = Joint.from_dict(joint.to_dict())
            new_joint.name = mirror_name(joint.name)

            # Update references
            if joint.parent in name_map:
                new_joint.parent = name_map[joint.parent]
            elif joint.child == links[0].name:
                new_joint.parent = parent  # Attach to same parent

            if joint.child in name_map:
                new_joint.child = name_map[joint.child]

            # Mirror origin
            xyz = list(new_joint.origin.xyz)
            xyz[axis_idx] = -xyz[axis_idx]
            new_joint.origin = Origin(
                xyz=(xyz[0], xyz[1], xyz[2]), rpy=new_joint.origin.rpy
            )

            # Mirror axis for revolute joints
            if new_joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                axis = list(new_joint.axis)
                axis[axis_idx] = -axis[axis_idx]
                new_joint.axis = (axis[0], axis[1], axis[2])

            model.joints.append(new_joint)

        logger.info(f"Created mirrored subtree with {len(created_links)} links")
        return created_links

    # ============================================================
    # Undo/Redo
    # ============================================================

    def undo(self) -> bool:
        """
        Undo the last operation.

        Returns:
            True if undone
        """
        if not self._undo_stack:
            logger.warning("Nothing to undo")
            return False

        # Save current state to redo stack
        current_state = self._create_state()
        self._redo_stack.append(current_state)

        # Restore previous state
        state = self._undo_stack.pop()
        self._restore_state(state)

        logger.info("Undone")
        return True

    def redo(self) -> bool:
        """
        Redo the last undone operation.

        Returns:
            True if redone
        """
        if not self._redo_stack:
            logger.warning("Nothing to redo")
            return False

        # Save current state to undo stack
        current_state = self._create_state()
        self._undo_stack.append(current_state)

        # Restore redo state
        state = self._redo_stack.pop()
        self._restore_state(state)

        logger.info("Redone")
        return True

    def _save_state(self) -> None:
        """Save current state to undo stack."""
        state = self._create_state()
        self._undo_stack.append(state)

        # Clear redo stack on new operation
        self._redo_stack = []

        # Limit history size
        while len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

    def _create_state(self) -> EditorState:
        """Create a state snapshot."""
        import time

        models_copy = {}
        for model_id, model in self._models.items():
            models_copy[model_id] = model.copy()

        return EditorState(
            models=models_copy,
            clipboard=copy.deepcopy(self._clipboard),
            operation_history=[],
            timestamp=time.time(),
        )

    def _restore_state(self, state: EditorState) -> None:
        """Restore from a state snapshot."""
        self._models = state.models
        self._clipboard = state.clipboard

    # ============================================================
    # Export
    # ============================================================

    def export_model(
        self,
        model_id: str,
        output_path: Path | None = None,
        pretty_print: bool = True,
    ) -> str:
        """
        Export a model to URDF.

        Args:
            model_id: Model to export
            output_path: Optional file path to write to
            pretty_print: Format output with indentation

        Returns:
            URDF XML string
        """
        model = self._models.get(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found")

        urdf_string = model.to_urdf(pretty_print=pretty_print)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(urdf_string)
            logger.info(f"Exported model to {output_path}")

        return urdf_string

    def compare_models(
        self,
        model_id_a: str,
        model_id_b: str,
    ) -> dict[str, Any]:
        """
        Compare two models.

        Args:
            model_id_a: First model
            model_id_b: Second model

        Returns:
            Comparison results
        """
        model_a = self._models.get(model_id_a)
        model_b = self._models.get(model_id_b)

        if not model_a or not model_b:
            return {"error": "Model not found"}

        links_a = {link.name for link in model_a.links}
        links_b = {link.name for link in model_b.links}

        joints_a = {j.name for j in model_a.joints}
        joints_b = {j.name for j in model_b.joints}

        return {
            "links": {
                "only_in_a": list(links_a - links_b),
                "only_in_b": list(links_b - links_a),
                "common": list(links_a & links_b),
            },
            "joints": {
                "only_in_a": list(joints_a - joints_b),
                "only_in_b": list(joints_b - joints_a),
                "common": list(joints_a & joints_b),
            },
            "stats": {
                "model_a_links": len(links_a),
                "model_a_joints": len(joints_a),
                "model_b_links": len(links_b),
                "model_b_joints": len(joints_b),
            },
        }

    # ============================================================
    # Utility Methods
    # ============================================================

    def _generate_unique_name(
        self,
        base_name: str,
        existing_names: set[str],
    ) -> str:
        """Generate a unique name by appending a number if needed."""
        if base_name not in existing_names:
            return base_name

        # Try numbered suffixes
        counter = 1
        while True:
            candidate = f"{base_name}_{counter}"
            if candidate not in existing_names:
                return candidate
            counter += 1

    def register_rename_callback(
        self, callback: Callable[[str, str, str], None]
    ) -> None:
        """
        Register a callback for rename events.

        Callback receives (model_id, old_name, new_name).
        """
        self._rename_callbacks.append(callback)

    def get_model_statistics(self, model_id: str) -> dict[str, Any]:
        """Get statistics about a model."""
        model = self._models.get(model_id)
        if not model:
            return {"error": "Model not found"}

        total_mass = sum(link.inertia.mass for link in model.links)

        joint_types: dict[str, int] = {}
        for j in model.joints:
            jt = j.joint_type.value
            joint_types[jt] = joint_types.get(jt, 0) + 1

        return {
            "name": model.name,
            "link_count": len(model.links),
            "joint_count": len(model.joints),
            "material_count": len(model.materials),
            "total_mass": total_mass,
            "joint_types": joint_types,
            "read_only": model.read_only,
            "has_warnings": len(model.warnings) > 0,
        }
