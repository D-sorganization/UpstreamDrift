"""
Frankenstein Editor for component composition.

Allows combining parts from multiple URDF models into a single composite model,
like building a video game character from different pieces.

Decomposed via SRP into:
- frankenstein_types.py: Shared data types and enums
- frankenstein_clipboard.py: ClipboardMixin (copy/paste operations)
- frankenstein_history.py: HistoryMixin (undo/redo state management)
- frankenstein_transforms.py: TransformMixin (prefix, mirror batch ops)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.types import Joint, JointType, Link, Material, Origin

from .frankenstein_clipboard import ClipboardMixin
from .frankenstein_history import HistoryMixin
from .frankenstein_transforms import TransformMixin
from .frankenstein_types import (
    ComponentReference,
    ComponentType,
    EditorState,
    PendingOperation,
)

logger = logging.getLogger(__name__)

# Re-export types for backwards compatibility
__all__ = [
    "ComponentReference",
    "ComponentType",
    "EditorState",
    "FrankensteinEditor",
    "PendingOperation",
]


class FrankensteinEditor(ClipboardMixin, HistoryMixin, TransformMixin):
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

    def __init__(self) -> None:
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
            """Recursively build a nested dict representing the link hierarchy."""
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
    # Shared validation (DRY)
    # ============================================================

    def _get_writable_model(self, model_id: str) -> ParsedModel | None:
        """Retrieve a model and verify it is writable.

        DRY helper: eliminates the repeated pattern of looking up a model,
        logging an error if not found, and checking read_only.

        Returns:
            The model if found and writable, otherwise None.
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"Model '{model_id}' not found")
            return None
        if model.read_only:
            logger.error(f"Model '{model_id}' is read-only")
            return None
        return model

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
        model = self._get_writable_model(model_id)
        if not model:
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
        model = self._get_writable_model(model_id)
        if not model:
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
        model = self._get_writable_model(model_id)
        if not model:
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
        model = self._get_writable_model(model_id)
        if not model:
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
        model = self._get_writable_model(model_id)
        if not model:
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
    # Export & Comparison
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

    @staticmethod
    def _generate_unique_name(
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
