"""
Frankenstein Editor for component composition.

Allows combining parts from multiple URDF models into a single composite model,
like building a video game character from different pieces.

This is the main editor class that composes functionality from:
- editor_clipboard: Clipboard operations (copy link/subtree/material)
- editor_modifications: Modification operations (delete, rename, attach, mirror)
- editor_types: Shared data types
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.contracts import precondition
from model_generation.core.types import Joint, JointType, Link, Material, Origin

from .editor_clipboard import ClipboardMixin
from .editor_modifications import ModificationMixin
from .editor_types import (
    ComponentReference,  # noqa: F401
    ComponentType,
    EditorState,
    PendingOperation,  # noqa: F401
)

logger = logging.getLogger(__name__)


def _is_non_empty_str(value: object) -> bool:
    """Return True if *value* is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


class FrankensteinEditor(ClipboardMixin, ModificationMixin):
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
        self._clipboard: list[  # type: ignore[assignment]
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

        name_map = self._build_paste_name_map(model, links, joints, prefix, suffix)
        self._paste_materials(model, materials, prefix, suffix)
        created_links = self._paste_links_and_joints(
            model,
            links,
            joints,
            name_map,
            prefix,
            suffix,
            attach_to,
            attachment_origin,
            joint_type,
        )

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
        """Build a name mapping for pasted elements to avoid conflicts."""
        name_map: dict[str, str] = {}
        existing_links = {link.name for link in model.links}
        existing_joints = {j.name for j in model.joints}

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

        return name_map

    def _paste_materials(
        self,
        model: ParsedModel,
        materials: dict[str, Material],
        prefix: str,
        suffix: str,
    ) -> None:
        """Copy materials into the target model, handling name conflicts."""
        for mat_name, mat in materials.items():
            new_mat_name = prefix + mat_name + suffix
            if new_mat_name not in model.materials:
                new_mat = Material.from_dict(mat.to_dict())
                new_mat.name = new_mat_name
                model.materials[new_mat_name] = new_mat

    def _paste_links_and_joints(
        self,
        model: ParsedModel,
        links: list[Link],
        joints: list[Joint],
        name_map: dict[str, str],
        prefix: str,
        suffix: str,
        attach_to: str | None,
        attachment_origin: Origin | None,
        joint_type: JointType,
    ) -> list[str]:
        """Create renamed copies of links and joints in the target model."""
        created_links: list[str] = []

        for link in links:
            new_link = Link.from_dict(link.to_dict())
            new_link.name = name_map[link.name]
            if new_link.visual_material:
                new_link.visual_material.name = (
                    prefix + new_link.visual_material.name + suffix
                )
            model.links.append(new_link)
            created_links.append(new_link.name)

        first_link = name_map.get(links[0].name) if links else None
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

        return created_links

    @precondition(
        lambda target_model_id: _is_non_empty_str(target_model_id),
        "target_model_id must be a non-empty string",
    )
    @precondition(
        lambda attach_to: _is_non_empty_str(attach_to),
        "attach_to must be a non-empty string",
    )
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
            clipboard=copy.deepcopy(self._clipboard),  # type: ignore
            operation_history=[],
            timestamp=time.time(),
        )

    def _restore_state(self, state: EditorState) -> None:
        """Restore from a state snapshot."""
        self._models = state.models
        self._clipboard = state.clipboard  # type: ignore

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
