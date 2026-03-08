"""Modification operations mixin for the Frankenstein Editor.

Provides link/joint deletion, renaming, modification, attachment/detachment,
batch prefix application, and subtree mirroring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from model_generation.core.contracts import precondition
from model_generation.core.types import Joint, JointType, Link, Origin
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from model_generation.converters.urdf_parser import ParsedModel
    from model_generation.core.types import Material
    from model_generation.editor.editor_types import ComponentType

logger = logging.getLogger(__name__)


def _is_non_empty_str(value: object) -> bool:
    """Return True if *value* is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


class ModificationMixin:
    """Modification operations for the FrankensteinEditor.

    Expects the host class to provide:
    - self._models: dict[str, ParsedModel]
    - self._rename_callbacks: list[Callable]
    - self._clipboard: list[...]
    - self._save_state() -> None
    - self.get_connecting_joint(model_id, link_name) -> Joint | None
    - self.copy_subtree(model_id, root_link) -> bool
    - self._generate_unique_name(base_name, existing_names) -> str
    """

    # Declare expected attributes from the host class (for mypy)
    _models: dict[str, ParsedModel]
    _rename_callbacks: list[Callable[[str, str, str], None]]
    _clipboard: list[tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]]

    def _save_state(self) -> None: ...

    def get_connecting_joint(self, model_id: str, link_name: str) -> Joint | None: ...

    def copy_subtree(self, model_id: str, root_link: str) -> bool: ...  # type: ignore[empty-body]

    def _generate_unique_name(  # type: ignore
        self, base_name: str, existing_names: set[str]
    ) -> str: ...  # type: ignore[empty-body]

    def _get_writable_model(self, model_id: str) -> ParsedModel | None:
        """Look up a model by *model_id* and verify it is writable.

        Returns the model if it exists and is not read-only, otherwise logs
        an error and returns ``None``.
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

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda link_name: _is_non_empty_str(link_name),
        "link_name must be a non-empty string",
    )
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
                        # Compose SE(3) transforms properly (not just position addition)
                        p_rot = Rotation.from_euler("xyz", parent_joint.origin.rpy)
                        p_pos = np.array(parent_joint.origin.xyz)
                        c_pos = np.array(child_joint.origin.xyz)
                        c_rpy = np.array(child_joint.origin.rpy)

                        # Transform child position through parent frame
                        new_pos = p_pos + p_rot.apply(c_pos)

                        # Compose rotations
                        c_rot = Rotation.from_euler("xyz", c_rpy)
                        new_rpy = (p_rot * c_rot).as_euler("xyz")

                        child_joint.origin = Origin(
                            xyz=tuple(new_pos.tolist()),
                            rpy=tuple(new_rpy.tolist()),
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

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda root_link: _is_non_empty_str(root_link),
        "root_link must be a non-empty string",
    )
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

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda old_name: _is_non_empty_str(old_name),
        "old_name must be a non-empty string",
    )
    @precondition(
        lambda new_name: _is_non_empty_str(new_name),
        "new_name must be a non-empty string",
    )
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
        if old_name == new_name:
            return True  # No-op

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

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda old_name: _is_non_empty_str(old_name),
        "old_name must be a non-empty string",
    )
    @precondition(
        lambda new_name: _is_non_empty_str(new_name),
        "new_name must be a non-empty string",
    )
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
        if old_name == new_name:
            return True  # No-op

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

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda joint_name: _is_non_empty_str(joint_name),
        "joint_name must be a non-empty string",
    )
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

        # Type-checked property updates
        _ALLOWED_KEYS = {"origin", "axis", "limits", "dynamics", "joint_type"}
        unknown_keys = set(kwargs) - _ALLOWED_KEYS
        if unknown_keys:
            logger.warning("Ignoring unknown joint properties: %s", unknown_keys)

        if "origin" in kwargs:
            if not isinstance(kwargs["origin"], Origin):
                logger.error("'origin' must be an Origin instance")
                return False
            joint.origin = kwargs["origin"]
        if "axis" in kwargs:
            axis = kwargs["axis"]
            if not isinstance(axis, tuple) or len(axis) != 3:
                logger.error("'axis' must be a 3-tuple of floats")
                return False
            joint.axis = axis
        if "limits" in kwargs:
            from model_generation.core.types import JointLimits

            if not isinstance(kwargs["limits"], JointLimits):
                logger.error("'limits' must be a JointLimits instance")
                return False
            joint.limits = kwargs["limits"]
        if "dynamics" in kwargs:
            from model_generation.core.types import JointDynamics

            if not isinstance(kwargs["dynamics"], JointDynamics):
                logger.error("'dynamics' must be a JointDynamics instance")
                return False
            joint.dynamics = kwargs["dynamics"]
        if "joint_type" in kwargs:
            if not isinstance(kwargs["joint_type"], JointType):
                logger.error("'joint_type' must be a JointType enum value")
                return False
            joint.joint_type = kwargs["joint_type"]

        logger.info(f"Modified joint '{joint_name}'")
        return True

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda parent_link: _is_non_empty_str(parent_link),
        "parent_link must be a non-empty string",
    )
    @precondition(
        lambda child_link: _is_non_empty_str(child_link),
        "child_link must be a non-empty string",
    )
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
        model = self._get_writable_model(model_id)
        if not model:
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

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda link_name: _is_non_empty_str(link_name),
        "link_name must be a non-empty string",
    )
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
        model = self._get_writable_model(model_id)
        if not model:
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

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda prefix: _is_non_empty_str(prefix),
        "prefix must be a non-empty string",
    )
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
        model = self._get_writable_model(model_id)
        if not model:
            return False

        self._save_state()

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

    @staticmethod
    def _mirror_links(
        links: list[Any],
        name_map: dict[str, str],
        axis_idx: int,
        model: Any,
    ) -> list[str]:
        """Create mirrored copies of links and add them to the model.

        Returns:
            List of created link names.
        """
        created_links: list[str] = []
        for link in links:
            new_link = Link.from_dict(link.to_dict())
            new_link.name = name_map[link.name]

            if new_link.visual_origin:
                xyz = list(new_link.visual_origin.xyz)
                xyz[axis_idx] = -xyz[axis_idx]
                new_link.visual_origin = Origin(
                    xyz=tuple(xyz),
                    rpy=new_link.visual_origin.rpy,  # type: ignore[arg-type]
                )

            if new_link.collision_origin:
                xyz = list(new_link.collision_origin.xyz)
                xyz[axis_idx] = -xyz[axis_idx]
                new_link.collision_origin = Origin(
                    xyz=tuple(xyz),
                    rpy=new_link.collision_origin.rpy,  # type: ignore[arg-type]
                )

            model.links.append(new_link)
            created_links.append(new_link.name)

        return created_links

    @staticmethod
    def _mirror_joints(
        joints: list[Any],
        links: list[Any],
        name_map: dict[str, str],
        axis_idx: int,
        parent: str,
        mirror_name_fn: Any,
        model: Any,
    ) -> None:
        """Create mirrored copies of joints and add them to the model."""
        for joint in joints:
            new_joint = Joint.from_dict(joint.to_dict())
            new_joint.name = mirror_name_fn(joint.name)

            if joint.parent in name_map:
                new_joint.parent = name_map[joint.parent]
            elif joint.child == links[0].name:
                new_joint.parent = parent

            if joint.child in name_map:
                new_joint.child = name_map[joint.child]

            xyz = list(new_joint.origin.xyz)
            xyz[axis_idx] = -xyz[axis_idx]
            new_joint.origin = Origin(xyz=tuple(xyz), rpy=new_joint.origin.rpy)  # type: ignore[arg-type]

            if new_joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                axis = list(new_joint.axis)
                axis[axis_idx] = -axis[axis_idx]
                new_joint.axis = (axis[0], axis[1], axis[2])  # type: ignore[misc]

            model.joints.append(new_joint)

    @precondition(
        lambda model_id: _is_non_empty_str(model_id),
        "model_id must be a non-empty string",
    )
    @precondition(
        lambda root_link: _is_non_empty_str(root_link),
        "root_link must be a non-empty string",
    )
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
        _VALID_AXES = {"x", "y", "z"}
        if mirror_axis not in _VALID_AXES:
            raise ValueError(
                f"mirror_axis must be one of {_VALID_AXES}, got '{mirror_axis}'"
            )

        if not self.copy_subtree(model_id, root_link):
            return []

        model = self._models.get(model_id)
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

        self._save_state()

        comp_type, links, joints, materials = self._clipboard[0]

        def mirror_name(name: str) -> str:
            result = name
            for old, new in name_replacements.items():
                result = result.replace(old, new)
            if result == name:
                result = name + "_mirrored"
            return result

        name_map: dict[str, str] = {}
        existing_links = {link.name for link in model.links}
        for link in links:
            new_name = mirror_name(link.name)
            new_name = self._generate_unique_name(new_name, existing_links)
            name_map[link.name] = new_name
            existing_links.add(new_name)

        axis_idx = {"x": 0, "y": 1, "z": 2}[mirror_axis]

        created_links = self._mirror_links(links, name_map, axis_idx, model)
        self._mirror_joints(
            joints, links, name_map, axis_idx, parent, mirror_name, model
        )

        logger.info(f"Created mirrored subtree with {len(created_links)} links")
        return created_links
