"""
Tests for Design by Contract (DbC) decorators on FrankensteinEditor,
ClipboardMixin, HumanoidURDFGenerator, and BaseURDFBuilder.

These tests verify that @precondition decorators enforce input validation
and raise PreconditionError / ContractViolationError for invalid inputs,
while valid inputs still work correctly.
"""

from __future__ import annotations

import os

import pytest

# Ensure contracts are enforced during testing
os.environ["DBC_LEVEL"] = "enforce"

from contracts import ContractViolationError  # noqa: E402

# ---------------------------------------------------------------------------
# Sample URDF fixtures
# ---------------------------------------------------------------------------

SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="simple_robot">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="arm_link">
        <inertial>
            <mass value="0.5"/>
            <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="hand_link">
        <inertial>
            <mass value="0.2"/>
            <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <joint name="base_to_arm" type="revolute">
        <parent link="base_link"/>
        <child link="arm_link"/>
        <origin xyz="0 0 0.5" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>
    <joint name="arm_to_hand" type="fixed">
        <parent link="arm_link"/>
        <child link="hand_link"/>
        <origin xyz="0 0 0.3" rpy="0 0 0"/>
    </joint>
</robot>
"""


@pytest.fixture
def editor():
    """Create a FrankensteinEditor with a loaded model."""
    from model_generation.editor import FrankensteinEditor

    ed = FrankensteinEditor()
    ed.load_model("test_model", SIMPLE_URDF)
    # Create a mutable duplicate so modifications work
    ed.duplicate_model("test_model", "editable")
    return ed


# ===================================================================
# FrankensteinEditor Modification Mixin -- @precondition tests
# ===================================================================


class TestDeleteLinkPreconditions:
    """Tests for delete_link precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.delete_link("", "arm_link")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.delete_link(None, "arm_link")

    def test_empty_link_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="link_name"):
            editor.delete_link("editable", "")

    def test_none_link_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="link_name"):
            editor.delete_link("editable", None)

    def test_valid_inputs_work(self, editor):
        result = editor.delete_link("editable", "hand_link", reparent_children=False)
        assert result is True


class TestDeleteSubtreePreconditions:
    """Tests for delete_subtree precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.delete_subtree("", "arm_link")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.delete_subtree(None, "arm_link")

    def test_empty_root_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="root_link"):
            editor.delete_subtree("editable", "")

    def test_none_root_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="root_link"):
            editor.delete_subtree("editable", None)

    def test_valid_inputs_work(self, editor):
        result = editor.delete_subtree("editable", "hand_link")
        assert result is True


class TestRenameLinkPreconditions:
    """Tests for rename_link precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.rename_link("", "arm_link", "new_name")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.rename_link(None, "arm_link", "new_name")

    def test_empty_old_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="old_name"):
            editor.rename_link("editable", "", "new_name")

    def test_empty_new_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="new_name"):
            editor.rename_link("editable", "arm_link", "")

    def test_valid_inputs_work(self, editor):
        result = editor.rename_link("editable", "arm_link", "renamed_arm")
        assert result is True


class TestRenameJointPreconditions:
    """Tests for rename_joint precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.rename_joint("", "base_to_arm", "new_joint")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.rename_joint(None, "base_to_arm", "new_joint")

    def test_empty_old_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="old_name"):
            editor.rename_joint("editable", "", "new_joint")

    def test_empty_new_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="new_name"):
            editor.rename_joint("editable", "base_to_arm", "")

    def test_valid_inputs_work(self, editor):
        result = editor.rename_joint("editable", "base_to_arm", "renamed_joint")
        assert result is True


class TestModifyJointPreconditions:
    """Tests for modify_joint precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.modify_joint("", "base_to_arm")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.modify_joint(None, "base_to_arm")

    def test_empty_joint_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="joint_name"):
            editor.modify_joint("editable", "")

    def test_none_joint_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="joint_name"):
            editor.modify_joint("editable", None)

    def test_valid_inputs_work(self, editor):
        from model_generation.core.types import Origin

        result = editor.modify_joint(
            "editable", "base_to_arm", origin=Origin(xyz=(0, 0, 1))
        )
        assert result is True


class TestAttachLinkPreconditions:
    """Tests for attach_link precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.attach_link("", "base_link", "arm_link")

    def test_empty_parent_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="parent_link"):
            editor.attach_link("editable", "", "arm_link")

    def test_empty_child_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="child_link"):
            editor.attach_link("editable", "base_link", "")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.attach_link(None, "base_link", "arm_link")


class TestDetachLinkPreconditions:
    """Tests for detach_link precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.detach_link("", "arm_link")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.detach_link(None, "arm_link")

    def test_empty_link_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="link_name"):
            editor.detach_link("editable", "")

    def test_valid_inputs_work(self, editor):
        result = editor.detach_link("editable", "arm_link")
        assert result is True


class TestApplyPrefixPreconditions:
    """Tests for apply_prefix precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.apply_prefix("", "robot_")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.apply_prefix(None, "robot_")

    def test_empty_prefix_raises(self, editor):
        with pytest.raises(ContractViolationError, match="prefix"):
            editor.apply_prefix("editable", "")

    def test_none_prefix_raises(self, editor):
        with pytest.raises(ContractViolationError, match="prefix"):
            editor.apply_prefix("editable", None)

    def test_valid_inputs_work(self, editor):
        result = editor.apply_prefix("editable", "robot_")
        assert result is True


class TestMirrorSubtreePreconditions:
    """Tests for mirror_subtree precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.mirror_subtree("", "arm_link")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.mirror_subtree(None, "arm_link")

    def test_empty_root_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="root_link"):
            editor.mirror_subtree("editable", "")

    def test_none_root_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="root_link"):
            editor.mirror_subtree("editable", None)


# ===================================================================
# ClipboardMixin -- @precondition tests
# ===================================================================


class TestCopySubtreePreconditions:
    """Tests for copy_subtree precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.copy_subtree("", "arm_link")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.copy_subtree(None, "arm_link")

    def test_empty_root_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="root_link"):
            editor.copy_subtree("test_model", "")

    def test_none_root_link_raises(self, editor):
        with pytest.raises(ContractViolationError, match="root_link"):
            editor.copy_subtree("test_model", None)

    def test_valid_inputs_work(self, editor):
        result = editor.copy_subtree("test_model", "arm_link")
        assert result is True


class TestPasteSubtreePreconditions:
    """Tests for paste_subtree precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        # First copy something to clipboard
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="target_model_id"):
            editor.paste_subtree("", "base_link")

    def test_none_model_id_raises(self, editor):
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="target_model_id"):
            editor.paste_subtree(None, "base_link")

    def test_empty_attach_to_raises(self, editor):
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="attach_to"):
            editor.paste_subtree("editable", "")

    def test_none_attach_to_raises(self, editor):
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="attach_to"):
            editor.paste_subtree("editable", None)


class TestCopyLinkPreconditions:
    """Tests for copy_link precondition decorators."""

    def test_empty_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.copy_link("", "arm_link")

    def test_none_model_id_raises(self, editor):
        with pytest.raises(ContractViolationError, match="model_id"):
            editor.copy_link(None, "arm_link")

    def test_empty_link_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="link_name"):
            editor.copy_link("test_model", "")

    def test_none_link_name_raises(self, editor):
        with pytest.raises(ContractViolationError, match="link_name"):
            editor.copy_link("test_model", None)

    def test_valid_inputs_work(self, editor):
        result = editor.copy_link("test_model", "arm_link")
        assert result is True


# ===================================================================
# HumanoidURDFGenerator -- @precondition tests
# ===================================================================


class TestHumanoidGeneratorPreconditions:
    """Tests for HumanoidURDFGenerator DbC decorators."""

    def test_generate_none_params_raises(self):
        from humanoid_character_builder.generators.urdf_generator import (
            HumanoidURDFGenerator,
        )

        gen = HumanoidURDFGenerator()
        with pytest.raises((ContractViolationError, AttributeError, TypeError)):
            gen.generate(None)

    def test_build_model_none_params_raises(self):
        from humanoid_character_builder.generators.urdf_generator import (
            HumanoidURDFGenerator,
        )

        gen = HumanoidURDFGenerator()
        with pytest.raises((ContractViolationError, AttributeError, TypeError)):
            gen.build_model(None)

    def test_generate_valid_params_works(self):
        from humanoid_character_builder.core.body_parameters import BodyParameters
        from humanoid_character_builder.generators.urdf_generator import (
            HumanoidURDFGenerator,
        )

        params = BodyParameters(name="test_human", height_m=1.75, mass_kg=70.0)
        gen = HumanoidURDFGenerator()
        result = gen.generate(params)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "<robot" in result


# ===================================================================
# BaseURDFBuilder -- _check_invariants tests
# ===================================================================


class TestBaseURDFBuilderInvariants:
    """Tests for BaseURDFBuilder._check_invariants method."""

    def _create_concrete_builder(self):
        """Create a concrete subclass of BaseURDFBuilder for testing."""
        from model_generation.builders.base_builder import BaseURDFBuilder, BuildResult
        from model_generation.core.types import Inertia, Joint, JointType, Link, Origin

        class ConcreteBuilder(BaseURDFBuilder):
            def build(self, **kwargs):
                return BuildResult(success=True)

            def clear(self):
                self._links.clear()
                self._joints.clear()

        return ConcreteBuilder, Link, Joint, JointType, Origin, Inertia

    def test_check_invariants_empty_model(self):
        """Empty model passes invariants."""
        ConcreteBuilder, *_ = self._create_concrete_builder()
        builder = ConcreteBuilder("test")
        # Should not raise
        builder._check_invariants()

    def test_check_invariants_consistent_model(self):
        """Model with matching link/joint references passes."""
        ConcreteBuilder, Link, Joint, JointType, Origin, Inertia = (
            self._create_concrete_builder()
        )
        builder = ConcreteBuilder("test")

        base = Link(name="base", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        arm = Link(name="arm", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        joint = Joint(
            name="j1",
            joint_type=JointType.FIXED,
            parent="base",
            child="arm",
            origin=Origin(),
        )

        builder.add_link(base)
        builder.add_link(arm)
        builder.add_joint(joint)

        # Should not raise
        builder._check_invariants()

    def test_check_invariants_dangling_joint_parent(self):
        """Joint referencing nonexistent parent raises InvariantError."""
        from contracts import InvariantError

        ConcreteBuilder, Link, Joint, JointType, Origin, Inertia = (
            self._create_concrete_builder()
        )
        builder = ConcreteBuilder("test")

        arm = Link(name="arm", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        joint = Joint(
            name="j1",
            joint_type=JointType.FIXED,
            parent="nonexistent",
            child="arm",
            origin=Origin(),
        )

        builder._links.append(arm)
        builder._joints.append(joint)

        with pytest.raises(InvariantError, match="parent.*nonexistent"):
            builder._check_invariants()

    def test_check_invariants_dangling_joint_child(self):
        """Joint referencing nonexistent child raises InvariantError."""
        from contracts import InvariantError

        ConcreteBuilder, Link, Joint, JointType, Origin, Inertia = (
            self._create_concrete_builder()
        )
        builder = ConcreteBuilder("test")

        base = Link(name="base", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        joint = Joint(
            name="j1",
            joint_type=JointType.FIXED,
            parent="base",
            child="nonexistent",
            origin=Origin(),
        )

        builder._links.append(base)
        builder._joints.append(joint)

        with pytest.raises(InvariantError, match="child.*nonexistent"):
            builder._check_invariants()

    def test_check_invariants_duplicate_link_names(self):
        """Duplicate link names raise InvariantError."""
        from contracts import InvariantError

        ConcreteBuilder, Link, Joint, JointType, Origin, Inertia = (
            self._create_concrete_builder()
        )
        builder = ConcreteBuilder("test")

        link1 = Link(name="base", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        link2 = Link(name="base", inertia=Inertia(ixx=0.2, iyy=0.2, izz=0.2))

        builder._links.append(link1)
        builder._links.append(link2)

        with pytest.raises(InvariantError, match="[Dd]uplicate.*link"):
            builder._check_invariants()

    def test_check_invariants_duplicate_joint_names(self):
        """Duplicate joint names raise InvariantError."""
        from contracts import InvariantError

        ConcreteBuilder, Link, Joint, JointType, Origin, Inertia = (
            self._create_concrete_builder()
        )
        builder = ConcreteBuilder("test")

        base = Link(name="base", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        arm = Link(name="arm", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        hand = Link(name="hand", inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))

        j1 = Joint(
            name="j1",
            joint_type=JointType.FIXED,
            parent="base",
            child="arm",
            origin=Origin(),
        )
        j2 = Joint(
            name="j1",  # Duplicate name
            joint_type=JointType.FIXED,
            parent="arm",
            child="hand",
            origin=Origin(),
        )

        builder._links.extend([base, arm, hand])
        builder._joints.extend([j1, j2])

        with pytest.raises(InvariantError, match="[Dd]uplicate.*joint"):
            builder._check_invariants()
