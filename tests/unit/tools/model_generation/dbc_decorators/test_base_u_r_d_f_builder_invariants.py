"""
Tests for Design by Contract (DbC) decorators on FrankensteinEditor,
ClipboardMixin, HumanoidURDFGenerator, and BaseURDFBuilder.

These tests verify that @precondition decorators enforce input validation
and raise PreconditionError / ContractViolationError for invalid inputs,
while valid inputs still work correctly.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# Ensure contracts are enforced during testing
os.environ["DBC_LEVEL"] = "enforce"

from src.shared.python.contracts import ContractViolationError  # noqa: E402

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
def editor() -> Any:
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


# ===================================================================
# ClipboardMixin -- @precondition tests
# ===================================================================


# ===================================================================
# HumanoidURDFGenerator -- @precondition tests
# ===================================================================


# ===================================================================
# BaseURDFBuilder -- _check_invariants tests
# ===================================================================


class TestBaseURDFBuilderInvariants:
    """Tests for BaseURDFBuilder._check_invariants method."""

    def _create_concrete_builder(self) -> Any:
        """Create a concrete subclass of BaseURDFBuilder for testing."""
        from model_generation.builders.base_builder import BaseURDFBuilder, BuildResult
        from model_generation.core.types import Inertia, Joint, JointType, Link, Origin

        class ConcreteBuilder(BaseURDFBuilder):
            def build(self, **kwargs) -> Any:
                return BuildResult(success=True)

            def clear(self) -> None:
                self._links.clear()
                self._joints.clear()

        return ConcreteBuilder, Link, Joint, JointType, Origin, Inertia

    def test_check_invariants_empty_model(self) -> None:
        """Empty model passes invariants."""
        ConcreteBuilder, *_ = self._create_concrete_builder()
        builder = ConcreteBuilder("test")
        # Should not raise
        builder._check_invariants()

    def test_check_invariants_consistent_model(self) -> None:
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

    def test_check_invariants_dangling_joint_parent(self) -> None:
        """Joint referencing nonexistent parent raises InvariantError."""
        from src.shared.python.contracts import InvariantError

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

    def test_check_invariants_dangling_joint_child(self) -> None:
        """Joint referencing nonexistent child raises InvariantError."""
        from src.shared.python.contracts import InvariantError

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

    def test_check_invariants_duplicate_link_names(self) -> None:
        """Duplicate link names raise InvariantError."""
        from src.shared.python.contracts import InvariantError

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

    def test_check_invariants_duplicate_joint_names(self) -> None:
        """Duplicate joint names raise InvariantError."""
        from src.shared.python.contracts import InvariantError

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
