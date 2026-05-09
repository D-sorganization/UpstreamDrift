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


class TestPasteSubtreePreconditions:
    """Tests for paste_subtree precondition decorators."""

    def test_empty_model_id_raises(self, editor) -> None:
        # First copy something to clipboard
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="target_model_id"):
            editor.paste_subtree("", "base_link")

    def test_none_model_id_raises(self, editor) -> None:
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="target_model_id"):
            editor.paste_subtree(None, "base_link")

    def test_empty_attach_to_raises(self, editor) -> None:
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="attach_to"):
            editor.paste_subtree("editable", "")

    def test_none_attach_to_raises(self, editor) -> None:
        editor.copy_subtree("test_model", "arm_link")
        with pytest.raises(ContractViolationError, match="attach_to"):
            editor.paste_subtree("editable", None)


# ===================================================================
# HumanoidURDFGenerator -- @precondition tests
# ===================================================================


# ===================================================================
# BaseURDFBuilder -- _check_invariants tests
# ===================================================================
