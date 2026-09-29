"""Tests for drake_pose_editor_tab.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.src.pose_editor_tab import (
    DrakePoseEditor,
    DrakePoseEditorTab,
    PYQT6_AVAILABLE,
)

if not PYQT6_AVAILABLE:
    pytest.skip(
        "Skipping PyQt tests since PyQt6 is not available", allow_module_level=True
    )


class TestDrakePoseEditor:
    @patch(
        "src.engines.physics_engines.drake.python.src.pose_editor_tab.RevoluteJoint",
        type("RevoluteJoint", (), {}),
    )
    @patch(
        "src.engines.physics_engines.drake.python.src.pose_editor_tab.PrismaticJoint",
        type("PrismaticJoint", (), {}),
    )
    @patch("src.engines.physics_engines.drake.python.src.pose_editor_tab.JointIndex")
    def test_initialization(self, mock_joint_index):
        plant = MagicMock()
        context = MagicMock()

        joint = MagicMock()
        joint.num_positions.return_value = 1
        joint.num_velocities.return_value = 1
        joint.name.return_value = "shoulder_l"

        plant.num_joints.return_value = 1
        plant.get_joint.return_value = joint

        editor = DrakePoseEditor(plant, context)
        info = editor.get_joint_info()
        assert len(info) == 1
        assert info[0].name == "shoulder_l"

    @patch(
        "src.engines.physics_engines.drake.python.src.pose_editor_tab.RevoluteJoint",
        type("RevoluteJoint", (), {}),
    )
    @patch(
        "src.engines.physics_engines.drake.python.src.pose_editor_tab.PrismaticJoint",
        type("PrismaticJoint", (), {}),
    )
    @patch("src.engines.physics_engines.drake.python.src.pose_editor_tab.JointIndex")
    def test_get_set_positions(self, mock_joint_index):
        plant = MagicMock()
        context = MagicMock()
        editor = DrakePoseEditor(plant, context)

        positions = np.array([1.0, 2.0])
        plant.GetPositions.return_value = positions

        assert np.array_equal(editor.get_all_positions(), positions)

        editor.set_all_positions(np.array([3.0, 4.0]))
        plant.SetPositions.assert_called_once()

    @patch(
        "src.engines.physics_engines.drake.python.src.pose_editor_tab.RevoluteJoint",
        type("RevoluteJoint", (), {}),
    )
    @patch(
        "src.engines.physics_engines.drake.python.src.pose_editor_tab.PrismaticJoint",
        type("PrismaticJoint", (), {}),
    )
    @patch("src.engines.physics_engines.drake.python.src.pose_editor_tab.JointIndex")
    def test_gravity_toggle(self, mock_joint_index):
        plant = MagicMock()
        context = MagicMock()
        editor = DrakePoseEditor(plant, context)

        editor.set_gravity_enabled(False)
        args = plant.mutable_gravity_field().set_gravity_vector.call_args[0][0]
        np.testing.assert_array_equal(args, np.zeros(3))


class TestDrakePoseEditorTab:
    @patch("src.engines.physics_engines.drake.python.src.pose_editor_tab.QtWidgets")
    def test_setup_ui(self, mock_qt_widgets, qapp):
        tab = DrakePoseEditorTab()

        # Verify ui properties
        assert hasattr(tab, "lbl_mode")
        assert hasattr(tab, "gravity_widget")
        assert hasattr(tab, "btn_reset")
        assert hasattr(tab, "txt_filter")

    @patch("src.engines.physics_engines.drake.python.src.pose_editor_tab.QtWidgets")
    def test_on_joint_changed(self, mock_qt_widgets, qapp):
        tab = DrakePoseEditorTab()
        tab._editor = MagicMock()
        tab._editor.get_all_positions.return_value = np.array([1.5])

        tab._on_joint_changed(0, 1.5)
        tab._editor.set_joint_position.assert_called_once_with(0, 1.5)
        tab._editor.update_visualization.assert_called_once()
