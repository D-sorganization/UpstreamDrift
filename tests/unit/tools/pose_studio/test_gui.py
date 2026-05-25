from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.tools.pose_studio.gui import PoseStudioWindow, main
from src.tools.pose_studio.core import EngineStatus
from src.shared.python.pose_interchange.canonical import (
    canonical_zero_pose,
)
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)


def test_gui_initialization() -> None:
    # Test fallback to "drake" if unknown engine is provided
    win = PoseStudioWindow(initial_engine="unknown_engine")
    assert win is not None
    assert win._engine_controller.engine_name == "drake"

    # Check that it doesn't crash on applying initial pose


def test_gui_on_engine_selected() -> None:
    win = PoseStudioWindow()
    # Mock the controller switch to avoid full initialization of real engines if they exist
    win._engine_controller.switch_engine = MagicMock(return_value=EngineStatus.MOCK)

    win._on_engine_selected("mujoco")

    win._engine_controller.switch_engine.assert_called_with("mujoco")
    # Real methods run without crashing
    assert True  # just checking no crash


def test_gui_on_angle_edited() -> None:
    win = PoseStudioWindow()
    win._history.push = MagicMock()

    # Edit valid angle
    joint = REFERENCE_GOLFER_FIELDS[0]
    win._on_angle_edited(joint, 45.0)

    assert win._history.push.called

    # Edit with an error
    win._history.push.reset_mock()

    # Triggering ValueError by injecting invalid joint name (though GUI doesn't do this, testing the try/except)
    win._on_angle_edited("unknown_joint", 45.0)
    assert not win._history.push.called


def test_gui_undo_redo() -> None:
    win = PoseStudioWindow()

    pose1 = canonical_zero_pose()

    # mock history controller
    win._history.undo = MagicMock(return_value=pose1)
    win._history.redo = MagicMock(return_value=pose1)

    win._on_undo()
    win._history.undo.assert_called_once()

    win._on_redo()
    win._history.redo.assert_called_once()

    # Test none return from history
    win._history.undo = MagicMock(return_value=None)
    win._history.redo = MagicMock(return_value=None)

    win._on_undo()
    win._on_redo()


def test_gui_load_poses() -> None:
    win = PoseStudioWindow()
    win._apply_pose = MagicMock()

    win._on_load_zero()
    assert win._apply_pose.call_count == 1

    win._on_load_reference()
    assert win._apply_pose.call_count == 2


def test_gui_save_load_clicked() -> None:
    win = PoseStudioWindow()
    # Ensure no crashes on clicking save/load
    win._on_save_clicked()
    win._on_load_clicked()


def test_gui_apply_pose_with_service_transforms() -> None:
    win = PoseStudioWindow()

    pose = canonical_zero_pose()

    # Mock service with get_link_transforms
    mock_service = MagicMock()
    mock_service.get_link_transforms.return_value = {"pelvis": np.eye(4)}
    win._engine_controller._service = mock_service

    # Patch the real view_3d method so we can assert on it
    with patch.object(win.view_3d, "update_from_service_transforms") as mock_update:
        win._apply_pose(pose, record_history=False)
        assert mock_update.call_count == 1
        args, _ = mock_update.call_args
        assert "pelvis" in args[0]
        np.testing.assert_array_equal(args[0]["pelvis"], np.eye(4))

    # Test exception fallback
    mock_service.get_link_transforms.side_effect = NotImplementedError()
    with patch.object(win.view_3d, "update_pose") as mock_update_pose:
        win._apply_pose(pose, record_history=False)
        mock_update_pose.assert_called_with(pose)


@patch("src.tools.pose_studio.gui.QtWidgets.QApplication")
def test_gui_main(mock_qapp) -> None:
    # Mock QApplication and its instance method
    mock_app_instance = MagicMock()
    mock_app_instance.exec.return_value = 0

    # If instance() returns None, it calls the constructor
    mock_qapp.instance.return_value = None
    mock_qapp.return_value = mock_app_instance

    assert main(["--test"]) == 0
    mock_qapp.assert_called()
    mock_app_instance.exec.assert_called_once()
