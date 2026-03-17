"""Tests for model_card widget."""

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QMimeData, QPoint, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent

from src.launchers.model_card import DraggableModelCard


@pytest.fixture
def parent_launcher():
    launcher = MagicMock()
    launcher.layout_edit_mode = True
    return launcher


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.id = "mujoco_unified"
    model.name = "MuJoCo"
    model.description = "Test Description"
    model.type = "engine_managed"
    model.engine_type = "mujoco"
    return model


def test_model_card_init(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)
    assert card.model == mock_model
    assert card.parent_launcher == parent_launcher
    assert card.acceptDrops() is True


def test_resolve_image_name(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)
    assert card._resolve_image_name() == "mujoco_humanoid.png"

    # Test fallback
    mock_model.name = "Unknown"
    mock_model.id = "drake_test"
    assert card._resolve_image_name() == "drake.png"


@patch("src.launchers.model_card.ASSETS_DIR")
def test_find_image_path(mock_assets_dir, mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_assets_dir.__truediv__.return_value = mock_path

    assert card._find_image_path("test.png") == mock_path

    mock_path.exists.return_value = False
    assert card._find_image_path("test.png") is None


def test_get_status_info(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    mock_model.type = "custom_humanoid"
    status, _, _ = card._get_status_info()
    assert status == "GUI Ready"

    mock_model.type = "mjcf"
    mock_model.path = "test.xml"
    status, _, _ = card._get_status_info()
    assert status == "Viewer"

    mock_model.type = "opensim"
    mock_model.path = ""
    status, _, _ = card._get_status_info()
    assert status == "Engine Ready"


def test_mouse_press_event(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QMouseEvent)
    event.button.return_value = Qt.MouseButton.LeftButton
    event.position().toPoint.return_value = QPoint(10, 10)

    card.mousePressEvent(event)
    parent_launcher.select_model.assert_called_once_with("mujoco_unified")
    assert card.drag_start_position == QPoint(10, 10)


def test_mouse_double_click_event(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QMouseEvent)
    card.mouseDoubleClickEvent(event)
    parent_launcher.launch_model_direct.assert_called_once_with("mujoco_unified")


def test_drag_enter_event(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QDragEnterEvent)
    mime = QMimeData()
    mime.setText("model_card:other_id")
    event.mimeData.return_value = mime

    card.dragEnterEvent(event)
    event.acceptProposedAction.assert_called_once()


def test_drop_event(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QDropEvent)
    mime = QMimeData()
    mime.setText("model_card:source_id")
    event.mimeData.return_value = mime

    card.dropEvent(event)
    parent_launcher._swap_models.assert_called_once_with("source_id", "mujoco_unified")
    event.acceptProposedAction.assert_called_once()
