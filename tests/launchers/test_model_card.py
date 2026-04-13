"""Tests for model_card widget."""

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtCore import QMimeData, QPoint, Qt  # noqa: E402
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

from src.launchers.model_card import DraggableModelCard  # noqa: E402


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
    mock_model.id = "mujoco_test"
    assert card._resolve_image_name() == "mujoco_humanoid.png"

    mock_model.id = "drake_test"
    assert card._resolve_image_name() == "drake.png"

    mock_model.id = "pinocchio_test"
    assert card._resolve_image_name() == "pinocchio.png"

    mock_model.id = "opensim_test"
    assert card._resolve_image_name() == "opensim.png"

    mock_model.id = "myosim_test"
    assert card._resolve_image_name() == "myosim.png"

    mock_model.id = "matlab_test"
    assert card._resolve_image_name() == "matlab_logo.png"

    mock_model.id = "motion_test"
    assert card._resolve_image_name() == "c3d_icon.png"

    mock_model.id = "model_explorer_test"
    assert card._resolve_image_name() == "urdf_icon.png"

    mock_model.id = "random_test"
    mock_model.type = "engine_managed"
    mock_model.engine_type = "mujoco"
    assert card._resolve_image_name() == "mujoco_humanoid.png"

    mock_model.id = "completely_unknown"
    mock_model.type = "unknown"
    mock_model.engine_type = "unknown"
    assert card._resolve_image_name() is None


@patch("src.launchers.model_card.ASSETS_DIR")
def test_find_image_path(mock_assets_dir, mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_assets_dir.__truediv__.return_value = mock_path

    assert card._find_image_path("test.png") == mock_path

    assert card._find_image_path("test.png") == mock_path

    mock_path.exists.return_value = False
    with patch("src.launchers.model_card.Path") as mock_base_path:
        mock_svg = MagicMock()
        mock_svg.exists.return_value = True
        mock_base_path.return_value.parent.parent.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_svg
        assert card._find_image_path("test.png") == mock_svg

    with patch("src.launchers.model_card.Path") as mock_base_path:
        mock_svg = MagicMock()
        mock_svg.exists.return_value = False
        mock_base_path.return_value.parent.parent.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_svg
        assert card._find_image_path("test.png") is None

    assert card._find_image_path(None) is None


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

    mock_model.type = "matlab"
    status, _, _ = card._get_status_info()
    assert status == "External"

    mock_model.type = "urdf_generator"
    status, _, _ = card._get_status_info()
    assert status == "Utility"

    mock_model.type = "something_else"
    status, _, _ = card._get_status_info()
    assert status == "Unknown"


def test_mouse_press_event(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QMouseEvent)
    event.button.return_value = Qt.MouseButton.LeftButton
    event.position().toPoint.return_value = QPoint(10, 10)

    card.mousePressEvent(event)
    parent_launcher.select_model.assert_called_once_with("mujoco_unified")
    assert card.drag_start_position == QPoint(10, 10)

    # Missing event or wrong button
    event.button.return_value = Qt.MouseButton.RightButton
    card.mousePressEvent(event)  # should do nothing

    # No parent launcher
    card.parent_launcher = None
    event.button.return_value = Qt.MouseButton.LeftButton
    card.mousePressEvent(event)  # should not crash


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


def test_no_image_widget(mock_model, parent_launcher, qapp):
    with patch(
        "src.launchers.model_card.DraggableModelCard._find_image_path",
        return_value=None,
    ):
        card = DraggableModelCard(mock_model, parent_launcher)
        img = card.findChild(QWidget, "CardImage")
        assert img is not None
        assert img.text() == "No Image"


def test_refresh_theme(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)
    with patch("src.launchers.model_card._get_theme_colors") as mock_colors:
        mock_c = MagicMock()
        mock_c.text_secondary = "#ff0000"
        mock_c.text_quaternary = "#00ff00"
        mock_c.success = "#0000ff"
        mock_colors.return_value = mock_c

        card.refresh_theme()

        # Missing children
        with patch.object(card, "findChild", return_value=None):
            card.refresh_theme()

        # Image has pixmap
        img = MagicMock()
        img.pixmap.return_value = True

        def find_mock(*args):
            if args[1] == "CardImage":
                return img
            return MagicMock()

        with patch.object(card, "findChild", side_effect=find_mock):
            card.refresh_theme()


def test_mouse_move_event(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)
    card.drag_start_position = QPoint(0, 0)

    event = MagicMock(spec=QMouseEvent)
    event.buttons.return_value = Qt.MouseButton.LeftButton
    event.position().toPoint.return_value = QPoint(
        50, 50
    )  # Over QApplication.startDragDistance()

    # Needs to be in edit mode
    parent_launcher.layout_edit_mode = True

    with patch("src.launchers.model_card.QDrag") as mock_drag:
        card.mouseMoveEvent(event)
        mock_drag.assert_called_once()
        mock_drag().exec.assert_called_once()

    # Not enough distance
    event.position().toPoint.return_value = QPoint(1, 1)
    with patch("src.launchers.model_card.QDrag") as mock_drag:
        card.mouseMoveEvent(event)
        mock_drag.assert_not_called()

    # Not edit mode
    parent_launcher.layout_edit_mode = False
    event.position().toPoint.return_value = QPoint(50, 50)
    with patch("src.launchers.model_card.QDrag") as mock_drag:
        card.mouseMoveEvent(event)
        mock_drag.assert_not_called()

    # Wrong button
    parent_launcher.layout_edit_mode = True
    event.buttons.return_value = Qt.MouseButton.RightButton
    with patch("src.launchers.model_card.QDrag") as mock_drag:
        card.mouseMoveEvent(event)
        mock_drag.assert_not_called()

    # None event
    with patch("src.launchers.model_card.QDrag") as mock_drag:
        card.mouseMoveEvent(None)
        mock_drag.assert_not_called()


def test_mouse_double_click_no_parent(mock_model, qapp):
    card = DraggableModelCard(mock_model, None)
    event = MagicMock(spec=QMouseEvent)
    card.mouseDoubleClickEvent(event)  # should not crash


def test_drag_enter_event_empty(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)
    card.dragEnterEvent(None)

    event = MagicMock(spec=QDragEnterEvent)
    event.mimeData.return_value = None
    card.dragEnterEvent(event)
    event.acceptProposedAction.assert_not_called()


def test_drop_event_empty(mock_model, parent_launcher, qapp):
    card = DraggableModelCard(mock_model, parent_launcher)
    card.dropEvent(None)

    event = MagicMock(spec=QDropEvent)
    event.mimeData.return_value = None
    card.dropEvent(event)
    event.acceptProposedAction.assert_not_called()

    # Same ID
    mime = QMimeData()
    mime.setText(f"model_card:{mock_model.id}")
    event.mimeData.return_value = mime
    card.dropEvent(event)
    parent_launcher._swap_models.assert_not_called()
