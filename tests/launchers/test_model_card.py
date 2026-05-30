"""Tests for model_card widget."""

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtCore import QMimeData, QPoint, Qt  # noqa: E402
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QLabel  # noqa: E402
from src.launchers.model_card import DraggableModelCard  # noqa: E402


@pytest.fixture
def parent_launcher() -> MagicMock:
    launcher = MagicMock()
    launcher.layout_edit_mode = True
    return launcher


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.id = "mujoco_unified"
    model.name = "MuJoCo"
    model.description = "Test Description"
    model.engine_type = "mujoco"
    model.launcher = None
    return model


def test_model_card_init(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)
    assert card.model == mock_model
    assert card.parent_launcher == parent_launcher
    assert card.acceptDrops() is True


def test_resolve_image_name(mock_model, parent_launcher, qapp) -> None:
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
    assert card._resolve_image_name() == "c3d_viewer_modern.png"

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
def test_find_image_path(mock_assets_dir, mock_model, parent_launcher, qapp) -> None:
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


def test_get_status_info(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)

    mock_model.type = "custom_humanoid"
    status, _ = card._get_status_info()
    assert status == "GUI Ready"

    mock_model.type = "mjcf"
    mock_model.path = "test.xml"
    status, _ = card._get_status_info()
    assert status == "Viewer"

    mock_model.type = "opensim"
    mock_model.path = ""
    status, _ = card._get_status_info()
    assert status == "Engine Ready"

    mock_model.type = "matlab"
    status, _ = card._get_status_info()
    assert status == "Ready"

    mock_model.type = "urdf_generator"
    status, _ = card._get_status_info()
    assert status == "Utility"

    mock_model.type = "something_else"
    status, _ = card._get_status_info()
    assert status == "Ready"


def test_mouse_press_event(mock_model, parent_launcher, qapp) -> None:
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


def test_mouse_double_click_event(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QMouseEvent)
    card.mouseDoubleClickEvent(event)
    parent_launcher.launch_model_direct.assert_called_once_with("mujoco_unified")


def test_drag_enter_event(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QDragEnterEvent)
    mime = QMimeData()
    mime.setText("model_card:other_id")
    event.mimeData.return_value = mime

    card.dragEnterEvent(event)
    event.acceptProposedAction.assert_called_once()


def test_drop_event(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)

    event = MagicMock(spec=QDropEvent)
    mime = QMimeData()
    mime.setText("model_card:source_id")
    event.mimeData.return_value = mime

    card.dropEvent(event)
    parent_launcher._swap_models.assert_called_once_with("source_id", "mujoco_unified")
    event.acceptProposedAction.assert_called_once()


def test_no_image_widget(mock_model, parent_launcher, qapp) -> None:
    with patch(
        "src.launchers.model_card.DraggableModelCard._find_image_path",
        return_value=None,
    ):
        card = DraggableModelCard(mock_model, parent_launcher)
        img = card.findChild(QLabel, "CardImage")
        assert img is not None
        assert img.text() == "No Image"


def test_refresh_theme(mock_model, parent_launcher, qapp) -> None:
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

        def find_mock(*args) -> MagicMock:
            if args[1] == "CardImage":
                return img
            return MagicMock()

        with patch.object(card, "findChild", side_effect=find_mock):
            card.refresh_theme()


def test_model_card_uses_shared_typography(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)
    labels = card.findChildren(QLabel)
    target_labels = [
        label
        for label in labels
        if label.text() in {"MuJoCo", "Test Description", "GUI Ready"}
    ]
    assert target_labels
    for label in target_labels:
        assert "Outfit" in label.font().families()


def test_mouse_move_event(mock_model, parent_launcher, qapp) -> None:
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


def test_mouse_double_click_no_parent(mock_model, qapp) -> None:
    card = DraggableModelCard(mock_model, None)
    event = MagicMock(spec=QMouseEvent)
    card.mouseDoubleClickEvent(event)  # should not crash


def test_drag_enter_event_empty(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)
    card.dragEnterEvent(None)

    event = MagicMock(spec=QDragEnterEvent)
    event.mimeData.return_value = None
    card.dragEnterEvent(event)
    event.acceptProposedAction.assert_not_called()


def test_drop_event_empty(mock_model, parent_launcher, qapp) -> None:
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


def test_enter_event(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QEnterEvent

    event = QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0))
    with patch.object(card._hover_anim, "start") as mock_start:
        card.enterEvent(event)
        mock_start.assert_called_once()
        assert card._hover_anim.endValue() == 4.0


def test_leave_event(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)
    from PyQt6.QtCore import QEvent

    event = QEvent(QEvent.Type.Leave)
    with patch.object(card._hover_anim, "start") as mock_start:
        card.leaveEvent(event)
        mock_start.assert_called_once()
        assert card._hover_anim.endValue() == 0.0


def test_quick_launch_button_at_bottom(mock_model, parent_launcher, qapp) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)
    card.resize(200, 300)
    card._reposition_quick_launch_button()

    # Assert launch button is at the bottom (grid mode)
    btn = card._btn_quick_launch
    margin = 6
    expected_y = card.height() - btn.height() - margin
    assert btn.y() == expected_y

    # Assert top-right position for favorite and info buttons
    fav_btn = card._btn_favorite
    info_btn = card._btn_info
    assert fav_btn.y() == margin
    assert info_btn.y() == margin

    # Test list mode
    card = DraggableModelCard(mock_model, parent_launcher, list_mode=True)
    card.resize(400, 85)
    card._reposition_quick_launch_button()
    btn = card._btn_quick_launch
    expected_list_y = card.height() - btn.height() - margin
    assert btn.y() == expected_list_y


def test_info_and_favorite_buttons_hidden_unless_hovered(
    mock_model, parent_launcher, qapp
) -> None:
    card = DraggableModelCard(mock_model, parent_launcher)
    fav_btn = card._btn_favorite
    info_btn = card._btn_info
    launch_btn = card._btn_quick_launch

    # Initially hidden
    assert fav_btn.isHidden()
    assert info_btn.isHidden()
    assert launch_btn.isHidden()

    # Selecting the card should not show favorite/info buttons
    card.set_selected(True)
    assert fav_btn.isHidden()
    assert info_btn.isHidden()

    # Hover Enter: should show all
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QEnterEvent

    enter_ev = QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0))
    card.enterEvent(enter_ev)
    assert not fav_btn.isHidden()
    assert not info_btn.isHidden()
    assert not launch_btn.isHidden()

    # Hover Leave (while selected): should hide all
    from PyQt6.QtCore import QEvent

    leave_ev = QEvent(QEvent.Type.Leave)
    card.leaveEvent(leave_ev)
    assert fav_btn.isHidden()
    assert info_btn.isHidden()
    assert launch_btn.isHidden()


def test_reposition_quick_launch_button_handles_tiny_sizes(
    mock_model, parent_launcher, qapp
) -> None:
    """Verify that repositioning doesn't raise AssertionError when widget is tiny/zero size."""
    card = DraggableModelCard(mock_model, parent_launcher)
    card.resize(0, 0)
    # Should not raise AssertionError
    card._reposition_quick_launch_button()
