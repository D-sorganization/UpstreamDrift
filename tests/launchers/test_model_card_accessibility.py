"""Unit tests for ModelCard accessibility, focus handling, context menu, and grid navigation (#8901).

Validates:
- WCAG 2.5.8 compliant target size (>= 24x24 px) on info and favorite buttons.
- StrongFocus policy and descriptive accessibleName on action buttons.
- Dynamic accessibleName update when toggling favorite state.
- focusInEvent and focusOutEvent revealing and hiding action buttons.
- Context menu actions ("Launch", "Add to/Remove from favorites", "Model Details...").
- Arrow-key navigation in QGridLayout shifting keyboard focus to adjacent cards.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QContextMenuEvent, QFocusEvent, QKeyEvent
from PyQt6.QtWidgets import QGridLayout, QMenu, QWidget

from src.launchers.model_card import DraggableModelCard

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_launcher() -> MagicMock:
    launcher = MagicMock()
    launcher.layout_edit_mode = False
    launcher.layout_manager = MagicMock()
    launcher.layout_manager.favorites = []
    launcher.layout_manager.current_category_filter = "All"
    return launcher


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.id = "test_model_1"
    model.name = "Test Model"
    model.description = "A sample model for accessibility tests"
    model.engine_type = "mujoco"
    model.launcher = None
    model.path = "virtual/matlab_suite"
    return model


def test_action_buttons_wcag_target_size(mock_model, mock_launcher, qapp) -> None:
    """Verify info and favorite buttons meet WCAG 2.5.8 minimum target size of 24x24 px."""
    card = DraggableModelCard(mock_model, mock_launcher)
    info_btn = card._btn_info
    fav_btn = card._btn_favorite

    assert info_btn.width() >= 24
    assert info_btn.height() >= 24
    assert fav_btn.width() >= 24
    assert fav_btn.height() >= 24


def test_action_buttons_focus_policy_and_accessible_name(
    mock_model, mock_launcher, qapp
) -> None:
    """Verify buttons have StrongFocus and meaningful accessibleName."""
    card = DraggableModelCard(mock_model, mock_launcher)
    info_btn = card._btn_info
    fav_btn = card._btn_favorite

    assert info_btn.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert fav_btn.focusPolicy() == Qt.FocusPolicy.StrongFocus

    assert info_btn.accessibleName() == "About Test Model"
    assert fav_btn.accessibleName() == "Add Test Model to favorites"


def test_toggle_favorite_updates_accessible_name(
    mock_model, mock_launcher, qapp
) -> None:
    """Verify toggling favorite updates both tooltip and accessibleName."""
    card = DraggableModelCard(mock_model, mock_launcher)
    fav_btn = card._btn_favorite

    # Initially not favorite
    assert fav_btn.accessibleName() == "Add Test Model to favorites"

    # Toggle to favorite
    card._toggle_favorite()
    assert "test_model_1" in mock_launcher.layout_manager.favorites
    assert fav_btn.accessibleName() == "Remove Test Model from favorites"
    assert fav_btn.text() == "★"

    # Toggle back
    card._toggle_favorite()
    assert "test_model_1" not in mock_launcher.layout_manager.favorites
    assert fav_btn.accessibleName() == "Add Test Model to favorites"
    assert fav_btn.text() == "☆"


def test_focus_in_and_out_visibility(mock_model, mock_launcher, qapp) -> None:
    """Verify focusInEvent reveals buttons and focusOutEvent conceals them when not focused."""
    card = DraggableModelCard(mock_model, mock_launcher)
    info_btn = card._btn_info
    fav_btn = card._btn_favorite
    launch_btn = card._btn_quick_launch

    assert info_btn.isHidden()
    assert fav_btn.isHidden()
    assert launch_btn.isHidden()

    # Trigger focusInEvent on card
    focus_in = QFocusEvent(QFocusEvent.Type.FocusIn, Qt.FocusReason.TabFocusReason)
    card.focusInEvent(focus_in)

    assert not info_btn.isHidden()
    assert not fav_btn.isHidden()
    assert not launch_btn.isHidden()

    # Trigger focusOutEvent when neither card nor children has focus
    focus_out = QFocusEvent(QFocusEvent.Type.FocusOut, Qt.FocusReason.TabFocusReason)
    card.focusOutEvent(focus_out)

    assert info_btn.isHidden()
    assert fav_btn.isHidden()
    assert launch_btn.isHidden()


def test_focus_out_does_not_hide_if_child_has_focus(
    mock_model, mock_launcher, qapp
) -> None:
    """Verify action buttons remain visible if a child button holds focus."""
    card = DraggableModelCard(mock_model, mock_launcher)
    info_btn = card._btn_info
    card._show_action_buttons()

    with patch.object(info_btn, "hasFocus", return_value=True):
        card._hide_action_buttons()
        # Buttons should still be visible because child button has focus
        assert not info_btn.isHidden()


def test_context_menu_event(mock_model, mock_launcher, qapp) -> None:
    """Verify context menu provides Launch, favorite, and Details actions."""
    card = DraggableModelCard(mock_model, mock_launcher)

    context_ev = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse, QPoint(10, 10), QPoint(50, 50)
    )

    executed_actions: list[str] = []

    def mock_exec(pos):
        # Inspect actions registered on the active menu
        menus = card.findChildren(QMenu)
        assert len(menus) >= 1
        active_menu = menus[-1]
        action_texts = [a.text() for a in active_menu.actions()]
        assert "Launch" in action_texts
        assert "Add to favorites" in action_texts
        assert "Model Details..." in action_texts

        # Trigger "Launch"
        for a in active_menu.actions():
            if a.text() == "Launch":
                executed_actions.append(a.text())
                return a
        return None

    with patch.object(QMenu, "exec", side_effect=mock_exec):
        card.contextMenuEvent(context_ev)
        assert executed_actions == ["Launch"]
        mock_launcher.launch_model_direct.assert_called_once_with("test_model_1")


def test_grid_arrow_key_navigation(mock_launcher, qapp) -> None:
    """Verify arrow keys shift keyboard focus to adjacent cards in a QGridLayout."""
    container = QWidget()
    grid = QGridLayout(container)
    mock_launcher.grid_layout = grid

    # Create 4 models in a 2x2 grid:
    # (0, 0): model_00 | (0, 1): model_01
    # (1, 0): model_10 | (1, 1): model_11
    cards: list[DraggableModelCard] = []
    for r in range(2):
        for c in range(2):
            m = MagicMock()
            m.id = f"model_{r}_{c}"
            m.name = f"Model {r},{c}"
            m.description = "Desc"
            m.engine_type = "mujoco"
            m.launcher = None
            m.path = "virtual/matlab_suite"
            card = DraggableModelCard(m, mock_launcher)
            grid.addWidget(card, r, c)
            cards.append(card)

    container.show()
    card_00 = cards[0]
    card_01 = cards[1]
    card_10 = cards[2]
    card_11 = cards[3]

    with (
        patch.object(card_00, "setFocus") as mock_focus_00,
        patch.object(card_01, "setFocus") as mock_focus_01,
        patch.object(card_10, "setFocus") as mock_focus_10,
        patch.object(card_11, "setFocus") as mock_focus_11,
    ):
        # Move Right: from (0, 0) to (0, 1)
        ev_right = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier
        )
        card_00.keyPressEvent(ev_right)
        mock_focus_01.assert_called_once_with(Qt.FocusReason.ShortcutFocusReason)
        mock_launcher.select_model.assert_called_with("model_0_1")

        # Move Down: from (0, 1) to (1, 1)
        ev_down = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier
        )
        card_01.keyPressEvent(ev_down)
        mock_focus_11.assert_called_once_with(Qt.FocusReason.ShortcutFocusReason)
        mock_launcher.select_model.assert_called_with("model_1_1")

        # Move Left: from (1, 1) to (1, 0)
        ev_left = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier
        )
        card_11.keyPressEvent(ev_left)
        mock_focus_10.assert_called_once_with(Qt.FocusReason.ShortcutFocusReason)
        mock_launcher.select_model.assert_called_with("model_1_0")

        # Move Up: from (1, 0) to (0, 0)
        ev_up = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier
        )
        card_10.keyPressEvent(ev_up)
        mock_focus_00.assert_called_once_with(Qt.FocusReason.ShortcutFocusReason)
        mock_launcher.select_model.assert_called_with("model_0_0")

    container.close()
