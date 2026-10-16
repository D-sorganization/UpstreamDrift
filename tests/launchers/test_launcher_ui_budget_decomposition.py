"""Characterization tests for the launcher UI budget decomposition (#12839)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.launchers import custom_title_bar as ctb
from src.launchers.launcher_constants import ViewMode
from src.launchers.launcher_layout_manager import LayoutManager


@pytest.fixture(scope="module")
def _qapp() -> object:
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def manager() -> LayoutManager:
    return LayoutManager(
        config_file=Path("/fake/config.json"),
        available_models={},
        get_model_func=lambda _id: None,
        create_card_func=lambda _m: MagicMock(),
        create_header_func=lambda _n: MagicMock(),
    )


@pytest.mark.unit
def test_rebuild_grid_rejects_none(manager: LayoutManager) -> None:
    with pytest.raises(ValueError, match="grid_layout"):
        manager.rebuild_grid(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_clear_grid_hides_reusable_cards_and_deletes_others(
    manager: LayoutManager,
) -> None:
    card = MagicMock()
    other = MagicMock()
    manager.model_cards["m"] = card
    grid = MagicMock()
    grid.columnCount.return_value = 2
    grid.count.side_effect = [1, 1, 0]
    grid.takeAt.side_effect = [
        MagicMock(widget=lambda: card),
        MagicMock(widget=lambda: other),
    ]
    manager._clear_grid(grid)
    card.hide.assert_called_once()
    card.deleteLater.assert_not_called()
    other.deleteLater.assert_called_once()
    card.setParent.assert_called_with(None)
    grid.setColumnStretch.assert_any_call(0, 0)
    grid.setColumnStretch.assert_any_call(1, 0)


@pytest.mark.unit
def test_available_width_falls_back_for_mock_container() -> None:
    grid = MagicMock()
    assert LayoutManager._available_width(grid) == 800


@pytest.mark.unit
def test_grid_spacing_default_and_override() -> None:
    grid = MagicMock()
    assert LayoutManager._grid_spacing(grid) == 20
    grid.spacing.return_value = 7
    assert LayoutManager._grid_spacing(grid) == 7
    grid.spacing.side_effect = TypeError
    assert LayoutManager._grid_spacing(grid) == 20


@pytest.mark.unit
def test_compute_columns_uses_width_scale_and_spacing(manager: LayoutManager) -> None:
    grid = MagicMock()
    grid.spacing.return_value = 20
    # width 800 fallback: usable 780; card 240 -> (780+20)//(240+20) == 3
    assert manager._compute_columns(grid, 1.0) == 3
    # tiny scale clamps card width to 100 -> (780+20)//120 == 6
    assert manager._compute_columns(grid, 0.1) == 6


@pytest.mark.unit
def test_place_widgets_wraps_tiles_by_columns() -> None:
    grid = MagicMock()
    widgets = [MagicMock() for _ in range(5)]
    LayoutManager._place_widgets(grid, widgets, 2, False)
    positions = [c.args[1:] for c in grid.addWidget.call_args_list]
    assert positions == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    assert all(w.show.called for w in widgets)


@pytest.mark.unit
def test_place_widgets_list_mode_one_row_each() -> None:
    grid = MagicMock()
    widgets = [MagicMock() for _ in range(3)]
    LayoutManager._place_widgets(grid, widgets, 1, True)
    positions = [c.args[1:] for c in grid.addWidget.call_args_list]
    assert positions == [(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 1, 1)]


@pytest.mark.unit
def test_sync_cards_applies_list_compact_flag(manager: LayoutManager) -> None:
    card = MagicMock()
    manager.model_cards["a"] = card
    manager.get_filtered_order = lambda: ["a"]  # type: ignore[method-assign]
    manager.current_view_mode = ViewMode.LIST_SMALL
    shown = manager._sync_cards(1.0, False, True)
    assert shown == [card]
    card.set_tile_scale.assert_called_once_with(
        1.0, show_description=False, list_mode=True, list_compact=True
    )


@pytest.mark.unit
def test_hover_style_defaults_and_frozen() -> None:
    style = ctb.HoverStyle()
    assert style.bg == "rgba(255, 255, 255, 0.1)"
    assert style.text_color == ""
    with pytest.raises(AttributeError):
        style.bg = "x"  # type: ignore[misc]


@pytest.mark.unit
def test_window_control_button_applies_hover_style(_qapp: object) -> None:
    btn = ctb.create_window_control_button(
        "close",
        "X",
        tooltip="t",
        accessible_name="a",
        object_name="o",
        hover=ctb.HoverStyle(bg="#E81123", text_color="#FFFFFF"),
    )
    sheet = btn.styleSheet()
    assert "#E81123" in sheet
    assert "color: #FFFFFF;" in sheet


@pytest.mark.unit
def test_title_bar_builds_controls_and_signals(_qapp: object) -> None:
    bar = ctb.CustomTitleBar(show_close_button=True)
    assert bar.btn_close is not None
    assert bar.btn_min.objectName() == "window-control-minimize"
    assert bar.btn_max.objectName() == "window-control-maximize"
    assert bar.btn_close.objectName() == "window-control-close"
    got: list[str] = []
    bar.close_requested.connect(lambda: got.append("close"))
    bar.minimize_requested.connect(lambda: got.append("min"))
    bar.btn_close.click()
    bar.btn_min.click()
    assert got == ["close", "min"]
    assert "#E81123" in bar.btn_close.styleSheet()


@pytest.mark.unit
def test_title_bar_without_close_button(_qapp: object) -> None:
    bar = ctb.CustomTitleBar(show_close_button=False)
    assert bar.btn_close is None
    assert bar.graphicsEffect() is not None


@pytest.mark.unit
def test_library_widget_panes_built(_qapp: object) -> None:
    from src.launchers.library_widget import LibraryWidget

    widget = LibraryWidget(manager=MagicMock(list_documents=list, documents=[]))
    assert widget.table.columnCount() == 4
    assert not widget.chat_input.isEnabled()
    assert widget.findChild(type(widget.search_bar.parent()), "LibraryToolbar")
    for name in ("LibraryListPane", "LibraryPreviewPane", "LibraryChatPanel"):
        assert widget.findChild(type(widget.search_bar.parent()), name) is not None
