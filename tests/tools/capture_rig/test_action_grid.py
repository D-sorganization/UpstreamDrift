"""The action grid wraps instead of setting a screen-wide floor (#9844)."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("cv2")

from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from src.tools.capture_rig.action_grid import NARROW_WIDTH, ActionGrid
from src.tools.capture_rig.flow_layout import FlowLayout
from src.tools.capture_rig.gui import CaptureRigWidget

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _buttons() -> dict[str, QPushButton]:
    return {a: QPushButton(label) for a, label in CaptureRigWidget._ACTIONS}


def _grid_at(width: int) -> ActionGrid:
    grid = ActionGrid(_buttons())
    grid.resize(width, grid.heightForWidth(width))
    grid.layout().activate()
    return grid


# -- the bound -----------------------------------------------------------------
def test_the_grid_minimum_width_fits_a_narrow_pane() -> None:
    _app()
    grid = ActionGrid(_buttons())
    assert grid.minimumSizeHint().width() <= NARROW_WIDTH


def test_every_button_is_still_present_and_reachable_when_narrow() -> None:
    _app()
    buttons = _buttons()
    grid = ActionGrid(buttons)
    grid.resize(NARROW_WIDTH, grid.heightForWidth(NARROW_WIDTH))
    grid.layout().activate()
    assert len(buttons) == len(CaptureRigWidget._ACTIONS)
    for action, button in buttons.items():
        assert not button.isHidden(), action
        assert button.width() > 0 and button.height() > 0, action
        assert button.x() >= 0 and button.geometry().right() <= grid.width(), action


def test_the_grid_reflows_onto_fewer_rows_when_it_is_wider() -> None:
    _app()
    grid = ActionGrid(_buttons())
    assert grid.heightForWidth(1200) < grid.heightForWidth(400)


def test_a_section_label_stays_next_to_its_first_button() -> None:
    _app()
    for width in (400, 1200):
        buttons = _buttons()
        grid = ActionGrid(buttons)
        grid.resize(width, grid.heightForWidth(width))
        grid.layout().activate()
        for title, actions in grid.groups:
            label = grid.section_labels[title]
            first = buttons[actions[0]]
            gap = first.y() - label.y()
            assert 0 <= gap <= label.height() + 8, (width, title)


def test_the_grid_keeps_every_tooltip() -> None:
    _app()
    buttons = _buttons()
    for index, button in enumerate(buttons.values()):
        button.setToolTip(f"hint {index}")
    grid = ActionGrid(buttons)
    assert grid.section_labels
    assert [b.toolTip() for b in buttons.values()] == [
        f"hint {i}" for i in range(len(buttons))
    ]


# -- the layout primitive ------------------------------------------------------
def test_flow_layout_wraps_and_reports_an_honest_minimum() -> None:
    _app()
    host = QWidget()
    flow = FlowLayout(host, spacing=4)
    for _ in range(4):
        flow.add_widget(QPushButton("a wide enough button"))
    one = flow.itemAt(0).sizeHint().width()
    assert flow.minimumSize().width() <= one + 1
    assert flow.sizeHint().width() >= 4 * one
    assert flow.heightForWidth(one + 8) > flow.heightForWidth(4 * one + 32)


def test_flow_layout_take_at_keeps_the_line_flags_aligned() -> None:
    _app()
    host = QWidget()
    flow = FlowLayout(host)
    first, second = QPushButton("first"), QPushButton("second")
    flow.add_widget(first)
    flow.add_widget(second, starts_line=True)
    assert flow.count() == 2
    assert flow.takeAt(0) is not None
    assert flow.count() == 1
    assert flow.itemAt(0).widget() is second
    assert flow.itemAt(5) is None
    assert flow.takeAt(5) is None


def test_flow_layout_rejects_negative_spacing() -> None:
    _app()
    with pytest.raises(ValueError):
        FlowLayout(QWidget(), spacing=-1)
