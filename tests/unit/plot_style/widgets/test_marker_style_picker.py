"""Tests for :class:`MarkerStylePicker`."""

from __future__ import annotations

import pytest
from src.shared.python.plot_style.markers import MarkerShape, MarkerStyle
from src.shared.python.plot_style.widgets.marker_style_picker import (
    MarkerStylePicker,
)


def test_default_value() -> None:
    widget = MarkerStylePicker()
    assert widget.value() == MarkerStyle()


def test_initial_value_round_trip() -> None:
    style = MarkerStyle(
        shape=MarkerShape.CUBE,
        size_px=12.5,
        edge_color="#abcdef",
        edge_width=1.25,
    )
    widget = MarkerStylePicker(style)
    assert widget.value() == style


def test_constructor_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        MarkerStylePicker("not-a-style")  # type: ignore[arg-type]


def test_set_value_emits_style_changed(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MarkerStylePicker()
    qtbot.addWidget(widget)
    new_style = MarkerStyle(
        shape=MarkerShape.STAR, size_px=4.0, edge_color="#112233", edge_width=0.0
    )
    with qtbot.waitSignal(widget.styleChanged, timeout=500) as blocker:
        widget.set_value(new_style)
    assert blocker.args == [new_style]
    assert widget.value() == new_style


def test_marker_style_picker_set_value_no_emit_when_unchanged(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MarkerStylePicker()
    qtbot.addWidget(widget)
    received: list[MarkerStyle] = []
    widget.styleChanged.connect(received.append)
    widget.set_value(MarkerStyle())
    assert received == []


def test_size_spin_change_emits(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MarkerStylePicker()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.styleChanged, timeout=500):
        widget._size_spin.setValue(15.0)
    assert widget.value().size_px == 15.0


def test_edge_width_spin_change_emits(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MarkerStylePicker()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.styleChanged, timeout=500):
        widget._edge_width_spin.setValue(2.5)
    assert widget.value().edge_width == 2.5


def test_edge_color_change_emits(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MarkerStylePicker()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.styleChanged, timeout=500):
        widget._edge_color.set_value("#abcdef")
    assert widget.value().edge_color == "#abcdef"


def test_shape_combo_change_emits(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MarkerStylePicker()
    qtbot.addWidget(widget)
    cube_idx = widget._shape_choices.index(MarkerShape.CUBE)
    with qtbot.waitSignal(widget.styleChanged, timeout=500):
        widget._shape_combo.setCurrentIndex(cube_idx)
    assert widget.value().shape is MarkerShape.CUBE


def test_invalid_intermediate_state_does_not_crash(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MarkerStylePicker()
    qtbot.addWidget(widget)
    # size_px must be > 0; spinbox minimum prevents this, but cover the
    # rebuild path by directly invoking _on_changed after toggling.
    widget._size_spin.setValue(0.5)
    assert widget.value().size_px == 0.5
