"""Tests for :class:`ColorPicker`."""

from __future__ import annotations

import pytest
from src.shared.python.plot_style.widgets.color_picker import ColorPicker


def test_default_initial_value() -> None:
    widget = ColorPicker()
    assert widget.value() == "#1f77b4"


def test_constructor_rejects_invalid_color() -> None:
    with pytest.raises(ValueError):
        ColorPicker("not-a-color")


def test_constructor_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        ColorPicker("")


def test_set_value_emits_color_changed(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColorPicker("#ffffff")
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.colorChanged, timeout=500) as blocker:
        widget.set_value("#abcdef")
    assert blocker.args == ["#abcdef"]
    assert widget.value() == "#abcdef"


def test_set_value_no_emit_when_unchanged(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColorPicker("#abcdef")
    qtbot.addWidget(widget)
    received: list[str] = []
    widget.colorChanged.connect(received.append)
    widget.set_value("#abcdef")
    assert received == []


def test_set_value_normalises_named_color(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColorPicker("#ffffff")
    qtbot.addWidget(widget)
    received: list[str] = []
    widget.colorChanged.connect(received.append)
    widget.set_value("red")
    assert widget.value() == "#ff0000"
    assert received == ["#ff0000"]


def test_set_value_round_trip(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColorPicker()
    qtbot.addWidget(widget)
    for color in ("#000000", "#abcdef", "#123456"):
        widget.set_value(color)
        assert widget.value() == color


def test_set_value_rejects_invalid(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColorPicker("#ffffff")
    qtbot.addWidget(widget)
    with pytest.raises(ValueError):
        widget.set_value("not-a-color")
    assert widget.value() == "#ffffff"


def test_invalid_text_in_edit_rolls_back(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColorPicker("#abcdef")
    qtbot.addWidget(widget)
    widget._edit.setText("not-a-color")
    received: list[str] = []
    widget.colorChanged.connect(received.append)
    # Simulate the user finishing editing.
    widget._on_edit_finished()
    assert widget.value() == "#abcdef"
    assert widget._edit.text() == "#abcdef"
    assert received == []


def test_valid_text_in_edit_emits(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColorPicker("#abcdef")
    qtbot.addWidget(widget)
    widget._edit.setText("#112233")
    with qtbot.waitSignal(widget.colorChanged, timeout=500) as blocker:
        widget._on_edit_finished()
    assert blocker.args == ["#112233"]
