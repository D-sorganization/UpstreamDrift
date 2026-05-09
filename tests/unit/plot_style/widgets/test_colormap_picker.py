"""Tests for :class:`ColormapPicker`."""

from __future__ import annotations

import pytest
from src.shared.python.plot_style.colormaps import ColormapId
from src.shared.python.plot_style.widgets.colormap_picker import ColormapPicker


def test_colormap_picker_default_initial_value() -> None:
    widget = ColormapPicker()
    assert widget.value() == ColormapId.VIRIDIS


def test_constructor_rejects_unknown_string() -> None:
    with pytest.raises(ValueError):
        ColormapPicker("not-a-real-colormap")


def test_constructor_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        ColormapPicker("")


def test_constructor_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        ColormapPicker(42)  # type: ignore[arg-type]


def test_combo_lists_all_built_in_ids() -> None:
    widget = ColormapPicker()
    items = [widget._combo.itemText(i) for i in range(widget._combo.count())]
    for cid in ColormapId:
        assert cid.value in items


def test_set_value_emits_colormap_changed(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColormapPicker(ColormapId.VIRIDIS)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.colormapChanged, timeout=500) as blocker:
        widget.set_value(ColormapId.PLASMA)
    assert blocker.args == [ColormapId.PLASMA]
    assert widget.value() == ColormapId.PLASMA


def test_set_value_string_form(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColormapPicker(ColormapId.VIRIDIS)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.colormapChanged, timeout=500) as blocker:
        widget.set_value("magma")
    assert blocker.args == [ColormapId.MAGMA]


def test_colormap_picker_set_value_no_emit_when_unchanged(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColormapPicker(ColormapId.PLASMA)
    qtbot.addWidget(widget)
    received: list[object] = []
    widget.colormapChanged.connect(received.append)
    widget.set_value(ColormapId.PLASMA)
    assert received == []


def test_set_value_round_trip(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColormapPicker()
    qtbot.addWidget(widget)
    for cid in (
        ColormapId.PLASMA,
        ColormapId.MAGMA,
        ColormapId.TURBO,
        ColormapId.VELOCITY,
        ColormapId.VIRIDIS,
    ):
        widget.set_value(cid)
        assert widget.value() == cid


def test_combo_user_interaction_emits(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = ColormapPicker(ColormapId.VIRIDIS)
    qtbot.addWidget(widget)
    plasma_idx = widget._combo.findText(ColormapId.PLASMA.value)
    with qtbot.waitSignal(widget.colormapChanged, timeout=500) as blocker:
        widget._combo.setCurrentIndex(plasma_idx)
    assert blocker.args == [ColormapId.PLASMA]


def test_refresh_after_register(qtbot) -> None:  # type: ignore[no-untyped-def]
    from src.shared.python.plot_style.colormaps import CustomColormap
    from src.shared.python.plot_style.registry import (
        register_custom_colormap,
        unregister_custom_colormap,
    )

    name = "_test_picker_cmap_xyz"
    cmap = CustomColormap(name=name, stops=((0.0, "#000000"), (1.0, "#ffffff")))
    register_custom_colormap(cmap)
    try:
        widget = ColormapPicker()
        qtbot.addWidget(widget)
        widget.refresh()
        items = [widget._combo.itemText(i) for i in range(widget._combo.count())]
        assert name in items
        widget.set_value(name)
        assert widget.value() == name
    finally:
        unregister_custom_colormap(name)
