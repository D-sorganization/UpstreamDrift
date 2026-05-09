"""plot_style integration tests for the C3D viewer's 2D + 3D tabs.

Covers:

* Smoke-boot of MarkerPlotTab, AnalogPlotTab, Viewer3DTab under
  ``QT_QPA_PLATFORM=offscreen``.
* Programmatic style change updates the renderer's artist.
* Persistence round-trip through ``PlotStyleSet.save/load``.
* DataChannelEditor wiring on Viewer3DTab triggers DataDrivenColor
  recompute.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

pytest.importorskip("PyQt6")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_model(
    marker_names: list[str],
    n_frames: int = 60,
    point_rate: float = 100.0,
    include_analog: bool = False,
):
    """Build a deterministic ``C3DDataModel`` for tests."""
    from src.apps.core.models import (  # type: ignore
        AnalogData,
        C3DDataModel,
        MarkerData,
    )

    point_time = np.arange(n_frames, dtype=float) / point_rate
    markers: dict[str, MarkerData] = {}
    rng = np.random.default_rng(42)
    for i, name in enumerate(marker_names):
        phase = 2 * np.pi * (i + 1) / max(1, len(marker_names))
        pos = np.column_stack(
            [
                np.cos(point_time * 2.0 + phase)
                + rng.normal(scale=0.005, size=n_frames),
                np.sin(point_time * 2.0 + phase)
                + rng.normal(scale=0.005, size=n_frames),
                point_time * 0.1 + i * 0.05,
            ]
        )
        markers[name] = MarkerData(name=name, position=pos)

    analog: dict[str, AnalogData] = {}
    if include_analog:
        analog["Fz1"] = AnalogData(
            name="Fz1", values=np.linspace(0.0, 1.0, n_frames), unit="N"
        )

    return C3DDataModel(
        filepath="synthetic.c3d",
        markers=markers,
        analog=analog,
        point_rate=point_rate,
        analog_rate=point_rate if include_analog else 0.0,
        point_time=point_time,
        analog_time=point_time if include_analog else None,
        metadata={},
        events=[],
    )


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


@pytest.fixture()
def isolated_persist(tmp_path: Path, monkeypatch):
    """Redirect the persist file into an isolated tmp directory."""
    target = tmp_path / "c3d_viewer_plot_styles.json"
    from src.apps.ui.tabs import _plot_style_helpers as helpers  # type: ignore

    monkeypatch.setattr(helpers, "PERSIST_FILE", target)
    return target


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_marker_plot_tab_boots(qt_app, isolated_persist) -> None:
    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore

    tab = MarkerPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0", "m1", "m2"]))
    assert tab.list_markers.count() == 3
    assert tab._current_marker == "m0"
    assert tab._current_handle is not None


def test_analog_plot_tab_boots(qt_app, isolated_persist) -> None:
    from src.apps.ui.tabs.analog_plot_tab import AnalogPlotTab  # type: ignore

    tab = AnalogPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0"], include_analog=True))
    assert tab.list_analog.count() == 1
    assert tab._current_channel == "Fz1"
    assert tab._current_handle is not None


def test_viewer_3d_tab_boots(qt_app, isolated_persist) -> None:
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    tab = Viewer3DTab()
    tab.update_from_model(_make_synthetic_model(["m0", "m1", "m2"]))
    # Force a deterministic single selection so the scene rebuilds.
    tab.list_markers_3d.clearSelection()
    item = tab.list_markers_3d.item(0)
    assert item is not None
    item.setSelected(True)
    assert tab._marker_style_renderer is not None
    assert tab._marker_style_handle is not None


# ---------------------------------------------------------------------------
# Programmatic style change updates the artist
# ---------------------------------------------------------------------------


def test_marker_apply_style_updates_artist(qt_app, isolated_persist) -> None:
    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        StaticColor,
    )

    tab = MarkerPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0"]))
    handle = tab._current_handle
    assert handle is not None
    new_style = MarkerStyle(
        shape=MarkerShape.DIAMOND,
        size_px=12.0,
        edge_color="#112233",
        edge_width=1.5,
        fill_color=StaticColor("#00ff00"),
        opacity=0.8,
    )
    tab.apply_style("m0", new_style)
    record = tab._renderer._handles[handle]  # type: ignore[attr-defined]
    assert record.style == new_style


def test_analog_apply_style_updates_artist(qt_app, isolated_persist) -> None:
    from src.apps.ui.tabs.analog_plot_tab import AnalogPlotTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        StaticColor,
    )

    tab = AnalogPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0"], include_analog=True))
    handle = tab._current_handle
    assert handle is not None
    new_style = MarkerStyle(
        shape=MarkerShape.STAR,
        size_px=15.0,
        edge_color="#ff00ff",
        fill_color=StaticColor("#00aaff"),
    )
    tab.apply_style("Fz1", new_style)
    record = tab._renderer._handles[handle]  # type: ignore[attr-defined]
    assert record.style == new_style


def test_viewer_3d_apply_marker_group_style(qt_app, isolated_persist) -> None:
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        StaticColor,
    )

    tab = Viewer3DTab()
    tab.update_from_model(_make_synthetic_model(["m0", "m1", "m2"]))
    tab.list_markers_3d.clearSelection()
    item = tab.list_markers_3d.item(0)
    assert item is not None
    item.setSelected(True)
    handle = tab._marker_style_handle
    assert handle is not None
    style = MarkerStyle(
        shape=MarkerShape.CUBE,
        size_px=10.0,
        edge_color="#000000",
        fill_color=StaticColor("#abcdef"),
    )
    # Make sure default group is selected.
    tab.combo_marker_group.setCurrentText("default")
    tab.apply_marker_group_style("default", style)
    record = tab._marker_style_renderer._handles[handle]  # type: ignore[attr-defined]
    assert record.style == style


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_persistence_roundtrip(qt_app, isolated_persist) -> None:
    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        PlotStyleSet,
        StaticColor,
    )

    tab = MarkerPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0"]))
    style = MarkerStyle(
        shape=MarkerShape.DIAMOND,
        size_px=11.0,
        edge_color="#101010",
        edge_width=0.8,
        fill_color=StaticColor("#abcdef"),
        opacity=0.9,
    )
    tab.apply_style("m0", style)
    # Force-flush the debounced save.
    tab._persistence.save_now()
    assert isolated_persist.is_file()

    loaded = PlotStyleSet.load(isolated_persist)
    matched = [e for e in loaded.entries if e.target == "marker:m0"]
    assert len(matched) == 1
    assert matched[0].style == style

    # Reload into a fresh tab and confirm the style is rehydrated.
    tab2 = MarkerPlotTab()
    assert tab2._persistence.get("m0") == style


def test_persistence_isolates_other_tabs(qt_app, isolated_persist) -> None:
    """Saving from one tab must preserve another tab's entries."""
    from src.apps.ui.tabs.analog_plot_tab import AnalogPlotTab  # type: ignore
    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        PlotStyleSet,
        StaticColor,
    )

    style_a = MarkerStyle(
        shape=MarkerShape.STAR,
        size_px=8.0,
        fill_color=StaticColor("#ffaa00"),
    )
    style_b = MarkerStyle(
        shape=MarkerShape.CROSS,
        size_px=9.0,
        fill_color=StaticColor("#00ffaa"),
    )

    marker_tab = MarkerPlotTab()
    marker_tab.update_from_model(_make_synthetic_model(["mx"]))
    marker_tab.apply_style("mx", style_a)
    marker_tab._persistence.save_now()

    analog_tab = AnalogPlotTab()
    analog_tab.update_from_model(_make_synthetic_model(["mx"], include_analog=True))
    analog_tab.apply_style("Fz1", style_b)
    analog_tab._persistence.save_now()

    loaded = PlotStyleSet.load(isolated_persist)
    targets = {e.target for e in loaded.entries}
    assert "marker:mx" in targets
    assert "channel:Fz1" in targets


# ---------------------------------------------------------------------------
# DataChannelEditor wiring on viewer_3d_tab
# ---------------------------------------------------------------------------


def test_viewer_3d_data_channel_editor_drives_data_driven_color(
    qt_app, isolated_persist
) -> None:
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore
    from src.shared.python.plot_style import DataChannel  # type: ignore

    tab = Viewer3DTab()
    tab.update_from_model(_make_synthetic_model(["m0", "m1", "m2"], n_frames=40))
    tab.list_markers_3d.clearSelection()
    for i in range(tab.list_markers_3d.count()):
        item = tab.list_markers_3d.item(i)
        assert item is not None
        item.setSelected(True)
    assert tab._marker_style_handle is not None

    speed = DataChannel(
        name="speed",
        values=np.linspace(0.0, 5.0, 40, dtype=float),
        unit="m/s",
    )
    force = DataChannel(
        name="force",
        values=np.linspace(0.0, 100.0, 40, dtype=float),
        unit="N",
    )
    tab.install_data_channel_editor((speed, force))
    assert tab.has_data_channel_editor is True
    assert tab.active_color_uses_data_driven is True

    # Range change recomputes color.
    tab._on_color_range_changed(0.5, 4.5)
    assert tab._color_range == (0.5, 4.5)
    assert tab.active_color_uses_data_driven is True

    # Channel change.
    tab._on_color_channel_changed(force)
    assert tab._color_channel is force
    assert tab.active_color_uses_data_driven is True


# ---------------------------------------------------------------------------
# default_style_for fallback
# ---------------------------------------------------------------------------


def test_default_style_for_unknown_returns_bare_style() -> None:
    from src.apps.ui.tabs._plot_style_helpers import default_style_for  # type: ignore
    from src.shared.python.plot_style import MarkerStyle  # type: ignore

    style = default_style_for("entirely-unknown-marker-name")
    assert isinstance(style, MarkerStyle)


def test_default_style_for_known_preset_entry() -> None:
    from src.apps.ui.tabs._plot_style_helpers import default_style_for  # type: ignore

    # The built-in default preset ships a "ball" entry.
    style = default_style_for("ball")
    # Different from the bare default because the preset overrides size.
    assert style.size_px != 6.0 or style.fill_color is not None


# ---------------------------------------------------------------------------
# Dialog plumbing: monkey-patch QDialog.exec to drive the Style… button.
# ---------------------------------------------------------------------------


def test_marker_style_button_opens_dialog_and_applies(
    qt_app, isolated_persist, monkeypatch
) -> None:
    from PyQt6 import QtWidgets

    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        StaticColor,
    )

    accepted_style = MarkerStyle(
        shape=MarkerShape.PLUS,
        size_px=14.0,
        edge_color="#abcdef",
        fill_color=StaticColor("#101010"),
    )

    def fake_exec(self) -> int:
        # Simulate the user changing the style and clicking OK.
        # Find the embedded MarkerStylePicker and seed it.
        from src.shared.python.plot_style.widgets.marker_style_picker import (
            MarkerStylePicker,
        )

        for child in self.findChildren(MarkerStylePicker):
            child.set_value(accepted_style)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec)

    tab = MarkerPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0"]))
    tab._on_style_clicked()
    assert tab._persistence.get("m0") == accepted_style


def test_analog_style_button_opens_dialog_and_applies(
    qt_app, isolated_persist, monkeypatch
) -> None:
    from PyQt6 import QtWidgets

    from src.apps.ui.tabs.analog_plot_tab import AnalogPlotTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        StaticColor,
    )

    accepted_style = MarkerStyle(
        shape=MarkerShape.DIAMOND,
        size_px=10.0,
        fill_color=StaticColor("#cccccc"),
    )

    def fake_exec(self) -> int:
        from src.shared.python.plot_style.widgets.marker_style_picker import (
            MarkerStylePicker,
        )

        for child in self.findChildren(MarkerStylePicker):
            child.set_value(accepted_style)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec)

    tab = AnalogPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0"], include_analog=True))
    tab._on_style_clicked()
    assert tab._persistence.get("Fz1") == accepted_style


def test_marker_style_dialog_cancel_no_apply(
    qt_app, isolated_persist, monkeypatch
) -> None:
    from PyQt6 import QtWidgets

    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore

    monkeypatch.setattr(
        QtWidgets.QDialog,
        "exec",
        lambda self: QtWidgets.QDialog.DialogCode.Rejected,
    )

    tab = MarkerPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0"]))
    tab._on_style_clicked()
    # Cancellation must not write anything to disk-side state.
    assert tab._persistence.get("m0") is None


def test_marker_plot_tab_changes_marker_then_apply(qt_app, isolated_persist) -> None:
    """Selecting a different marker rewires the renderer handle correctly."""
    from src.apps.ui.tabs.marker_plot_tab import MarkerPlotTab  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerShape,
        MarkerStyle,
        StaticColor,
    )

    tab = MarkerPlotTab()
    tab.update_from_model(_make_synthetic_model(["m0", "m1"]))
    tab.list_markers.setCurrentRow(1)
    assert tab._current_marker == "m1"
    style = MarkerStyle(
        shape=MarkerShape.POINT,
        size_px=6.0,
        fill_color=StaticColor("#aabbcc"),
    )
    tab.apply_style("m1", style)
    handle = tab._current_handle
    assert handle is not None
    record = tab._renderer._handles[handle]  # type: ignore[attr-defined]
    assert record.style == style


def test_persistence_request_save_is_debounced(qt_app, isolated_persist) -> None:
    """Two rapid request_save calls coalesce into a single deferred save."""
    from src.apps.ui.tabs._plot_style_helpers import StylePersistence  # type: ignore
    from src.shared.python.plot_style import (  # type: ignore
        MarkerStyle,
        StaticColor,
    )

    p = StylePersistence(target_prefix="test:", path=isolated_persist)
    p.set("alpha", MarkerStyle(fill_color=StaticColor("#ff0000")))
    p.request_save()
    p.request_save()  # debounced — should not stack.
    assert p._save_pending is True  # type: ignore[attr-defined]
    p.save_now()
    assert p._save_pending is False  # type: ignore[attr-defined]
    assert isolated_persist.is_file()
