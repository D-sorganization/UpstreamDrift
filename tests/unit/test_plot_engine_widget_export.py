"""Tests for PlotWidget export wiring (Issue #8828).

Covers end-to-end wiring: PlotWidget._export_plot now routes through
plotting.export.export_figure so ExportConfig.include_metadata actually
does something for the generic dashboard export path, and so attached
PlotIdentity (engine/model/run) is embedded in the saved file's metadata.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import pytest
from PyQt6.QtWidgets import QApplication
from src.shared.python.plot_engine.pyqt6_widget import PlotWidget
from src.shared.python.plot_engine.specs import PlotSpec, SeriesData
from src.shared.python.plotting.identity import PlotIdentity


@pytest.fixture
def app(qapp) -> QApplication:
    return qapp


def _make_spec(title: str = "Joint Positions") -> PlotSpec:
    return PlotSpec(
        title=title,
        series=[SeriesData(name="s1", x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 4.0])],
    )


@pytest.mark.unit
class TestPlotWidgetIdentity:
    def test_default_identity_is_none(self, app) -> None:
        widget = PlotWidget()
        assert widget.get_identity() is None

    def test_set_identity_roundtrip(self, app) -> None:
        widget = PlotWidget()
        identity = PlotIdentity(engine="mujoco", model="golfer_v3")
        widget.set_identity(identity)
        assert widget.get_identity() is identity


@pytest.mark.unit
class TestPlotWidgetExportWiring:
    """(b)-adjacent: the generic dashboard export path now embeds metadata."""

    def test_export_plot_writes_png_with_identity_metadata(
        self, app, tmp_path: Path
    ) -> None:
        from PIL import Image

        widget = PlotWidget()
        widget.set_spec(_make_spec())
        widget.set_identity(PlotIdentity(engine="mujoco", model="golfer_v3"))

        out_path = tmp_path / "joint_positions.png"
        with patch(
            "src.shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=(str(out_path), "PNG Files (*.png)"),
        ):
            widget._format_combo.setCurrentText("PNG")
            widget._export_plot()

        assert out_path.exists()
        info = Image.open(out_path).info
        assert info.get("engine") == "mujoco"
        assert info.get("model") == "golfer_v3"

    def test_export_plot_without_identity_still_embeds_timestamp(
        self, app, tmp_path: Path
    ) -> None:
        from PIL import Image

        widget = PlotWidget()
        widget.set_spec(_make_spec())

        out_path = tmp_path / "no_identity.png"
        with patch(
            "src.shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=(str(out_path), "PNG Files (*.png)"),
        ):
            widget._format_combo.setCurrentText("PNG")
            widget._export_plot()

        assert out_path.exists()
        info = Image.open(out_path).info
        assert "Creation Time" in info
        assert "engine" not in info

    def test_export_plot_default_filename_uses_spec_title(
        self, app, tmp_path: Path
    ) -> None:
        widget = PlotWidget()
        widget.set_spec(_make_spec(title="Custom Title"))

        with patch(
            "src.shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ) as mock_dialog:
            widget._format_combo.setCurrentText("PNG")
            widget._export_plot()
            args, _kwargs = mock_dialog.call_args
            assert "Custom Title" in args[2]
