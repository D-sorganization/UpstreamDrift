"""Dockable Terrain Engine explorer for launcher-hosted terrain presets."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.launchers.startup import _get_theme_colors
from src.shared.python.physics import terrain_presets
from src.shared.python.physics.terrain import Terrain
from src.shared.python.physics.terrain_presets import (
    ENVIRONMENT_PRESETS,
    build_environment_preset,
)
from src.shared.python.theme.layout_metrics import LayoutMetrics


def _color(colors: Any, attr: str, fallback: str) -> str:
    if isinstance(colors, dict):
        return str(colors.get(attr, fallback))
    return str(getattr(colors, attr, fallback))


class TerrainExplorerWidget(QWidget):
    """Terrain preset browser with elevation and material query tools.

    Design by Contract:
        Precondition: presets must be non-empty and every preset must have a builder.
        Postcondition: a terrain is loaded before query controls are enabled.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if not ENVIRONMENT_PRESETS:
            raise ValueError("at least one terrain preset must be available")
        self._terrain: Terrain | None = None
        self._build_ui()
        self._on_preset_changed()
        self._load_selected_preset()

    def _build_ui(self) -> None:
        colors = _get_theme_colors()
        bg = _color(colors, "surface_primary", "#1f2329")
        panel = _color(colors, "surface_secondary", "#252a31")
        border = _color(colors, "border_default", "#3a414a")
        text = _color(colors, "text_primary", "#f0f3f6")
        muted = _color(colors, "text_secondary", "#a8b0bb")

        self.setObjectName("TerrainExplorerWidget")
        self.setStyleSheet(f"""
            QWidget#TerrainExplorerWidget {{
                background: {bg};
                color: {text};
            }}
            QGroupBox {{
                border: 1px solid {border};
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                background: {panel};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: {muted};
            }}
            QTableWidget {{
                background: {panel};
                color: {text};
                gridline-color: {border};
                border: 1px solid {border};
                border-radius: 6px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(
            LayoutMetrics.MARGIN_PAGE,
            LayoutMetrics.MARGIN_PAGE,
            LayoutMetrics.MARGIN_PAGE,
            LayoutMetrics.MARGIN_PAGE,
        )
        root.setSpacing(LayoutMetrics.SPACING_MD)

        title = QLabel("Terrain Engine")
        title.setObjectName("TerrainTitle")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {text};")
        root.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        root.addWidget(splitter, 1)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(LayoutMetrics.SPACING_MD)

        preset_group = QGroupBox("Preset")
        preset_form = QFormLayout(preset_group)
        self.preset_combo = QComboBox()
        for name, info in ENVIRONMENT_PRESETS.items():
            self.preset_combo.addItem(str(info["description"]), name)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_form.addRow("Environment", self.preset_combo)

        self.width_spin = self._make_spin(1.0, 1000.0, " m")
        self.length_spin = self._make_spin(1.0, 2000.0, " m")
        self.slope_spin = self._make_spin(-20.0, 20.0, " deg")
        self.direction_spin = self._make_spin(0.0, 360.0, " deg")
        preset_form.addRow("Width", self.width_spin)
        preset_form.addRow("Length", self.length_spin)
        preset_form.addRow("Slope", self.slope_spin)
        preset_form.addRow("Direction", self.direction_spin)

        load_btn = QPushButton("Load Terrain")
        load_btn.clicked.connect(self._load_selected_preset)
        preset_form.addRow(load_btn)
        controls_layout.addWidget(preset_group)

        query_group = QGroupBox("Query")
        query_form = QFormLayout(query_group)
        self.query_x_spin = self._make_spin(0.0, 2000.0, " m")
        self.query_y_spin = self._make_spin(0.0, 2000.0, " m")
        query_form.addRow("X", self.query_x_spin)
        query_form.addRow("Y", self.query_y_spin)
        self.query_btn = QPushButton("Query Surface")
        self.query_btn.clicked.connect(self._query_surface)
        query_form.addRow(self.query_btn)
        self.query_result = QLabel("")
        self.query_result.setWordWrap(True)
        query_form.addRow(self.query_result)
        controls_layout.addWidget(query_group)
        controls_layout.addStretch()
        splitter.addWidget(controls)

        self._set_query_controls_enabled(False)

        preview = QFrame()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(LayoutMetrics.SPACING_MD)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        preview_layout.addWidget(self.summary)

        self.sample_table = QTableWidget(0, 4)
        self.sample_table.setHorizontalHeaderLabels(["X", "Y", "Elevation", "Terrain"])
        self.sample_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sample_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        preview_layout.addWidget(self.sample_table, 1)

        splitter.addWidget(preview)
        splitter.setSizes([360, 760])

    def _set_query_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable query surface controls based on terrain loaded state."""
        if hasattr(self, "query_btn"):
            self.query_btn.setEnabled(enabled)
        if hasattr(self, "query_x_spin"):
            self.query_x_spin.setEnabled(enabled)
        if hasattr(self, "query_y_spin"):
            self.query_y_spin.setEnabled(enabled)

    def _show_error(self, title: str, exc: Exception) -> None:
        """Log exception and display a user-friendly message/dialog instead of aborting."""
        logger.exception("Terrain Engine error: %s", exc)
        msg = f"{title}: {exc}"
        if hasattr(self, "query_result"):
            self.query_result.setText(msg)
        if hasattr(self, "summary") and not self.summary.text():
            self.summary.setText(msg)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            try:
                QMessageBox.warning(self, title, f"{title}:\n\n{exc}")
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _make_spin(minimum: float, maximum: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSuffix(suffix)
        return spin

    def _on_preset_changed(self) -> None:
        try:
            info = ENVIRONMENT_PRESETS[self._selected_preset()]
            self.width_spin.setValue(float(info["width"]))
            self.length_spin.setValue(float(info["length"]))
            self.query_x_spin.setMaximum(float(info["width"]))
            self.query_y_spin.setMaximum(float(info["length"]))
            self.query_x_spin.setValue(float(info["width"]) / 2)
            self.query_y_spin.setValue(float(info["length"]) / 2)
        except (KeyError, ValueError) as exc:
            self._show_error("Preset Error", exc)

    def _selected_preset(self) -> str:
        preset = self.preset_combo.currentData()
        if not isinstance(preset, str) or not preset:
            raise ValueError("selected terrain preset must be a non-empty string")
        return preset

    def _load_selected_preset(self) -> None:
        try:
            preset = self._selected_preset()
            terrain = terrain_presets.build_environment_preset(
                preset,
                width=self.width_spin.value(),
                length=self.length_spin.value(),
                slope=self.slope_spin.value(),
                direction=self.direction_spin.value(),
            )
            if terrain is None or not terrain.name:
                raise ValueError(
                    f"Preset '{preset}' produced an invalid terrain instance"
                )
            self._terrain = terrain
            self._set_query_controls_enabled(True)
            self._refresh_summary()
            self._populate_samples()
            self._query_surface()
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            self._set_query_controls_enabled(False)
            self._show_error("Failed to Load Terrain", exc)

    def _refresh_summary(self) -> None:
        terrain = self._require_terrain()
        self.summary.setText(
            f"{terrain.name}: {terrain.elevation.width:.1f} m x "
            f"{terrain.elevation.length:.1f} m, "
            f"{terrain.elevation.resolution:.2f} m resolution, "
            f"{len(terrain.patches)} patches, {len(terrain.regions)} regions."
        )

    def _populate_samples(self) -> None:
        terrain = self._require_terrain()
        width = terrain.elevation.width
        length = terrain.elevation.length
        points = [
            (width * 0.25, length * 0.25),
            (width * 0.50, length * 0.50),
            (width * 0.75, length * 0.75),
            (width * 0.50, max(0.0, length - terrain.elevation.resolution)),
        ]
        self.sample_table.setRowCount(0)
        for row, (x, y) in enumerate(points):
            self.sample_table.insertRow(row)
            elevation = terrain.elevation.get_elevation(x, y)
            terrain_type = terrain.get_terrain_type(x, y).name.lower()
            for col, value in enumerate(
                [f"{x:.1f}", f"{y:.1f}", f"{elevation:.3f}", terrain_type]
            ):
                self.sample_table.setItem(row, col, QTableWidgetItem(value))
        self.sample_table.resizeColumnsToContents()

    def _query_surface(self) -> None:
        try:
            terrain = self._require_terrain()
            x = self.query_x_spin.value()
            y = self.query_y_spin.value()
            elevation = terrain.elevation.get_elevation(x, y)
            slope = terrain.elevation.get_slope_angle(x, y)
            terrain_type = terrain.get_terrain_type(x, y).name.lower()
            material = terrain.get_material(x, y)
            self.query_result.setText(
                f"{terrain_type} at {elevation:.3f} m, "
                f"{slope:.2f} deg slope, friction {material.friction_coefficient:.2f}, "
                f"rolling resistance {material.rolling_resistance:.3f}."
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            self._show_error("Terrain Query Error", exc)

    def _require_terrain(self) -> Terrain:
        if self._terrain is None:
            raise RuntimeError("terrain must be loaded before querying")
        return self._terrain


class _EmbedAdapter:
    """Embed adapter for the Terrain Engine explorer."""

    tool_id = "terrain_engine"

    def __init__(self) -> None:
        self._widget: TerrainExplorerWidget | None = None

    def embed_capabilities(self) -> Any:
        from src.shared.python.launcher_embed import EmbedCapabilities

        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(800, 480),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        self._widget = TerrainExplorerWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        self._widget = None

    def is_dirty(self) -> bool:
        return False


def _register() -> None:
    try:
        from src.shared.python.launcher_embed import register_embeddable_tool

        register_embeddable_tool(_EmbedAdapter())
    except Exception:  # noqa: BLE001
        logger.warning("terrain_engine: EmbeddableTool registration failed")


_register()


def get_dockable_ui() -> TerrainExplorerWidget:
    """Return the widget hosted by the launcher tab system."""
    return TerrainExplorerWidget()
