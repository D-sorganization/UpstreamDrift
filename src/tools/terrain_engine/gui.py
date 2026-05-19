"""Dockable Terrain Engine explorer for launcher-hosted terrain presets."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.launchers.startup import _get_theme_colors
from src.shared.python.physics.terrain import Terrain
from src.shared.python.physics.terrain_presets import (
    ENVIRONMENT_PRESETS,
    build_environment_preset,
)
from src.shared.python.theme.style_constants import Styles


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
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
        )
        root.setSpacing(Styles.SPACING_MD)

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
        controls_layout.setSpacing(Styles.SPACING_MD)

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
        query_btn = QPushButton("Query Surface")
        query_btn.clicked.connect(self._query_surface)
        query_form.addRow(query_btn)
        self.query_result = QLabel("")
        self.query_result.setWordWrap(True)
        query_form.addRow(self.query_result)
        controls_layout.addWidget(query_group)
        controls_layout.addStretch()
        splitter.addWidget(controls)

        preview = QFrame()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(Styles.SPACING_MD)

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

    @staticmethod
    def _make_spin(minimum: float, maximum: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSuffix(suffix)
        return spin

    def _on_preset_changed(self) -> None:
        info = ENVIRONMENT_PRESETS[self._selected_preset()]
        self.width_spin.setValue(float(info["width"]))
        self.length_spin.setValue(float(info["length"]))
        self.query_x_spin.setMaximum(float(info["width"]))
        self.query_y_spin.setMaximum(float(info["length"]))
        self.query_x_spin.setValue(float(info["width"]) / 2)
        self.query_y_spin.setValue(float(info["length"]) / 2)

    def _selected_preset(self) -> str:
        preset = self.preset_combo.currentData()
        if not isinstance(preset, str) or not preset:
            raise ValueError("selected terrain preset must be a non-empty string")
        return preset

    def _load_selected_preset(self) -> None:
        preset = self._selected_preset()
        self._terrain = build_environment_preset(
            preset,
            width=self.width_spin.value(),
            length=self.length_spin.value(),
            slope=self.slope_spin.value(),
            direction=self.direction_spin.value(),
        )
        assert self._terrain is not None and self._terrain.name
        self._refresh_summary()
        self._populate_samples()
        self._query_surface()

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

    def _require_terrain(self) -> Terrain:
        if self._terrain is None:
            raise RuntimeError("terrain must be loaded before querying")
        return self._terrain


def get_dockable_ui() -> TerrainExplorerWidget:
    """Return the widget hosted by the launcher tab system."""
    return TerrainExplorerWidget()
