"""Desktop presentation of the same canonical biomechanics results as the web."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

from src.shared.python.analysis.biomechanics_display import (
    biomechanics_csv,
    prepare_biomechanics_plot,
    render_biomechanics_figure,
)


class BiomechanicsWidget(QtWidgets.QWidget):
    """Selectable channels, unit controls, gap-safe plots and portable exports."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        configure: Callable[[Any], None] | None = None,
        refresh: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._configure = configure
        self._result: Mapping[str, Any] | None = None
        self._colors: dict[str, str] = {}
        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel(
            "Import a Trajectory or Result JSON to Explore Biomechanics"
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        if configure is not None:
            binding_button = QtWidgets.QPushButton("Import Model Binding")
            binding_button.clicked.connect(self._import_binding)
            controls.addWidget(binding_button)
        if refresh is not None:
            refresh_button = QtWidgets.QPushButton("Load Recorded Biomechanics")
            refresh_button.clicked.connect(refresh)
            controls.addWidget(refresh_button)
        for title, callback in (
            ("Import JSON", self._import),
            ("Channel Color", self._color),
            ("Export CSV", self._export),
        ):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.angle_unit = QtWidgets.QComboBox()
        self.angle_unit.addItems(["deg", "rad"])
        self.angle_unit.setAccessibleName("Angle Units")
        self.angle_unit.currentTextChanged.connect(self._redraw)
        controls.addWidget(self.angle_unit)
        self.channels = QtWidgets.QListWidget()
        self.channels.setAccessibleName("Biomechanics Channels")
        self.channels.setMaximumHeight(150)
        self.channels.itemChanged.connect(self._redraw)
        layout.addWidget(self.channels)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Standard toolbar exposes pan, zoom, axis customization and PNG/SVG/PDF.
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def set_result(self, result: Mapping[str, Any]) -> None:
        """Validate and display a canonical result without changing its values."""
        prepared = prepare_biomechanics_plot(result)
        self._result = result
        self.channels.blockSignals(True)
        self.channels.clear()
        for index, (name, channel) in enumerate(prepared["channels"].items()):
            item = QtWidgets.QListWidgetItem(f"{name} ({channel['unit']})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            item.setToolTip(
                f"{channel.get('definition', '')}\nFrame: {channel.get('frame', '')}"
            )
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if index < 4
                else QtCore.Qt.CheckState.Unchecked
            )
            self.channels.addItem(item)
        self.channels.blockSignals(False)
        unavailable = "; ".join(
            f"{name}: {reason}"
            for name, reason in result.get("unavailable", {}).items()
        )
        self.status.setText(f"Source: {prepared['source']}\n{unavailable}")
        self._redraw()

    def _prepared(self) -> dict[str, Any] | None:
        if self._result is None:
            return None
        selected = []
        for index in range(self.channels.count()):
            item = self.channels.item(index)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return prepare_biomechanics_plot(
            self._result, selected, self.angle_unit.currentText()
        )

    def set_status(self, message: str) -> None:
        """Set a user-facing status without exposing widget internals."""
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        self.status.setText(message)

    def _redraw(self) -> None:
        prepared = self._prepared()
        if prepared is not None:
            render_biomechanics_figure(prepared, self.figure, self._colors)
            self.canvas.draw_idle()

    def _color(self) -> None:
        item = self.channels.currentItem()
        if item is None:
            return
        color = QtWidgets.QColorDialog.getColor(parent=self)
        if color.isValid():
            self._colors[item.data(QtCore.Qt.ItemDataRole.UserRole)] = color.name()
            self._redraw()

    def _import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Biomechanics", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            if "channels" not in result:
                from src.shared.python.biomechanics.golf_trajectory import (
                    compute_golf_metrics,
                    golf_trajectory_from_dict,
                )

                result = compute_golf_metrics(
                    golf_trajectory_from_dict(result)
                ).to_dict()
            self.set_result(result)
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.status.setText(f"Cannot Import Biomechanics: {error}")

    def _import_binding(self) -> None:
        from src.shared.python.biomechanics.model_bindings import (
            model_binding_from_dict,
        )

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Model Binding", "", "JSON (*.json)"
        )
        if not path or self._configure is None:
            return
        try:
            binding = model_binding_from_dict(
                json.loads(Path(path).read_text(encoding="utf-8"))
            )
            self._configure(binding)
            self.status.setText(
                "Model Binding Loaded. Start a Fresh Recording, Then Load Recorded Biomechanics."
            )
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.status.setText(f"Cannot Configure Biomechanics: {error}")

    def _export(self) -> None:
        prepared = self._prepared()
        if prepared is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Biomechanics", "biomechanics.csv", "CSV (*.csv)"
        )
        if path:
            try:
                Path(path).write_text(biomechanics_csv(prepared), encoding="utf-8")
            except OSError as error:
                self.status.setText(f"Cannot Export Biomechanics: {error}")
