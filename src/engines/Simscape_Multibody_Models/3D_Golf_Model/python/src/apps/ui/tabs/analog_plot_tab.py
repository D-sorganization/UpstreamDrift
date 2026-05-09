"""Analog-data plot tab for the 3-D Golf Model GUI.

Per-channel styling is wired through :mod:`plot_style` — see the
docstring of :mod:`marker_plot_tab` for the integration contract.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6 import QtWidgets

from src.shared.python.plot_style import (
    MarkerStyle,
    MatplotlibMarkerRenderer,
)

from ...core.models import C3DDataModel
from ..widgets.mpl_canvas import MplCanvas
from ._plot_style_helpers import StylePersistence, default_style_for
from .marker_plot_tab import _open_style_dialog

logger = logging.getLogger(__name__)


class AnalogPlotTab(QtWidgets.QWidget):
    """Analog channel plotting tab with per-channel style customisation."""

    PERSIST_PREFIX = "channel:"

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None

        self._renderer: MatplotlibMarkerRenderer | None = None
        self._current_handle: str | None = None
        self._current_channel: str | None = None
        self._persistence = StylePersistence(target_prefix=self.PERSIST_PREFIX)
        self._persistence.load()

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        left_panel = QtWidgets.QVBoxLayout()
        self.list_analog = QtWidgets.QListWidget()
        self.list_analog.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_analog.itemSelectionChanged.connect(self.update_plot)
        left_panel.addWidget(QtWidgets.QLabel("Analog channels:"))
        left_panel.addWidget(self.list_analog)

        self.btn_style = QtWidgets.QPushButton("Style…")
        self.btn_style.setObjectName("analog_style_button")
        self.btn_style.clicked.connect(self._on_style_clicked)
        left_panel.addWidget(self.btn_style)

        layout.addLayout(left_panel, 1)

        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_analog = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_analog)

        layout.addLayout(right_panel, 3)

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        self.model = model
        self.list_analog.clear()

        if model is None:
            self.canvas_analog.clear_axes()
            return

        for name in model.analog_names():
            self.list_analog.addItem(name)

        if model.analog_names():
            self.list_analog.setCurrentRow(0)

    def update_plot(self) -> None:
        """Update the analog plot based on selected channel."""
        if self.model is None:
            return

        selected_items = self.list_analog.selectedItems()
        if not selected_items:
            self.canvas_analog.clear_axes()
            self._current_handle = None
            self._current_channel = None
            return

        name = selected_items[0].text()
        channel = self.model.analog.get(name)
        if channel is None or self.model.analog_time is None:
            self.canvas_analog.clear_axes()
            self._current_handle = None
            self._current_channel = None
            return

        t = self.model.analog_time
        values = channel.values

        self.canvas_analog.fig.clear()
        ax = self.canvas_analog.add_subplot(111)
        ax.plot(t, values, label=name)
        unit = f" ({channel.unit})" if channel.unit else ""
        ax.set_ylabel(f"Value{unit}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Analog channel: {name}")
        ax.grid(True)
        ax.legend()

        self._renderer = MatplotlibMarkerRenderer(ax)
        first_v = float(values[0]) if values.size > 0 else 0.0
        glyph_pos = np.asarray([[float(t[0]), first_v]], dtype=float)
        style = self._persistence.get(name) or default_style_for(name)
        try:
            self._current_handle = self._renderer.add_markers(glyph_pos, style, name)
        except (TypeError, ValueError) as exc:
            logger.warning("could not register channel glyph for %s: %s", name, exc)
            self._current_handle = None
        self._current_channel = name

        self.canvas_analog.fig.tight_layout()
        self.canvas_analog.draw()  # type: ignore

    def apply_style(self, name: str, style: MarkerStyle) -> None:
        """Programmatic entry point used by tests and the dialog flow."""
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        self._persistence.set(name, style)
        if (
            self._renderer is not None
            and self._current_handle is not None
            and self._current_channel == name
        ):
            try:
                self._renderer.update_style(self._current_handle, style)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("update_style failed: %s", exc)
            self.canvas_analog.draw_idle()
        self._persistence.request_save()

    def _on_style_clicked(self) -> None:
        if self._current_channel is None:
            return
        current = self._persistence.get(self._current_channel) or default_style_for(
            self._current_channel
        )
        new_style = _open_style_dialog(
            self, current, f"Style — {self._current_channel}"
        )
        if new_style is None:
            return
        self.apply_style(self._current_channel, new_style)
