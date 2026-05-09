"""Marker-trajectory plot tab for the 3-D Golf Model GUI.

Per-marker styling is wired through :mod:`plot_style`:

* a "Style…" button opens a :class:`MarkerStylePicker` dialog,
* the chosen :class:`MarkerStyle` is applied via
  :meth:`MatplotlibMarkerRenderer.update_style` to the marker's artist,
* user customisations persist to
  ``~/.golf_modeling_suite/c3d_viewer_plot_styles.json`` (debounced
  300 ms via :func:`QTimer.singleShot`).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6 import QtWidgets

from src.shared.python.plot_style import (
    MarkerStyle,
    MatplotlibMarkerRenderer,
)
from src.shared.python.qt_utils.wheel_event_filter import suppress_wheel_on_widgets

from ...core.models import C3DDataModel
from ..widgets.mpl_canvas import MplCanvas
from ._plot_style_helpers import StylePersistence, default_style_for

logger = logging.getLogger(__name__)


def _open_style_dialog(
    parent: QtWidgets.QWidget,
    initial: MarkerStyle,
    title: str,
) -> MarkerStyle | None:
    """Open a modal :class:`MarkerStylePicker` and return the chosen style."""
    # Lazy import — the widgets package depends on PyQt6 and we want the
    # tab module to import cleanly even when the optional widgets surface
    # isn't available (e.g. during static type checks).
    from src.shared.python.plot_style.widgets.marker_style_picker import (
        MarkerStylePicker,
    )

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QtWidgets.QVBoxLayout(dialog)
    picker = MarkerStylePicker(initial=initial, parent=dialog)
    layout.addWidget(picker)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return picker.value()


class MarkerPlotTab(QtWidgets.QWidget):
    """Marker 2D plotting tab with per-marker style customisation."""

    PERSIST_PREFIX = "marker:"

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None

        # plot_style integration state.
        self._renderer: MatplotlibMarkerRenderer | None = None
        self._current_handle: str | None = None
        self._current_marker: str | None = None
        self._persistence = StylePersistence(target_prefix=self.PERSIST_PREFIX)
        self._persistence.load()

        self._init_ui()

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        left_panel = QtWidgets.QVBoxLayout()
        self.list_markers = QtWidgets.QListWidget()
        self.list_markers.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_markers.itemSelectionChanged.connect(self.update_plot)
        left_panel.addWidget(QtWidgets.QLabel("Markers:"))
        left_panel.addWidget(self.list_markers)

        self.combo_component = QtWidgets.QComboBox()
        self.combo_component.addItems(["All (X/Y/Z)", "X", "Y", "Z", "Speed magnitude"])
        self.combo_component.currentIndexChanged.connect(self.update_plot)
        left_panel.addWidget(QtWidgets.QLabel("Component:"))
        left_panel.addWidget(self.combo_component)

        suppress_wheel_on_widgets(self.combo_component)

        # Style button — opens MarkerStylePicker for the active marker.
        self.btn_style = QtWidgets.QPushButton("Style…")
        self.btn_style.setObjectName("marker_style_button")
        self.btn_style.clicked.connect(self._on_style_clicked)
        left_panel.addWidget(self.btn_style)

        layout.addLayout(left_panel, 1)

        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_marker = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_marker)

        layout.addLayout(right_panel, 3)

    # ----------------------------------------------------------- Public API

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        self.model = model
        self.list_markers.clear()

        if model is None:
            self.canvas_marker.clear_axes()
            return

        for name in model.marker_names():
            self.list_markers.addItem(name)

        if model.marker_names():
            self.list_markers.setCurrentRow(0)

    def update_plot(self) -> None:
        """Update the marker plot based on selected marker and component."""
        if self.model is None:
            return

        selected_items = self.list_markers.selectedItems()
        if not selected_items:
            self.canvas_marker.clear_axes()
            self._current_handle = None
            self._current_marker = None
            return

        name = selected_items[0].text()
        marker = self.model.markers.get(name)
        if marker is None or self.model.point_time is None:
            self.canvas_marker.clear_axes()
            self._current_handle = None
            self._current_marker = None
            return

        t = self.model.point_time
        pos = marker.position  # (N,3)

        self.canvas_marker.fig.clear()
        ax = self.canvas_marker.add_subplot(111)

        idx = self.combo_component.currentIndex()
        if idx == 0:
            ax.plot(t, pos[:, 0], label="X")
            ax.plot(t, pos[:, 1], label="Y")
            ax.plot(t, pos[:, 2], label="Z")
            ax.set_ylabel("Position")
            ax.legend()
        elif idx in [1, 2, 3]:
            comp_idx = idx - 1
            comp_label = ["X", "Y", "Z"][comp_idx]
            ax.plot(t, pos[:, comp_idx], label=comp_label)
            ax.set_ylabel(f"{comp_label} position")
            ax.legend()
        else:
            disp = np.diff(pos, axis=0)
            dt = np.diff(t)
            dt[dt <= 0] = np.nan
            disp_float = disp.astype(float, copy=False)
            speed = np.sqrt(np.einsum("...i,...i->...", disp_float, disp_float)) / dt
            ax.plot(t[1:], speed, label="Speed magnitude")
            ax.set_ylabel("Speed (units/s)")
            ax.legend()

        ax.set_title(f"Marker: {name}")
        ax.set_xlabel("Time (s)")
        ax.grid(True)

        # plot_style: register/refresh a marker glyph at the first frame
        # so users have a visible artist to re-style. The renderer keeps
        # its own state across plot regenerations.
        self._renderer = MatplotlibMarkerRenderer(ax)
        first = pos[0] if pos.shape[0] > 0 else np.zeros(3, dtype=float)
        # 2D projection — drop Z (matplotlib 2D Axes only).
        glyph_pos = np.asarray([[float(t[0]), float(first[1])]], dtype=float)
        style = self._persistence.get(name) or default_style_for(name)
        try:
            self._current_handle = self._renderer.add_markers(glyph_pos, style, name)
        except (TypeError, ValueError) as exc:
            logger.warning("could not register marker glyph for %s: %s", name, exc)
            self._current_handle = None
        self._current_marker = name

        self.canvas_marker.fig.tight_layout()
        self.canvas_marker.draw()  # type: ignore

    # ------------------------------------------------------- Style dialog

    def apply_style(self, name: str, style: MarkerStyle) -> None:
        """Programmatic entry point used by tests and the dialog flow."""
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        self._persistence.set(name, style)
        if (
            self._renderer is not None
            and self._current_handle is not None
            and self._current_marker == name
        ):
            try:
                self._renderer.update_style(self._current_handle, style)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("update_style failed: %s", exc)
            self.canvas_marker.draw_idle()
        self._persistence.request_save()

    def _on_style_clicked(self) -> None:
        if self._current_marker is None:
            return
        current = self._persistence.get(self._current_marker) or default_style_for(
            self._current_marker
        )
        new_style = _open_style_dialog(self, current, f"Style — {self._current_marker}")
        if new_style is None:
            return
        self.apply_style(self._current_marker, new_style)
