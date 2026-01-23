import numpy as np
from PyQt6 import QtWidgets

from ...core.models import C3DDataModel
from ..widgets.mpl_canvas import MplCanvas


class MarkerPlotTab(QtWidgets.QWidget):
    """Marker 2D plotting tab."""

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        # Left: marker list + options
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

        layout.addLayout(left_panel, 1)

        # Right: plotting area
        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_marker = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_marker)

        layout.addLayout(right_panel, 3)

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
            return

        name = selected_items[0].text()
        marker = self.model.markers.get(name)
        if marker is None or self.model.point_time is None:
            self.canvas_marker.clear_axes()
            return

        t = self.model.point_time
        pos = marker.position  # (N,3)

        self.canvas_marker.fig.clear()
        ax = self.canvas_marker.add_subplot(111)

        idx = self.combo_component.currentIndex()
        if idx == 0:
            # All components
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
            # Speed magnitude
            disp = np.diff(pos, axis=0)
            dt = np.diff(t)
            dt[dt <= 0] = np.nan
            speed = np.linalg.norm(disp, axis=1) / dt
            # Align length with t (N-1)
            ax.plot(t[1:], speed, label="Speed magnitude")
            ax.set_ylabel("Speed (units/s)")
            ax.legend()

        ax.set_title(f"Marker: {name}")
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        self.canvas_marker.fig.tight_layout()
        self.canvas_marker.draw()  # type: ignore
