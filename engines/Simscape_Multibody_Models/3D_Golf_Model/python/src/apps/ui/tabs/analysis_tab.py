from PyQt6 import QtWidgets
import numpy as np
from ...core.models import C3DDataModel
from ...services.analysis import compute_marker_statistics
from ..widgets.mpl_canvas import MplCanvas

class AnalysisTab(QtWidgets.QWidget):
    """Kinematic analysis tab."""

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top_layout = QtWidgets.QHBoxLayout()
        self.combo_marker_analysis = QtWidgets.QComboBox()
        self.combo_marker_analysis.currentIndexChanged.connect(
            self.update_panel
        )
        top_layout.addWidget(QtWidgets.QLabel("Marker:"))
        top_layout.addWidget(self.combo_marker_analysis)

        self.button_recompute_stats = QtWidgets.QPushButton("Recompute stats")
        self.button_recompute_stats.setToolTip(
            "Recalculate statistics for the selected marker"
        )
        self.button_recompute_stats.clicked.connect(self.update_panel)
        top_layout.addWidget(self.button_recompute_stats)

        layout.addLayout(top_layout)

        # Stats area
        self.text_analysis = QtWidgets.QTextEdit()
        self.text_analysis.setReadOnly(True)
        layout.addWidget(self.text_analysis)

        # Optional: simple speed plot in analysis tab
        self.canvas_analysis = MplCanvas(self, width=5, height=3, dpi=100)
        layout.addWidget(self.canvas_analysis)

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        self.model = model
        self.combo_marker_analysis.clear()
        
        if model is None:
            self.text_analysis.clear()
            self.canvas_analysis.clear_axes()
            return

        self.combo_marker_analysis.addItems(model.marker_names())
        if model.marker_names():
            self.combo_marker_analysis.setCurrentIndex(0)
            
        self.update_panel()

    def update_panel(self) -> None:
        """Update the analysis panel with statistics for the selected marker."""
        if self.model is None:
            self.text_analysis.clear()
            self.canvas_analysis.clear_axes()
            return

        marker_name = self.combo_marker_analysis.currentText()
        marker = self.model.markers.get(marker_name)
        if marker is None or self.model.point_time is None:
            self.text_analysis.setPlainText("No marker / time data available.")
            self.canvas_analysis.clear_axes()
            return

        t = self.model.point_time
        pos = marker.position

        # Use extracted service for stats
        stats = compute_marker_statistics(t, pos)

        text = (
            f"Marker: {marker_name}\n\n"
            f"Path length: {stats.get('path_length', 0.0):.4f} units\n"
            f"Max speed:   {stats.get('max_speed', 0.0):.4f} units/s\n"
            f"Mean speed:  {stats.get('mean_speed', 0.0):.4f} units/s\n"
        )
        self.text_analysis.setPlainText(text)

        # Update mini-plot (recalculate speed for visualization)
        self.canvas_analysis.fig.clear()

        if pos.shape[0] > 1:
            ax = self.canvas_analysis.add_subplot(111)
            disp = np.diff(pos, axis=0)
            dt = np.diff(t)
            dt[dt <= 0] = np.nan
            speed = np.linalg.norm(disp, axis=1) / dt

            # Plot
            ax.plot(t[1:], speed, color="green", label="Speed")
            ax.set_title("Speed Profile")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed")
            ax.grid(True)
            self.canvas_analysis.fig.tight_layout()
            self.canvas_analysis.draw()  # type: ignore
        else:
            self.canvas_analysis.clear_axes()
