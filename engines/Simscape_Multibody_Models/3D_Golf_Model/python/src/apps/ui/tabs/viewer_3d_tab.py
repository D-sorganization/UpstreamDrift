from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
import numpy as np
from typing import Any
from ...core.models import C3DDataModel
from ..widgets.mpl_canvas import MplCanvas

class Viewer3DTab(QtWidgets.QWidget):
    """3D marker trajectory viewer tab."""

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        left_panel = QtWidgets.QVBoxLayout()
        self.list_markers_3d = QtWidgets.QListWidget()
        self.list_markers_3d.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self.list_markers_3d.itemSelectionChanged.connect(self.update_view)
        left_panel.addWidget(QtWidgets.QLabel("Markers to display in 3D:"))
        left_panel.addWidget(self.list_markers_3d)

        # Frame slider
        self.slider_frame = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.setValue(0)
        self.slider_frame.valueChanged.connect(self.update_view)
        left_panel.addWidget(QtWidgets.QLabel("Frame index:"))
        left_panel.addWidget(self.slider_frame)

        self.label_frame_info = QtWidgets.QLabel("Frame: - / Time: -")
        left_panel.addWidget(self.label_frame_info)

        layout.addLayout(left_panel, 1)

        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_3d = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_3d)

        layout.addLayout(right_panel, 3)

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        self.model = model
        self.list_markers_3d.clear()
        
        if model is None:
            self.slider_frame.setMaximum(0)
            self.canvas_3d.clear_axes()
            return

        for name in model.marker_names():
            self.list_markers_3d.addItem(name)

        # Frame slider for 3D
        if model.point_time is not None:
            n_frames = len(model.point_time)
            self.slider_frame.setMinimum(0)
            self.slider_frame.setMaximum(n_frames - 1)
            self.slider_frame.setValue(0)
        else:
            self.slider_frame.setMaximum(0)
            self.slider_frame.setValue(0)

        if model.marker_names():
            self.list_markers_3d.setCurrentRow(0)

    def update_view(self) -> None:
        """Update the 3D view based on selected Markers and frame."""
        if self.model is None:
            return

        # Get selected markers
        selected_items = self.list_markers_3d.selectedItems()
        marker_names = [item.text() for item in selected_items]
        if not marker_names:
            self.canvas_3d.clear_axes()
            self.label_frame_info.setText("Frame: - / Time: -")
            return

        frame_index = self.slider_frame.value()
        if self.model.point_time is None:
            time_str = "-"
        else:
            if 0 <= frame_index < len(self.model.point_time):
                time_str = f"{self.model.point_time[frame_index]:.4f} s"
            else:
                time_str = "-"

        self.label_frame_info.setText(f"Frame: {frame_index} / Time: {time_str}")

        self.canvas_3d.fig.clear()
        ax: Any = self.canvas_3d.add_subplot(111, projection="3d")

        # Plot full trajectories (faint) and current point (bold)
        for name in marker_names:
            marker = self.model.markers.get(name)
            if marker is None:
                continue
            pos = marker.position  # (N,3)
            if pos.shape[0] == 0:
                continue

            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.3, label=name)
            if 0 <= frame_index < pos.shape[0]:
                x, y, z = pos[frame_index]
                ax.scatter([x], [y], [z], s=40)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Marker Trajectories")

        # Try to set equal aspect ratio
        all_pts = []
        for name in marker_names:
            m = self.model.markers.get(name)
            if m is not None and m.position.size > 0:
                all_pts.append(m.position)
        if all_pts:
            pts = np.vstack(all_pts)
            x_min, y_min, z_min = pts.min(axis=0)
            x_max, y_max, z_max = pts.max(axis=0)
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            if max_range > 0:
                mid_x = 0.5 * (x_max + x_min)
                mid_y = 0.5 * (y_max + y_min)
                mid_z = 0.5 * (z_max + z_min)
                half = max_range / 2.0
                ax.set_xlim(mid_x - half, mid_x + half)
                ax.set_ylim(mid_y - half, mid_y + half)
                ax.set_zlim(mid_z - half, mid_z + half)

        ax.legend()
        self.canvas_3d.fig.tight_layout()
        self.canvas_3d.draw()  # type: ignore
