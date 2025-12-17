#!/usr/bin/env python
"""
C3D Motion Analysis GUI

Features:
- Load C3D files (via ezc3d)
- Inspect metadata, markers, analog channels
- 2D plots of marker/analog time-series
- 3D marker trajectory viewer
- Basic kinematic analysis: speed, path length, extrema

Dependencies:
    See python/requirements.txt for required packages.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import ezc3d
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt

# ---------------------------------------------------------------------------
# Data model for C3D content
# ---------------------------------------------------------------------------


@dataclass
class MarkerData:
    name: str
    position: npt.NDArray[np.float64]  # shape (N, 3)
    residuals: npt.NDArray[np.float64] | None = None


@dataclass
class AnalogData:
    name: str
    values: npt.NDArray[np.float64]  # shape (N,)
    unit: str = ""


@dataclass
class C3DDataModel:
    filepath: str
    markers: dict[str, MarkerData] = field(default_factory=dict)
    analog: dict[str, AnalogData] = field(default_factory=dict)
    point_rate: float = 0.0
    analog_rate: float = 0.0
    point_time: npt.NDArray[np.float64] | None = None
    analog_time: npt.NDArray[np.float64] | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def marker_names(self) -> list[str]:
        """Return list of marker names."""
        return list(self.markers.keys())

    def analog_names(self) -> list[str]:
        """Return list of analog channel names."""
        return list(self.analog.keys())


# ---------------------------------------------------------------------------
# Matplotlib canvas embedded in Qt
# ---------------------------------------------------------------------------


class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget for embedding plots in Qt."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        width: float = 5.0,
        height: float = 4.0,
        dpi: int = 100,
    ) -> None:
        """Initialize the matplotlib canvas with specified dimensions."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)  # type: ignore
        self.setParent(parent)

    def clear_axes(self) -> None:
        """Clear all axes from the figure."""
        self.fig.clear()
        self.draw()  # type: ignore

    def add_subplot(self, *args: Any, **kwargs: Any) -> Axes:
        """Add a subplot to the figure and return the axes."""
        ax = self.fig.add_subplot(*args, **kwargs)
        return ax


# ---------------------------------------------------------------------------
# Utility: simple kinematic analysis for markers
# ---------------------------------------------------------------------------


def compute_marker_statistics(
    time: npt.NDArray[np.float64], pos: npt.NDArray[np.float64]
) -> dict[str, float]:
    """
    Compute basic kinematic quantities for a single marker trajectory:
    - total path length
    - max speed
    - mean speed
    """
    if pos.shape[0] < 2 or time is None or len(time) != pos.shape[0]:
        return {
            "path_length": np.nan,
            "max_speed": np.nan,
            "mean_speed": np.nan,
        }

    dt = np.diff(time)
    dt[dt <= 0] = np.nan  # avoid division by zero

    disp = np.diff(pos, axis=0)  # (N-1, 3)
    segment_length = np.linalg.norm(disp, axis=1)
    speed = segment_length / dt

    path_length = np.nansum(segment_length)
    max_speed = np.nanmax(speed)
    mean_speed = np.nanmean(speed)

    return {
        "path_length": float(path_length),
        "max_speed": float(max_speed),
        "mean_speed": float(mean_speed),
    }


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------


class C3DViewerMainWindow(QtWidgets.QMainWindow):
    """Main window for the C3D motion analysis viewer application."""

    def __init__(self) -> None:
        """Initialize the main window and create UI components."""
        super().__init__()

        self.setWindowTitle("C3D Motion Analysis Viewer")
        self.resize(1400, 900)

        self.model: C3DDataModel | None = None

        self._create_actions()
        self._create_menus()
        self._create_central_widget()
        self._update_ui_state(False)

        if (sb := self.statusBar()) is not None:
            sb.showMessage("Ready")

    # ----------------------------- UI setup --------------------------------

    def _create_actions(self) -> None:
        """Create menu actions for the application."""
        self.action_open = QtGui.QAction("Open &C3D…", self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.setStatusTip("Open a C3D file for analysis")
        self.action_open.triggered.connect(self.open_c3d_file)

        self.action_exit = QtGui.QAction("E&xit", self)
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.triggered.connect(self.close)

        self.action_about = QtGui.QAction("&About", self)
        self.action_about.triggered.connect(self.show_about_dialog)

    def _create_menus(self) -> None:
        """Create menu bar and menus."""
        menubar = self.menuBar()
        if menubar is None:
            return

        file_menu = menubar.addMenu("&File")
        if file_menu is not None:
            file_menu.addAction(self.action_open)
            file_menu.addSeparator()
            file_menu.addAction(self.action_exit)

        help_menu = menubar.addMenu("&Help")
        if help_menu is not None:
            help_menu.addAction(self.action_about)

    def _create_central_widget(self) -> None:
        """Create the central tab widget with all tabs."""
        self.tabs = QtWidgets.QTabWidget()

        self.overview_tab = self._create_overview_tab()
        self.marker_plot_tab = self._create_marker_plot_tab()
        self.analog_plot_tab = self._create_analog_plot_tab()
        self.viewer3d_tab = self._create_3d_viewer_tab()
        self.analysis_tab = self._create_analysis_tab()

        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.addTab(self.marker_plot_tab, "Markers (2D)")
        self.tabs.addTab(self.analog_plot_tab, "Analog")
        self.tabs.addTab(self.viewer3d_tab, "3D Viewer")
        self.tabs.addTab(self.analysis_tab, "Analysis")

        self.setCentralWidget(self.tabs)

    # ------------------------- Overview tab --------------------------------

    def _create_overview_tab(self) -> QtWidgets.QWidget:
        """Create the overview tab showing file metadata."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # File info
        self.label_file = QtWidgets.QLabel("No file loaded")
        self.label_file.setWordWrap(True)
        self.label_file.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label_file)

        # Basic info table
        self.table_metadata = QtWidgets.QTableWidget()
        self.table_metadata.setColumnCount(2)
        self.table_metadata.setHorizontalHeaderLabels(["Field", "Value"])
        header = self.table_metadata.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_metadata)

        return widget

    # ---------------------- Marker 2D plot tab -----------------------------

    def _create_marker_plot_tab(self) -> QtWidgets.QWidget:
        """Create the marker 2D plotting tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        # Left: marker list + options
        left_panel = QtWidgets.QVBoxLayout()
        self.list_markers = QtWidgets.QListWidget()
        self.list_markers.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_markers.itemSelectionChanged.connect(self.update_marker_plot)
        left_panel.addWidget(QtWidgets.QLabel("Markers:"))
        left_panel.addWidget(self.list_markers)

        self.combo_component = QtWidgets.QComboBox()
        self.combo_component.addItems(["All (X/Y/Z)", "X", "Y", "Z", "Speed magnitude"])
        self.combo_component.currentIndexChanged.connect(self.update_marker_plot)
        left_panel.addWidget(QtWidgets.QLabel("Component:"))
        left_panel.addWidget(self.combo_component)

        layout.addLayout(left_panel, 1)

        # Right: plotting area
        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_marker = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_marker)

        layout.addLayout(right_panel, 3)
        return widget

    # ---------------------- Analog plot tab --------------------------------

    def _create_analog_plot_tab(self) -> QtWidgets.QWidget:
        """Create the analog channel plotting tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        left_panel = QtWidgets.QVBoxLayout()
        self.list_analog = QtWidgets.QListWidget()
        self.list_analog.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_analog.itemSelectionChanged.connect(self.update_analog_plot)
        left_panel.addWidget(QtWidgets.QLabel("Analog channels:"))
        left_panel.addWidget(self.list_analog)

        layout.addLayout(left_panel, 1)

        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_analog = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_analog)

        layout.addLayout(right_panel, 3)
        return widget

    # ---------------------- 3D viewer tab ----------------------------------

    def _create_3d_viewer_tab(self) -> QtWidgets.QWidget:
        """Create the 3D marker trajectory viewer tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        left_panel = QtWidgets.QVBoxLayout()
        self.list_markers_3d = QtWidgets.QListWidget()
        self.list_markers_3d.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self.list_markers_3d.itemSelectionChanged.connect(self.update_3d_view)
        left_panel.addWidget(QtWidgets.QLabel("Markers to display in 3D:"))
        left_panel.addWidget(self.list_markers_3d)

        # Frame slider
        self.slider_frame = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.setValue(0)
        self.slider_frame.valueChanged.connect(self.update_3d_view)
        left_panel.addWidget(QtWidgets.QLabel("Frame index:"))
        left_panel.addWidget(self.slider_frame)

        self.label_frame_info = QtWidgets.QLabel("Frame: - / Time: -")
        left_panel.addWidget(self.label_frame_info)

        layout.addLayout(left_panel, 1)

        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_3d = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_3d)

        layout.addLayout(right_panel, 3)
        return widget

    # ---------------------- Analysis tab -----------------------------------

    def _create_analysis_tab(self) -> QtWidgets.QWidget:
        """Create the kinematic analysis tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        top_layout = QtWidgets.QHBoxLayout()
        self.combo_marker_analysis = QtWidgets.QComboBox()
        self.combo_marker_analysis.currentIndexChanged.connect(
            self.update_analysis_panel
        )
        top_layout.addWidget(QtWidgets.QLabel("Marker:"))
        top_layout.addWidget(self.combo_marker_analysis)

        self.button_recompute_stats = QtWidgets.QPushButton("Recompute stats")
        self.button_recompute_stats.setToolTip(
            "Recalculate statistics for the selected marker"
        )
        self.button_recompute_stats.clicked.connect(self.update_analysis_panel)
        top_layout.addWidget(self.button_recompute_stats)

        layout.addLayout(top_layout)

        # Stats area
        self.text_analysis = QtWidgets.QTextEdit()
        self.text_analysis.setReadOnly(True)
        layout.addWidget(self.text_analysis)

        # Optional: simple speed plot in analysis tab
        self.canvas_analysis = MplCanvas(self, width=5, height=3, dpi=100)
        layout.addWidget(self.canvas_analysis)

        return widget

    # ---------------------- UI state management ----------------------------

    def _update_ui_state(self, enabled: bool) -> None:
        """Update the enabled state of UI widgets after loading a model."""
        widgets = [
            self.tabs,
        ]
        for w in widgets:
            w.setEnabled(enabled)

    # --------------------------- File I/O ----------------------------------

    def open_c3d_file(self) -> None:
        """Open a file dialog to load a C3D file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open C3D file",
            "",
            "C3D files (*.c3d);;All files (*.*)",
        )
        if not path:
            return

        if (sb := self.statusBar()) is not None:
            sb.showMessage(f"Loading {os.path.basename(path)}...")
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            model = self._load_c3d(path)
            self.model = model
            self._populate_ui_with_model()
            self._update_ui_state(True)
            if (sb := self.statusBar()) is not None:
                sb.showMessage(f"Loaded {os.path.basename(path)} successfully.")
        except Exception as e:
            if (sb := self.statusBar()) is not None:
                sb.showMessage("Error loading file.")
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading C3D",
                f"Failed to load file:\n{path}\n\nError:\n{e}",
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _load_c3d(self, filepath: str) -> C3DDataModel:
        """Load and parse a C3D file into a C3DDataModel."""
        c3d_obj = ezc3d.c3d(filepath)

        # Point (marker) data
        points = c3d_obj["data"]["points"]  # shape (4, Npoints, Nframes)
        n_dim, n_points, n_frames = points.shape
        if n_dim != 4:
            raise ValueError(
                f"Expected 4 dimensions for marker data (x, y, z, residual), got {n_dim}"
            )

        labels_points = c3d_obj["parameters"]["POINT"]["LABELS"]["value"]
        try:
            units_points = c3d_obj["parameters"]["POINT"]["UNITS"]["value"][0]
        except (KeyError, IndexError, TypeError):
            units_points = "unknown"

        frame_rate = float(c3d_obj["parameters"]["POINT"]["RATE"]["value"][0])

        # Build marker dictionary
        markers: dict[str, MarkerData] = {}
        for i, name in enumerate(labels_points):
            xyz = points[0:3, i, :].T  # (Nframes, 3)
            residuals = points[3, i, :].T  # (Nframes,)
            markers[name] = MarkerData(name=name, position=xyz, residuals=residuals)

        # Analog data
        analog = {}
        if "ANALOG" in c3d_obj["parameters"]:
            labels_analog = c3d_obj["parameters"]["ANALOG"]["LABELS"]["value"]
            analog_data = c3d_obj["data"][
                "analogs"
            ]  # shape (Nsubframes, Nchannels, Nframes)
            # ezc3d stores analog as (Nsubframes, Nchannels, Nframes); flatten in time
            n_sub, n_ch, n_f = analog_data.shape
            analog_rate = float(c3d_obj["parameters"]["ANALOG"]["RATE"]["value"][0])

            analog_flat = analog_data.transpose(2, 0, 1).reshape(
                n_sub * n_f, n_ch
            )  # (n_sub * n_f, n_ch)

            # Units per channel, if available
            try:
                analog_units = c3d_obj["parameters"]["ANALOG"]["UNITS"]["value"]
            except (KeyError, TypeError):
                analog_units = [""] * len(labels_analog)

            for j, name in enumerate(labels_analog):
                unit = analog_units[j] if j < len(analog_units) else ""
                analog[name] = AnalogData(
                    name=name, values=analog_flat[:, j], unit=unit
                )
        else:
            analog_rate = 0.0

        # Time vectors
        time_point = np.arange(n_frames) / frame_rate if frame_rate > 0 else None
        n_analog_samples = next(iter(analog.values())).values.shape[0] if analog else 0
        time_analog = (
            np.arange(n_analog_samples) / analog_rate
            if analog_rate > 0 and n_analog_samples > 0
            else None
        )

        # Metadata extraction
        metadata = {
            "File": os.path.basename(filepath),
            "Path": filepath,
            "Point rate (Hz)": f"{frame_rate:.3f}",
            "Analog rate (Hz)": f"{analog_rate:.3f}",
            "Frames": str(n_frames),
            "Points": str(n_points),
            "Units (POINT)": units_points,
        }

        if "TRIAL" in c3d_obj["parameters"]:
            for key, param in c3d_obj["parameters"]["TRIAL"].items():
                if "value" in param:
                    value = param["value"]
                    if isinstance(value, list | tuple | np.ndarray):
                        v = ", ".join(str(x) for x in value)
                    else:
                        v = str(value)
                    metadata[f"TRIAL::{key}"] = v

        model = C3DDataModel(
            filepath=filepath,
            markers=markers,
            analog=analog,
            point_rate=frame_rate,
            analog_rate=analog_rate,
            point_time=time_point,
            analog_time=time_analog,
            metadata=metadata,
        )

        return model

    # --------------------- Populate UI from model --------------------------

    def _populate_ui_with_model(self) -> None:
        """Populate UI components with data from the loaded model."""
        if self.model is None:
            return

        # Overview tab
        self.label_file.setText(f"Loaded file: {self.model.filepath}")
        self._populate_metadata_table()

        # Marker list (2D tab)
        self.list_markers.clear()
        for name in self.model.marker_names():
            self.list_markers.addItem(name)

        # Marker list (3D tab)
        self.list_markers_3d.clear()
        for name in self.model.marker_names():
            self.list_markers_3d.addItem(name)

        # Analog list
        self.list_analog.clear()
        for name in self.model.analog_names():
            self.list_analog.addItem(name)

        # Frame slider for 3D
        if self.model.point_time is not None:
            n_frames = len(self.model.point_time)
            self.slider_frame.setMinimum(0)
            self.slider_frame.setMaximum(n_frames - 1)
            self.slider_frame.setValue(0)
        else:
            self.slider_frame.setMinimum(0)
            self.slider_frame.setMaximum(0)
            self.slider_frame.setValue(0)

        # Analysis tab marker selection
        self.combo_marker_analysis.clear()
        self.combo_marker_analysis.addItems(self.model.marker_names())
        if self.model.marker_names():
            self.combo_marker_analysis.setCurrentIndex(0)

        # Clear plots
        self.canvas_marker.clear_axes()
        self.canvas_analog.clear_axes()
        self.canvas_3d.clear_axes()
        self.canvas_analysis.clear_axes()
        self.text_analysis.clear()

        # Trigger initial plots/analysis
        if self.model.marker_names():
            self.list_markers.setCurrentRow(0)
            self.list_markers_3d.setCurrentRow(0)
        if self.model.analog_names():
            self.list_analog.setCurrentRow(0)
        self.update_analysis_panel()

    def _populate_metadata_table(self) -> None:
        """Populate the metadata table with model metadata."""
        if self.model is None:
            return
        self.table_metadata.setRowCount(0)
        for key, value in self.model.metadata.items():
            row = self.table_metadata.rowCount()
            self.table_metadata.insertRow(row)
            item_key = QtWidgets.QTableWidgetItem(key)
            item_value = QtWidgets.QTableWidgetItem(str(value))
            self.table_metadata.setItem(row, 0, item_key)
            self.table_metadata.setItem(row, 1, item_value)

    # ------------------------ Marker plotting ------------------------------

    def update_marker_plot(self) -> None:
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

    # ------------------------ Analog plotting ------------------------------

    def update_analog_plot(self) -> None:
        """Update the analog plot based on selected channel."""
        if self.model is None:
            return
        selected_items = self.list_analog.selectedItems()
        if not selected_items:
            self.canvas_analog.clear_axes()
            return

        name = selected_items[0].text()
        channel = self.model.analog.get(name)
        if channel is None or self.model.analog_time is None:
            self.canvas_analog.clear_axes()
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

        self.canvas_analog.fig.tight_layout()
        self.canvas_analog.draw()  # type: ignore

    # ------------------------ 3D view --------------------------------------

    def update_3d_view(self) -> None:
        """Update the 3D view based on selected markers and frame."""
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

    # ------------------------ Analysis tab ---------------------------------

    def update_analysis_panel(self) -> None:
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

        stats = compute_marker_statistics(t, pos)

        # Text summary
        text_lines = [
            f"Marker: {marker_name}",
            f"Number of frames: {pos.shape[0]}",
            "",
            "Basic kinematic summary:",
            f"  Total path length: {stats['path_length']:.4f} (position units)",
            f"  Max speed:         {stats['max_speed']:.4f} (position units/s)",
            f"  Mean speed:        {stats['mean_speed']:.4f} (position units/s)",
        ]

        # Find approximate peak speed frame, if sensible
        if pos.shape[0] > 2:
            disp = np.diff(pos, axis=0)
            dt = np.diff(t)
            dt[dt <= 0] = np.nan
            speed = np.linalg.norm(disp, axis=1) / dt
            if np.all(np.isnan(speed)):
                text_lines.append("")
                text_lines.append("  Peak speed: N/A (all speeds are NaN)")
            else:
                peak_idx = int(np.nanargmax(speed))
                peak_time = t[peak_idx + 1]
                text_lines.append("")
                text_lines.append(
                    f"  Peak speed at time: {peak_time:.4f} s (frame {peak_idx + 1})"
                )

        self.text_analysis.setPlainText("\n".join(text_lines))

        # Speed plot
        self.canvas_analysis.fig.clear()
        ax = self.canvas_analysis.add_subplot(111)
        if pos.shape[0] > 2:
            disp = np.diff(pos, axis=0)
            dt = np.diff(t)
            dt[dt <= 0] = np.nan
            speed = np.linalg.norm(disp, axis=1) / dt
            ax.plot(t[1:], speed, label="Speed magnitude")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (units/s)")
            ax.set_title(f"Speed profile: {marker_name}")
            ax.grid(True)
            ax.legend()
            self.canvas_analysis.fig.tight_layout()

        self.canvas_analysis.draw()  # type: ignore

    # ------------------------- About dialog --------------------------------

    def show_about_dialog(self) -> None:
        """Display the about dialog with application information."""
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "C3D Motion Analysis Viewer\n\n"
            "Reads C3D files using ezc3d and offers plotting, 3D visualization, "
            "and basic marker kinematic analysis.\n\n"
            "Extend this to domain-specific metrics (golf swing, gait, etc.).",
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the C3D viewer application."""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("C3D Motion Analysis Viewer")

    window = C3DViewerMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
