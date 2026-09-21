"""PyQt6 GUI for the Tour Matching Viewer (Visuals Handoff Step 3).

Provides interactive 3D playback and visual comparison of model kinematics
against optical mocap target markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Default to Agg unless canvas installs backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.tour_matching_viewer.core import (
    ReplayData,
    ViewerFrame,
    load_replay,
    viewer_frame,
)

logger = get_logger(__name__)

ENGINE_COLORS: dict[str, str] = {
    "mujoco": "#1f77b4",  # Blue
    "pinocchio": "#d62728",  # Red
    "drake": "#2ca02c",  # Green
    "opensim": "#9467bd",  # Purple
    "default": "#ff7f0e",  # Orange
}

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


class TourMatchingViewerWidget(QtWidgets.QWidget):
    """Main embeddable widget for tour matching replay and visual analysis."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        spec_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec_path = spec_path or DEFAULT_SPEC_PATH
        self._spec: dict[str, Any] | None = None
        self._replay: ReplayData | None = None
        self._current_frame: int = 0
        self._candidate_hash: str = "unknown"
        self._engine_name: str = "default"
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._is_playing: bool = False

        self._init_ui()
        self._load_spec()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header bar: title, engine, hash, and RMS
        header_layout = QtWidgets.QHBoxLayout()
        self._title_label = QtWidgets.QLabel("Tour Matching Viewer — No replay loaded")
        font = self._title_label.font()
        font.setBold(True)
        self._title_label.setFont(font)
        header_layout.addWidget(self._title_label)

        header_layout.addStretch()

        self._rms_label = QtWidgets.QLabel("Valid Marker RMS: — mm")
        header_layout.addWidget(self._rms_label)

        self._open_btn = QtWidgets.QPushButton("Open Replay…")
        self._open_btn.clicked.connect(self._on_open_clicked)
        header_layout.addWidget(self._open_btn)

        layout.addLayout(header_layout)

        # 3D Viewport
        self._figure = Figure(figsize=(6, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111, projection="3d")
        self._setup_3d_axes()
        layout.addWidget(self._canvas, stretch=1)

        # Controls bar: Play/Pause, Slider, Frame Label
        controls_layout = QtWidgets.QHBoxLayout()
        self._play_btn = QtWidgets.QPushButton("▶ Play")
        self._play_btn.setFixedWidth(80)
        self._play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self._play_btn)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        controls_layout.addWidget(self._slider, stretch=1)

        self._frame_label = QtWidgets.QLabel("Frame: 0 / 0 (0.000 s)")
        self._frame_label.setFixedWidth(160)
        controls_layout.addWidget(self._frame_label)

        layout.addLayout(controls_layout)

    def _setup_3d_axes(self) -> None:
        self._ax.clear()
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")
        self._ax.set_xlim(-1.5, 1.5)
        self._ax.set_ylim(-1.5, 1.5)
        self._ax.set_zlim(0.0, 2.0)
        self._ax.view_init(elev=20, azim=45)

        # Draw ground grid at Z = 0
        gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 7), np.linspace(-1.5, 1.5, 7))
        gz = np.zeros_like(gx)
        self._ax.plot_wireframe(gx, gy, gz, color="#d3d3d3", linewidth=0.5, alpha=0.5)

    def _load_spec(self) -> None:
        if self._spec_path.exists():
            import json

            try:
                self._spec = json.loads(self._spec_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception(
                    "Failed to load specification from %s", self._spec_path
                )

    def load_replay_data(
        self,
        replay: ReplayData,
        *,
        candidate_hash: str = "returned81",
        engine_name: str = "mujoco",
    ) -> None:
        """Populate viewer with pre-loaded replay data."""
        self._replay = replay
        self._candidate_hash = candidate_hash
        self._engine_name = engine_name.lower()
        self._current_frame = 0

        n_frames = replay.frame_count
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, n_frames - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        title = f"Candidate: {self._candidate_hash} | Engine: {self._engine_name.capitalize()}"
        self._title_label.setText(title)

        self.render_frame(0)

    def load_file(self, path: Path | str) -> None:
        """Load a replay file (.npz or .mot)."""
        p = Path(path)
        if self._spec is None:
            self._load_spec()
        replay = load_replay(p, self._spec)

        # Infer candidate hash and engine name from path
        candidate_hash = "returned81" if "returned81" in p.stem else p.stem[:12]
        engine_name = "default"
        for eng in ("mujoco", "pinocchio", "drake", "opensim"):
            if eng in p.name.lower() or eng in str(p.parent).lower():
                engine_name = eng
                break

        self.load_replay_data(
            replay, candidate_hash=candidate_hash, engine_name=engine_name
        )

    def render_frame(self, frame_idx: int) -> None:
        """Evaluate kinematics and render 3D elements for a given frame index."""
        if self._replay is None or self._spec is None:
            return

        n_frames = self._replay.frame_count
        if frame_idx < 0 or frame_idx >= n_frames:
            return

        self._current_frame = frame_idx
        t = float(self._replay.time_s[frame_idx])
        self._frame_label.setText(f"Frame: {frame_idx + 1} / {n_frames} ({t:.3f} s)")

        vframe: ViewerFrame = viewer_frame(self._spec, self._replay, frame_idx)
        self._rms_label.setText(f"Valid Marker RMS: {vframe.rms_error * 1000.0:.2f} mm")

        self._setup_3d_axes()

        # 1. Render visual skeleton segments as 3D lines
        lines = []
        for seg in vframe.segments:
            lines.append([seg.start_m, seg.end_m])
        if lines:
            line_coll = Line3DCollection(
                lines, colors="#2b5c8f", linewidths=2.5, alpha=0.85
            )
            self._ax.add_collection3d(line_coll)

        # 2. Render target markers (black)
        if vframe.target_markers is not None:
            valid = (
                vframe.valid_mask
                if vframe.valid_mask is not None
                else np.ones(len(vframe.target_markers), dtype=bool)
            )
            tm = vframe.target_markers[valid]
            if len(tm) > 0:
                self._ax.scatter(
                    tm[:, 0],
                    tm[:, 1],
                    tm[:, 2],
                    color="black",
                    s=18,
                    label="Target Markers",
                    alpha=0.7,
                )

        # 3. Render model markers (engine colour)
        if vframe.model_markers is not None:
            eng_color = ENGINE_COLORS.get(self._engine_name, ENGINE_COLORS["default"])
            mm = vframe.model_markers
            self._ax.scatter(
                mm[:, 0],
                mm[:, 1],
                mm[:, 2],
                color=eng_color,
                s=22,
                label=f"Model ({self._engine_name})",
                alpha=0.9,
            )

        self._canvas.draw_idle()

    def toggle_playback(self) -> None:
        """Toggle animation timer."""
        if self._is_playing:
            self._timer.stop()
            self._play_btn.setText("▶ Play")
            self._is_playing = False
        else:
            if self._replay is not None and self._replay.frame_count > 1:
                # Target ~30 fps playback
                self._timer.start(33)
                self._play_btn.setText("⏸ Pause")
                self._is_playing = True

    def _on_timer_tick(self) -> None:
        if self._replay is None:
            return
        n_frames = self._replay.frame_count
        next_frame = (self._current_frame + 1) % n_frames
        self._slider.blockSignals(True)
        self._slider.setValue(next_frame)
        self._slider.blockSignals(False)
        self.render_frame(next_frame)

    def _on_slider_changed(self, value: int) -> None:
        self.render_frame(value)

    def _on_open_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Tour Matching Replay",
            "",
            "Replay Files (*.npz *.mot *.sto);;All Files (*)",
        )
        if path:
            try:
                self.load_file(Path(path))
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error Loading Replay", f"Failed to load replay:\n{e}"
                )

    def cleanup(self) -> None:
        """Halt playback and release resources."""
        self._timer.stop()
        self._is_playing = False
        self._figure.clf()


class TourMatchingViewerWindow(QtWidgets.QMainWindow):
    """Standalone main window hosting the Tour Matching Viewer."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tour Matching Viewer")
        self.resize(1024, 768)
        self.widget = TourMatchingViewerWidget(self)
        self.setCentralWidget(self.widget)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:  # noqa: N802
        self.widget.cleanup()
        if event is not None:
            event.accept()
