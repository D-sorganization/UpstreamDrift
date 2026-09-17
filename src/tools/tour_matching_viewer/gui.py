"""PyQt6 GUI for the Tour Matching Viewer (Visuals Handoff Step 3).

Provides interactive 3D playback and visual comparison of model kinematics
against optical mocap target markers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # Default to Agg unless canvas installs backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.golf_simulator import MonotonicReplayClock
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.simulation_store import SimulationDataStore
from src.shared.python.simulation_store.replay_bundle import load_simscape_bundle
from src.tools.tour_matching_viewer.core import (
    ReplayData,
    ViewerFrame,
    cylinder_faces,
    load_replay,
    marker_error_vectors,
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


class ReportInspectorDialog(QtWidgets.QDialog):
    """Inspect retained R2025b qualification report, gate evaluation, and parity metrics."""

    def __init__(
        self,
        report_data: dict[str, Any],
        run_id: str = "",
        report_path: Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Replay Report — {run_id or 'Saved Run'}")
        self.resize(560, 500)

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(f"Qualification Report: {run_id}")
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        title.setFont(font)
        layout.addWidget(title)

        if report_path is not None:
            path_lbl = QtWidgets.QLabel(f"Source: {report_path}")
            path_lbl.setWordWrap(True)
            path_lbl.setStyleSheet("color: #7f8c8d; font-size: 10px;")
            layout.addWidget(path_lbl)

        notice = QtWidgets.QLabel(
            "Notice: Offline inspection of retained MATLAB R2025b evidence. No physics rerun."
        )
        notice.setStyleSheet("color: #2980b9; font-style: italic; font-size: 11px;")
        layout.addWidget(notice)

        tabs = QtWidgets.QTabWidget()

        overview_widget = QtWidgets.QWidget()
        ov_layout = QtWidgets.QVBoxLayout(overview_widget)

        # Environment & Release info
        rel = report_data.get("matlab_release", "unknown")
        ver = report_data.get("matlab_version", "unknown")
        dur = report_data.get("duration_s", "—")
        samples = report_data.get("n_samples", "—")
        elapsed = report_data.get("elapsed_s", "—")

        env_group = QtWidgets.QGroupBox("Environment & Solver")
        env_grid = QtWidgets.QGridLayout(env_group)
        env_grid.addWidget(QtWidgets.QLabel(f"Release: {rel}"), 0, 0)
        env_grid.addWidget(QtWidgets.QLabel(f"Version: {ver}"), 0, 1)
        env_grid.addWidget(
            QtWidgets.QLabel(f"Duration: {dur} s ({samples} samples)"), 1, 0
        )
        env_grid.addWidget(QtWidgets.QLabel(f"Elapsed: {elapsed} s"), 1, 1)
        ov_layout.addWidget(env_group)

        # Gates
        gates = report_data.get("gates", {})
        gates_group = QtWidgets.QGroupBox("Gate Certification")
        gates_layout = QtWidgets.QVBoxLayout(gates_group)
        if gates:
            for gate_name, passed in gates.items():
                status_text = "PASS" if passed else "FAIL"
                lbl = QtWidgets.QLabel(f"{gate_name}: {status_text}")
                lbl.setStyleSheet(
                    "color: #27ae60; font-weight: bold;"
                    if passed
                    else "color: #c0392b; font-weight: bold;"
                )
                gates_layout.addWidget(lbl)
        else:
            gates_layout.addWidget(QtWidgets.QLabel("No explicit gates recorded."))
        ov_layout.addWidget(gates_group)

        # Metrics
        metrics = report_data.get("metrics", {})
        if metrics:
            met_group = QtWidgets.QGroupBox("Recorded Metrics")
            met_layout = QtWidgets.QFormLayout(met_group)
            for k, v in metrics.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                met_layout.addRow(k, QtWidgets.QLabel(val_str))
            ov_layout.addWidget(met_group)

        ov_layout.addStretch()
        tabs.addTab(overview_widget, "Overview & Gates")

        # Raw JSON Tab
        raw_text = QtWidgets.QTextEdit()
        raw_text.setReadOnly(True)
        raw_text.setFont(QtGui.QFont("Courier New", 9))
        raw_text.setPlainText(json.dumps(report_data, indent=2))
        tabs.addTab(raw_text, "Raw JSON")

        layout.addWidget(tabs)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)


class AnimationInspectorDialog(QtWidgets.QDialog):
    """View the actual retained cylinder animation GIF for the saved run."""

    def __init__(
        self,
        animation_path: Path,
        run_id: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Cylinder Animation — {run_id or 'Saved Run'}")
        self.resize(720, 560)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(f"Retained Cylinder Animation: {run_id}")
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        path_lbl = QtWidgets.QLabel(f"Source file: {animation_path}")
        path_lbl.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        path_lbl.setWordWrap(True)
        layout.addWidget(path_lbl)

        # Image display with QMovie
        self._img_label = QtWidgets.QLabel()
        self._img_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(480, 360)
        self._movie = QtGui.QMovie(str(animation_path))
        if self._movie.isValid():
            self._img_label.setMovie(self._movie)
            self._movie.start()
        else:
            self._img_label.setText(
                f"Animation file found: {animation_path.name}\n(Preview unavailable in offscreen backend)"
            )
        layout.addWidget(self._img_label, stretch=1)

        bottom_layout = QtWidgets.QHBoxLayout()
        open_ext_btn = QtWidgets.QPushButton("Open File in System Viewer")
        open_ext_btn.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(animation_path))
            )
        )
        bottom_layout.addWidget(open_ext_btn)
        bottom_layout.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)
        layout.addLayout(bottom_layout)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:  # noqa: N802
        if self._movie is not None and self._movie.isValid():
            self._movie.stop()
        super().closeEvent(event)


class TourMatchingViewerWidget(QtWidgets.QWidget):
    """Main embeddable widget for tour matching replay and visual analysis."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        spec_path: Path | None = None,
        store: SimulationDataStore | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec_path = spec_path or DEFAULT_SPEC_PATH
        self._store = store or SimulationDataStore()
        self._spec: dict[str, Any] | None = None
        self._replay: ReplayData | None = None
        self._current_frame: int = 0
        self._candidate_hash: str = "unknown"
        self._engine_name: str = "default"
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._is_playing: bool = False
        self._clock = MonotonicReplayClock()
        self._report_data: dict[str, Any] | None = None
        self._report_path: Path | None = None
        self._animation_path: Path | None = None

        self._init_ui()
        self._load_spec()
        self._refresh_catalog()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Row 1: Title, Effort Badge, RMS, and Inspection Actions
        row1 = QtWidgets.QHBoxLayout()
        self._title_label = QtWidgets.QLabel("Tour Matching Viewer — No replay loaded")
        font = self._title_label.font()
        font.setBold(True)
        self._title_label.setFont(font)
        row1.addWidget(self._title_label)

        row1.addStretch()

        self._effort_badge = QtWidgets.QLabel("Torque (τ): —")
        row1.addWidget(self._effort_badge)

        self._rms_label = QtWidgets.QLabel("Valid Marker RMS: — mm")
        row1.addWidget(self._rms_label)

        self._report_btn = QtWidgets.QPushButton("Inspect Report")
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._on_inspect_report_clicked)
        row1.addWidget(self._report_btn)

        self._anim_btn = QtWidgets.QPushButton("Inspect Animation")
        self._anim_btn.setEnabled(False)
        self._anim_btn.clicked.connect(self._on_inspect_anim_clicked)
        row1.addWidget(self._anim_btn)

        self._open_btn = QtWidgets.QPushButton("Open Replay…")
        self._open_btn.clicked.connect(self._on_open_clicked)
        row1.addWidget(self._open_btn)

        layout.addLayout(row1)

        # Row 2: Catalog Selector and View Configuration
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Catalog:"))
        self._catalog_combo = QtWidgets.QComboBox()
        self._catalog_combo.setMinimumWidth(260)
        self._catalog_combo.currentIndexChanged.connect(self._on_catalog_selected)
        row2.addWidget(self._catalog_combo)

        row2.addSpacing(10)
        row2.addWidget(QtWidgets.QLabel("Render:"))
        self._render_mode_combo = QtWidgets.QComboBox()
        self._render_mode_combo.addItems(["Cylinders", "Line Skeleton"])
        self._render_mode_combo.currentIndexChanged.connect(
            self._on_render_mode_changed
        )
        row2.addWidget(self._render_mode_combo)

        self._error_overlay_check = QtWidgets.QCheckBox("Marker Error Vectors")
        self._error_overlay_check.setChecked(True)
        self._error_overlay_check.toggled.connect(self._on_error_overlay_toggled)
        row2.addWidget(self._error_overlay_check)

        row2.addSpacing(10)
        row2.addWidget(QtWidgets.QLabel("Camera:"))
        self._camera_combo = QtWidgets.QComboBox()
        self._camera_combo.addItems(
            [
                "Perspective (20°, 45°)",
                "Face-On (0°, 0°)",
                "Down-the-Line (0°, -90°)",
                "Top-Down (90°, 0°)",
            ]
        )
        self._camera_combo.currentIndexChanged.connect(self._on_camera_preset_changed)
        row2.addWidget(self._camera_combo)

        row2.addStretch()
        layout.addLayout(row2)

        # 3D Viewport
        self._figure = Figure(figsize=(6, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = cast(Axes3D, self._figure.add_subplot(111, projection="3d"))
        self._setup_3d_axes()
        layout.addWidget(self._canvas, stretch=1)

        # Controls bar: Restart, Play/Pause, Slider, Frame Label, Speed
        controls_layout = QtWidgets.QHBoxLayout()
        self._restart_btn = QtWidgets.QPushButton("⏮ Restart")
        self._restart_btn.setFixedWidth(80)
        self._restart_btn.clicked.connect(self.restart_playback)
        controls_layout.addWidget(self._restart_btn)

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

        controls_layout.addSpacing(8)
        controls_layout.addWidget(QtWidgets.QLabel("Speed:"))
        self._speed_combo = QtWidgets.QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x"])
        self._speed_combo.setCurrentIndex(2)  # Default 1.0x
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        controls_layout.addWidget(self._speed_combo)

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
            try:
                self._spec = json.loads(self._spec_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception(
                    "Failed to load specification from %s", self._spec_path
                )

    def _refresh_catalog(self) -> None:
        """Discover known manifests, register in store, and populate combo box."""
        self._catalog_combo.blockSignals(True)
        self._catalog_combo.clear()
        self._catalog_combo.addItem("-- Select Catalog Run --", None)

        try:
            discovered = self._store.discover_known_manifests()
            for m in discovered:
                try:
                    self._store.register_replay_bundle(m)
                except (OSError, ValueError, KeyError, RuntimeError):
                    logger.debug("Could not register manifest %s", m)
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.debug("Manifest discovery failed")

        entries = self._store.list_catalog_entries()
        for entry in entries:
            run_id = entry.get("run_id", "unknown")
            status = entry.get("status", "unknown")
            dur = entry.get("duration_s", 0.0)
            eng = entry.get("engine", "simscape").capitalize()
            label = f"{run_id} [{status}] ({dur:.3f} s, {eng})"
            self._catalog_combo.addItem(label, entry)

        self._catalog_combo.blockSignals(False)

    def load_replay_data(
        self,
        replay: ReplayData,
        *,
        candidate_hash: str = "unknown",
        engine_name: str = "unknown",
    ) -> None:
        """Populate viewer with pre-loaded replay data."""
        if self._is_playing:
            self.toggle_playback()
        self._clock.stop()
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
        """Load a verified Simscape manifest or a legacy replay archive."""
        p = Path(path).resolve()
        if p.suffix.lower() == ".json":
            bundle = load_simscape_bundle(p)
            arrays = bundle.arrays
            replay = ReplayData(
                time_s=arrays["time_s"],
                coordinates=arrays["q"],
                model_markers_m=arrays["markers_m"],
                target_markers_m=arrays["target_m"],
                valid_mask=arrays["valid"],
                coordinate_names=bundle.coordinate_names,
            )
            spec = dict(bundle.model)
            viewer_frame(spec, replay, 0)
            self._spec = spec
            if self._is_playing:
                self.toggle_playback()
            self.load_replay_data(
                replay, candidate_hash=bundle.run_id, engine_name="simscape"
            )

            # Store report metadata
            self._report_data = dict(bundle.report)
            if "report" in bundle.artifact_paths:
                self._report_path = bundle.artifact_paths["report"]
                self._report_btn.setEnabled(True)
            else:
                self._report_btn.setEnabled(False)

            # Resolve animation path
            repo_root = Path(__file__).resolve().parents[3]
            possible_anims = [
                p.parent.parent
                / "visuals_returned102"
                / f"{bundle.run_id.replace('-', '_')}_cylinders.gif",
                p.parent.parent
                / "visuals_returned102"
                / "simscape_returned102_cylinders.gif",
                repo_root
                / "docs/development/simscape_tour_matching/visuals_returned102/simscape_returned102_cylinders.gif",
            ]
            self._animation_path = None
            for anim in possible_anims:
                if anim.is_file():
                    self._animation_path = anim
                    break
            self._anim_btn.setEnabled(self._animation_path is not None)

            # Explicit effort display
            tau_valid = arrays["tau_valid"]
            if tau_valid.all():
                effort_text = "Recorded Torque Available"
                badge_style = "color: #27ae60; font-weight: bold;"
            elif tau_valid.any():
                effort_text = "Recorded Torque Partly Unavailable"
                badge_style = "color: #e67e22; font-weight: bold;"
            else:
                effort_text = "Recorded Torque Unavailable"
                badge_style = "color: #c0392b; font-weight: bold;"

            self._effort_badge.setText(f"Torque (τ): {effort_text}")
            self._effort_badge.setStyleSheet(badge_style)
            self._title_label.setText(
                f"{bundle.run_id} | Simscape | {bundle.status} | "
                f"{replay.time_s[-1]:.3f} s | {effort_text}"
            )
            return

        # Legacy / non-bundle file loading
        if self._spec is None:
            self._load_spec()
        replay = load_replay(p, self._spec)

        candidate_hash = "returned81" if "returned81" in p.stem else p.stem[:12]
        engine_name = "default"
        for eng in ("mujoco", "pinocchio", "drake", "opensim"):
            if eng in p.name.lower() or eng in str(p.parent).lower():
                engine_name = eng
                break

        self._report_data = None
        self._report_path = None
        self._animation_path = None
        self._report_btn.setEnabled(False)
        self._anim_btn.setEnabled(False)
        self._effort_badge.setText("Torque (τ): —")
        self._effort_badge.setStyleSheet("")

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

        camera = (self._ax.elev, self._ax.azim, self._ax.roll)
        limits = (self._ax.get_xlim(), self._ax.get_ylim(), self._ax.get_zlim())
        self._setup_3d_axes()
        self._ax.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        self._ax.set_xlim(limits[0])
        self._ax.set_ylim(limits[1])
        self._ax.set_zlim(limits[2])

        # 1. Render segments: Cylinders or Line Skeleton
        render_mode = self._render_mode_combo.currentText()
        eng_color = ENGINE_COLORS.get(self._engine_name, ENGINE_COLORS["default"])

        if render_mode == "Cylinders":
            polygons = []
            colors = []
            for seg in vframe.segments:
                faces = cylinder_faces(seg.start_m, seg.end_m, seg.radius_m)
                polygons.extend(faces)
                colors.extend([eng_color] * len(faces))
            if polygons:
                coll = Poly3DCollection(
                    polygons,
                    facecolors=colors,
                    edgecolors="#1e3d59",
                    linewidths=0.2,
                    alpha=0.9,
                )
                self._ax.add_collection3d(coll)
        else:
            lines = [[seg.start_m, seg.end_m] for seg in vframe.segments]
            if lines:
                line_coll = Line3DCollection(
                    lines, colors=eng_color, linewidths=2.5, alpha=0.85
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

        # 4. Render marker error vectors (if enabled)
        if (
            self._error_overlay_check.isChecked()
            and vframe.target_markers is not None
            and vframe.model_markers is not None
        ):
            err_lines = marker_error_vectors(
                vframe.target_markers,
                vframe.model_markers,
                vframe.valid_mask,
            )
            if err_lines:
                err_coll = Line3DCollection(
                    err_lines,
                    colors="#e74c3c",
                    linewidths=1.2,
                    linestyles="--",
                    alpha=0.85,
                )
                self._ax.add_collection3d(err_coll)

        self._canvas.draw_idle()

    def restart_playback(self) -> None:
        """Reset playback to frame 0 and source time 0.0 s."""
        was_playing = self._is_playing
        if self._is_playing:
            self.toggle_playback()
        self._clock.stop()
        self._slider.blockSignals(True)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self.render_frame(0)
        if was_playing:
            self.toggle_playback()

    def toggle_playback(self) -> None:
        """Toggle animation timer."""
        if self._is_playing:
            self._clock.pause()
            self._timer.stop()
            self._play_btn.setText("▶ Play")
            self._is_playing = False
        else:
            if self._replay is not None and self._replay.frame_count > 1:
                if self._current_frame == self._replay.frame_count - 1:
                    self._slider.setValue(0)
                    self.render_frame(0)
                self._clock.seek(
                    float(
                        self._replay.time_s[self._current_frame]
                        - self._replay.time_s[0]
                    )
                )
                self._clock.play()
                # Display ticks sample the source clock; frames may be skipped.
                self._timer.start(33)
                self._play_btn.setText("⏸ Pause")
                self._is_playing = True

    def _on_timer_tick(self) -> None:
        if self._replay is None or not self._is_playing:
            return
        _, elapsed_s, _ = self._clock.tick()
        times = self._replay.time_s
        source_s = float(times[0]) + elapsed_s
        next_frame = min(
            len(times) - 1,
            max(0, int(np.searchsorted(times, source_s, side="right")) - 1),
        )
        self._slider.blockSignals(True)
        self._slider.setValue(next_frame)
        self._slider.blockSignals(False)
        self.render_frame(next_frame)
        if source_s >= times[-1]:
            self.toggle_playback()

    def _on_slider_changed(self, value: int) -> None:
        if self._replay is not None:
            self._clock.seek(float(self._replay.time_s[value] - self._replay.time_s[0]))
        self.render_frame(value)

    def _on_speed_changed(self, index: int) -> None:
        rates = [0.25, 0.5, 1.0, 2.0]
        if 0 <= index < len(rates):
            self._clock.set_playback_rate(rates[index])

    def _on_camera_preset_changed(self, index: int) -> None:
        presets = [
            (20.0, 45.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, -90.0, 0.0),
            (90.0, 0.0, 0.0),
        ]
        if 0 <= index < len(presets):
            elev, azim, roll = presets[index]
            self._ax.view_init(elev=elev, azim=azim, roll=roll)
            self.render_frame(self._current_frame)

    def _on_render_mode_changed(self, index: int) -> None:  # noqa: ARG002
        self.render_frame(self._current_frame)

    def _on_error_overlay_toggled(self, checked: bool) -> None:  # noqa: ARG002
        self.render_frame(self._current_frame)

    def _on_catalog_selected(self, index: int) -> None:
        if index <= 0:
            return
        entry = self._catalog_combo.itemData(index)
        if isinstance(entry, dict) and "manifest_path" in entry:
            self.load_file(Path(entry["manifest_path"]))

    def _on_inspect_report_clicked(self) -> None:
        if self._report_data is not None:
            dlg = ReportInspectorDialog(
                self._report_data,
                run_id=self._candidate_hash,
                report_path=self._report_path,
                parent=self,
            )
            dlg.exec()

    def _on_inspect_anim_clicked(self) -> None:
        if self._animation_path is not None and self._animation_path.is_file():
            dlg = AnimationInspectorDialog(
                self._animation_path,
                run_id=self._candidate_hash,
                parent=self,
            )
            dlg.exec()

    def _on_open_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Tour Matching Replay",
            "",
            "Replay Files (*.json *.npz *.mot *.sto);;All Files (*)",
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
        self._clock.stop()
        self._timer.stop()
        self._is_playing = False
        self._figure.clf()


class TourMatchingViewerWindow(QtWidgets.QMainWindow):
    """Standalone main window hosting the Tour Matching Viewer."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        store: SimulationDataStore | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tour Matching Viewer")
        self.resize(1024, 768)
        self.widget = TourMatchingViewerWidget(self, store=store)
        self.setCentralWidget(self.widget)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:  # noqa: N802
        self.widget.cleanup()
        if event is not None:
            event.accept()
