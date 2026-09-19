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
    ENGINE_COLORS,
    MultiCandidateReplay,
    ReplayData,
    ViewerFrame,
    export_animation_gif,
    load_replay,
    viewer_frame,
)

logger = get_logger(__name__)

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
        self._multi_replay: MultiCandidateReplay | None = None
        self._current_frame: int = 0
        self._candidate_hash: str = "unknown"
        self._engine_name: str = "default"
        self._is_accepted: bool = True
        self._rejection_reason: str = ""
        self._supports_forces: bool = False
        self._supports_counterfactuals: bool = False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._is_playing: bool = False

        self._init_ui()
        self._load_spec()
        self._populate_runs_combo()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header bar 1: runs combo, title, RMS, GIF export, open button
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(QtWidgets.QLabel("Runs / Ledger:"))
        self._runs_combo = QtWidgets.QComboBox()
        self._runs_combo.setMinimumWidth(160)
        self._runs_combo.currentIndexChanged.connect(self._on_run_selected)
        header_layout.addWidget(self._runs_combo)

        self._title_label = QtWidgets.QLabel("Tour Matching Viewer — No replay loaded")
        font = self._title_label.font()
        font.setBold(True)
        self._title_label.setFont(font)
        header_layout.addWidget(self._title_label)

        header_layout.addStretch()

        self._rms_label = QtWidgets.QLabel("Valid Marker RMS: — mm")
        header_layout.addWidget(self._rms_label)

        self._export_gif_btn = QtWidgets.QPushButton("Export GIF…")
        self._export_gif_btn.clicked.connect(self._on_export_gif_clicked)
        header_layout.addWidget(self._export_gif_btn)

        self._open_btn = QtWidgets.QPushButton("Open Replay…")
        self._open_btn.clicked.connect(self._on_open_clicked)
        header_layout.addWidget(self._open_btn)

        layout.addLayout(header_layout)

        # Header bar 2: capabilities status label
        sub_layout = QtWidgets.QHBoxLayout()
        self._capabilities_label = QtWidgets.QLabel(
            "Capabilities: Forces: Unsupported | Counterfactuals: Unsupported"
        )
        self._capabilities_label.setStyleSheet("color: #555555; font-size: 11px;")
        sub_layout.addWidget(self._capabilities_label)
        sub_layout.addStretch()
        layout.addLayout(sub_layout)

        # Conspicuous Rejection Banner
        self._rejection_banner = QtWidgets.QLabel(
            "⚠ REJECTED CANDIDATE FIT — Visual inspection only; fit criteria not met"
        )
        self._rejection_banner.setStyleSheet(
            "background-color: #d9534f; color: white; font-weight: bold; padding: 4px 8px; border-radius: 4px;"
        )
        self._rejection_banner.setVisible(False)
        layout.addWidget(self._rejection_banner)

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

    def _populate_runs_combo(self) -> None:
        """Populate runs combo from reports/matched_swing_ledger.json if available."""
        ledger_path = ROOT / "reports" / "matched_swing_ledger.json"
        self._runs_combo.blockSignals(True)
        self._runs_combo.clear()
        self._runs_combo.addItem("Select run / candidate…", None)
        if ledger_path.exists():
            try:
                import json

                data = json.loads(ledger_path.read_text(encoding="utf-8"))
                rows = data.get("rows", [])
                for i, row in enumerate(rows):
                    eng = row.get("engine", "engine")
                    lane = row.get("lane", f"run_{i}")
                    sha = str(row.get("sha256", ""))[:8]
                    label = f"{eng} | {lane} | {sha}"
                    self._runs_combo.addItem(label, row)
            except (OSError, ValueError, KeyError):
                logger.exception("Failed to parse matched_swing_ledger.json")
        self._runs_combo.blockSignals(False)

    def _on_run_selected(self, index: int) -> None:
        if index <= 0:
            return
        row = self._runs_combo.itemData(index)
        if not isinstance(row, dict):
            return

        artefacts = row.get("artefacts", {}) or {}
        cand_path = artefacts.get("npz") or artefacts.get("mot")
        receipt_path = row.get("receipt_path")

        is_accepted = row.get("acceptance") != "rejected"
        rejection_reason = str(row.get("reason") or "")
        self.set_acceptance(is_accepted, rejection_reason)

        if cand_path:
            full_cand = (
                ROOT / cand_path
                if not Path(cand_path).is_absolute()
                else Path(cand_path)
            )
            if full_cand.exists():
                full_rcpt = (
                    (ROOT / receipt_path)
                    if receipt_path and (ROOT / receipt_path).exists()
                    else None
                )
                try:
                    from src.shared.python.motion_matching.candidate_session import (
                        ingest_candidate_session,
                    )

                    session = ingest_candidate_session(
                        candidate_path=full_cand,
                        model_path=self._spec_path,
                        receipt_path=full_rcpt,
                    )
                    self.load_candidate_session(session)
                    return
                except (OSError, ValueError, RuntimeError, KeyError):
                    logger.debug("Falling back to load_file for %s", full_cand)
                    self.load_file(full_cand)

    def set_acceptance(self, is_accepted: bool, reason: str = "") -> None:
        """Update acceptance status and conspicuous rejection banner."""
        self._is_accepted = is_accepted
        self._rejection_reason = reason
        if not is_accepted:
            msg = (
                f"⚠ REJECTED CANDIDATE FIT — Visual inspection only "
                f"({reason or 'fit criteria not met'})"
            )
            self._rejection_banner.setText(msg)
            self._rejection_banner.setVisible(True)
        else:
            self._rejection_banner.setVisible(False)

    def load_candidate_session(self, session: Any) -> None:
        """Populate viewer with a qualified CandidateSession."""
        self.set_acceptance(session.is_accepted, session.rejection_reason)
        self._supports_forces = session.supports_forces
        self._supports_counterfactuals = session.supports_counterfactuals

        forces_str = (
            "Supported" if session.supports_forces else "Unsupported (kinematic only)"
        )
        cf_str = "Supported" if session.supports_counterfactuals else "Unsupported"
        self._capabilities_label.setText(
            f"Capabilities: Forces: {forces_str} | Counterfactuals: {cf_str}"
        )

        replay = session.replay
        cand_hash = getattr(session, "candidate_sha256", "unknown")[:12]
        eng = getattr(session, "engine", "default")
        self.load_replay_data(replay, candidate_hash=cand_hash, engine_name=eng)

    def load_multi_candidates(
        self,
        candidates: list[tuple[str, ReplayData]] | tuple[tuple[str, ReplayData], ...],
    ) -> None:
        """Load up to 4 candidate replays overlaid on a shared timeline."""
        self._multi_replay = MultiCandidateReplay(candidates=tuple(candidates))
        self._replay = self._multi_replay.candidates[0][1]
        self._engine_name = "multi"
        self._current_frame = 0

        n_frames = self._multi_replay.frame_count
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, n_frames - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        names = ", ".join(eng.capitalize() for eng, _ in self._multi_replay.candidates)
        self._title_label.setText(
            f"Multi-Candidate Replay ({len(self._multi_replay.candidates)}) | Engines: {names}"
        )
        self.render_frame(0)

    def load_replay_data(
        self,
        replay: ReplayData,
        *,
        candidate_hash: str = "returned81",
        engine_name: str = "mujoco",
    ) -> None:
        """Populate viewer with pre-loaded replay data."""
        self._multi_replay = None
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

    def _render_multi_frame(self, frame_idx: int) -> None:
        """Render overlaid candidate models and compute per-engine marker RMS."""
        if self._multi_replay is None or self._spec is None:
            return

        rms_parts = []
        for engine, rep in self._multi_replay.candidates:
            if frame_idx >= rep.frame_count:
                continue
            vframe = viewer_frame(self._spec, rep, frame_idx)
            eng_color = ENGINE_COLORS.get(engine, ENGINE_COLORS["default"])
            lines = [[seg.start_m, seg.end_m] for seg in vframe.segments]
            if lines:
                self._ax.add_collection3d(
                    Line3DCollection(lines, colors=eng_color, linewidths=2.0, alpha=0.8)
                )
            if vframe.model_markers is not None:
                self._ax.scatter(
                    vframe.model_markers[:, 0],
                    vframe.model_markers[:, 1],
                    vframe.model_markers[:, 2],
                    color=eng_color,
                    s=20,
                    label=f"{engine.capitalize()} ({vframe.rms_error * 1000.0:.1f} mm)",
                    alpha=0.9,
                )
            rms_parts.append(
                f"{engine[:3].capitalize()}: {vframe.rms_error * 1000.0:.1f} mm"
            )

        # Target markers from first candidate
        first_rep = self._multi_replay.candidates[0][1]
        if first_rep.target_markers_m is not None and frame_idx < first_rep.frame_count:
            tm = first_rep.target_markers_m[frame_idx]
            vm = (
                first_rep.valid_mask[frame_idx]
                if first_rep.valid_mask is not None
                else None
            )
            valid_tm = tm[vm] if vm is not None else tm
            if len(valid_tm) > 0:
                self._ax.scatter(
                    valid_tm[:, 0],
                    valid_tm[:, 1],
                    valid_tm[:, 2],
                    color="black",
                    s=18,
                    label="Target Markers",
                    alpha=0.7,
                )

        self._rms_label.setText("RMS: " + " | ".join(rms_parts))

    def _render_single_frame(self, frame_idx: int) -> None:
        """Render single replay model and target markers."""
        if self._replay is None or self._spec is None:
            return

        vframe: ViewerFrame = viewer_frame(self._spec, self._replay, frame_idx)
        self._rms_label.setText(f"Valid Marker RMS: {vframe.rms_error * 1000.0:.2f} mm")

        # 1. Render visual skeleton segments as 3D lines
        lines = [[seg.start_m, seg.end_m] for seg in vframe.segments]
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

    def render_frame(self, frame_idx: int) -> None:
        """Evaluate kinematics and render 3D elements for a given frame index."""
        if (self._replay is None and self._multi_replay is None) or self._spec is None:
            return

        n_frames = (
            self._multi_replay.frame_count
            if self._multi_replay is not None
            else self._replay.frame_count  # type: ignore[union-attr]
        )
        if frame_idx < 0 or frame_idx >= n_frames:
            return

        self._current_frame = frame_idx
        ref_replay = (
            self._multi_replay.candidates[0][1]
            if self._multi_replay is not None
            else self._replay
        )
        t = float(ref_replay.time_s[frame_idx])  # type: ignore[union-attr]
        self._frame_label.setText(f"Frame: {frame_idx + 1} / {n_frames} ({t:.3f} s)")

        self._setup_3d_axes()
        if self._multi_replay is not None:
            self._render_multi_frame(frame_idx)
        else:
            self._render_single_frame(frame_idx)

        self._canvas.draw_idle()

    def _on_export_gif_clicked(self) -> None:
        """Export current replay (single or multi-candidate) as an animated GIF."""
        active = self._multi_replay or self._replay
        if active is None or self._spec is None:
            QtWidgets.QMessageBox.warning(
                self, "Export GIF", "No candidate replay is currently loaded."
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Swing Animation GIF",
            "matched_swing.gif",
            "GIF Files (*.gif);;All Files (*)",
        )
        if path:
            try:
                out_path = export_animation_gif(
                    active,
                    spec=self._spec,
                    output_path=Path(path),
                    fps=30,
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "GIF Exported",
                    f"Successfully exported animation GIF to:\n{out_path}",
                )
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Failed", f"Failed to export animation GIF:\n{e}"
                )

    def toggle_playback(self) -> None:
        """Toggle animation timer."""
        if self._is_playing:
            self._timer.stop()
            self._play_btn.setText("▶ Play")
            self._is_playing = False
        else:
            active_rep = self._multi_replay or self._replay
            if active_rep is not None and active_rep.frame_count > 1:
                # Target ~30 fps playback
                self._timer.start(33)
                self._play_btn.setText("⏸ Pause")
                self._is_playing = True

    def _on_timer_tick(self) -> None:
        active_rep = self._multi_replay or self._replay
        if active_rep is None:
            return
        n_frames = active_rep.frame_count
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
