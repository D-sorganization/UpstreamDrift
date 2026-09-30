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
    ResidualSummary,
    ViewerFrame,
    compute_residual_summary,
    export_animation_gif,
    load_replay,
    viewer_frame,
)


def _ensure_tools_path() -> None:
    try:
        import rate_of_closure.simulation.playback_transport  # noqa: F401
    except ImportError:
        import sys

        root = Path(__file__).resolve().parents[3]
        for p in (root / "vendor" / "ud-tools" / "src", root.parent / "Tools" / "src"):
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
                break


_ensure_tools_path()

from rate_of_closure.ui.pyqt6.playback_transport_controls import (  # noqa: E402
    PlaybackTransportControls,
)
from src.shared.python.motion_matching.playback import (  # noqa: E402
    InterpolatedPlaybackState,
    PhysicalTimePlayback,
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
        self._drive_mode: str = "kinematic_prescribed"
        self._camera_view: str = "perspective"
        self._appearance_preset: str = "default"
        self._current_rms_error: float = 0.0
        self._physics_score: float = 1.0
        self._residual_summary: ResidualSummary | None = None
        self._show_residual_vectors: bool = False
        self._is_accepted: bool = True
        self._rejection_reason: str = ""
        self._supports_forces: bool = False
        self._supports_counterfactuals: bool = False
        self._playback: PhysicalTimePlayback | None = None
        self._is_playing: bool = False

        self._init_ui()
        self._load_spec()
        self._populate_runs_combo()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._build_header_bars(layout)
        self._build_viewport_layout(layout)
        self._build_transport_controls(layout)

    def _build_header_bars(self, layout: QtWidgets.QVBoxLayout) -> None:
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

        self._open_native_btn = QtWidgets.QPushButton("Open Native…")
        self._open_native_btn.setToolTip(
            "Launch current candidate in native 3D engine (MeshCat, Gepetto, MuJoCo, etc.)"
        )
        self._open_native_btn.clicked.connect(self._on_open_native_clicked)
        header_layout.addWidget(self._open_native_btn)
        layout.addLayout(header_layout)

        # Header bar 2: capabilities status label + camera, presets, and worst residual jump
        sub_layout = QtWidgets.QHBoxLayout()
        self._capabilities_label = QtWidgets.QLabel(
            "Capabilities: Forces: Unsupported | Counterfactuals: Unsupported"
        )
        self._capabilities_label.setStyleSheet("color: gray; font-size: 11px;")
        sub_layout.addWidget(self._capabilities_label)
        sub_layout.addStretch()

        sub_layout.addWidget(QtWidgets.QLabel("View:"))
        self._camera_combo = QtWidgets.QComboBox()
        self._camera_combo.addItems(
            ["perspective", "front", "side", "top", "isometric"]
        )
        self._camera_combo.currentTextChanged.connect(self.set_camera_view)
        sub_layout.addWidget(self._camera_combo)

        sub_layout.addWidget(QtWidgets.QLabel("Preset:"))
        self._appearance_combo = QtWidgets.QComboBox()
        self._appearance_combo.addItems(
            ["default", "high_contrast", "residual_vectors", "dots_and_mesh"]
        )
        self._appearance_combo.currentTextChanged.connect(self.set_appearance_preset)
        sub_layout.addWidget(self._appearance_combo)

        self._worst_residual_label = QtWidgets.QLabel("Worst: —")
        self._worst_residual_label.setStyleSheet(
            "color: darkred; font-size: 11px; font-weight: bold;"
        )
        sub_layout.addWidget(self._worst_residual_label)
        self._select_worst_btn = QtWidgets.QPushButton("Worst Residual")
        self._select_worst_btn.setToolTip(
            "Jump to the frame and phase with the maximum marker residual"
        )
        self._select_worst_btn.clicked.connect(self.select_worst_residual)
        sub_layout.addWidget(self._select_worst_btn)
        layout.addLayout(sub_layout)

        # Conspicuous Rejection Banner
        self._rejection_banner = QtWidgets.QLabel(
            "⚠ REJECTED CANDIDATE FIT — Visual inspection only; fit criteria not met"
        )
        self._rejection_banner.setStyleSheet(
            "background-color: darkred; color: white; font-weight: bold; padding: 4px 8px; border-radius: 4px;"
        )
        self._rejection_banner.setVisible(False)
        layout.addWidget(self._rejection_banner)

    def _build_viewport_layout(self, layout: QtWidgets.QVBoxLayout) -> None:
        content_layout = QtWidgets.QHBoxLayout()
        self._figure = Figure(figsize=(6, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax: Any = self._figure.add_subplot(111, projection="3d")
        self._setup_3d_axes()
        content_layout.addWidget(self._canvas, stretch=3)

        from src.tools.tour_matching_viewer.force_inspection import (
            ForceInspectionWidget,
        )

        self._force_widget = ForceInspectionWidget(self)
        self._force_widget.setMaximumWidth(280)
        content_layout.addWidget(self._force_widget, stretch=1)
        layout.addLayout(content_layout, stretch=1)

    def _build_transport_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._transport = PlaybackTransportControls(
            subject_label="Swing",
            subject_phrase="swing",
            event_labels=("Address", "Top", "Impact", "Finish"),
            scrub_tooltip="Scrub physical swing time [s] from address to finish.",
            help_text="Physical time authority (1x source in 1.814s) with quaternion SLERP. Drag to orbit; wheel to zoom.",
            help_tooltip="Physical seconds along the swing timeline.",
            parent=self,
        )
        self._transport.timeChanged.connect(self._on_transport_time_changed)
        layout.addWidget(self._transport)

        # Backwards-compatible aliases for legacy properties and tests
        self._play_btn = self._transport.play_button
        self._slider = self._transport.scrubber
        self._frame_label = self._transport.time_label

    @property
    def current_frame(self) -> int:
        """Current frame index."""
        return self._current_frame

    @property
    def current_rendered_frame_index(self) -> int:
        """Current rendered discrete frame index (Criterion 5: clock alignment)."""
        return self._current_frame

    @property
    def current_frame_time_s(self) -> float:
        """Timestamp in seconds of the current rendered frame."""
        if (
            self._replay is not None
            and 0 <= self._current_frame < self._replay.frame_count
        ):
            return float(self._replay.time_s[self._current_frame])
        return 0.0

    @property
    def current_rms_error(self) -> float:
        """Current marker RMS error in meters."""
        return self._current_rms_error

    @property
    def physics_score(self) -> float:
        """Objective physics agreement score [0, 1] derived from residual summary."""
        return self._physics_score

    @property
    def candidate_hash(self) -> str:
        """Current candidate SHA-256 or identifier."""
        return self._candidate_hash

    @property
    def drive_mode(self) -> str:
        """Current candidate drive mode."""
        return self._drive_mode

    @property
    def title_caption(self) -> str:
        """Title banner caption."""
        return self._title_label.text()

    @property
    def capabilities_caption(self) -> str:
        """Capabilities and drive mode sub-caption."""
        return self._capabilities_label.text()

    @property
    def force_widget(self) -> Any:
        """Force/torque and counterfactual inspection widget."""
        return self._force_widget

    def select_worst_residual(self) -> None:
        """Jump to the exact frame and phase with the worst residual error."""
        if self._residual_summary is None and self._replay is not None:
            self._residual_summary = compute_residual_summary(self._replay)
        if self._residual_summary is None or self._replay is None:
            return

        worst_idx = self._residual_summary.worst_frame_idx
        if 0 <= worst_idx < self._replay.frame_count:
            t = float(self._replay.time_s[worst_idx])
            if hasattr(self, "_transport") and self._transport is not None:
                self._transport.blockSignals(True)
                self._transport.jump_to_time(t)
                self._transport.blockSignals(False)
            self._current_frame = worst_idx
            self.render_frame(worst_idx)

    def set_camera_view(self, view: str) -> None:
        """Set camera view preset ('perspective', 'front', 'side', 'top', 'isometric')."""
        self._camera_view = view.lower()
        if (
            hasattr(self, "_camera_combo")
            and self._camera_combo.currentText().lower() != self._camera_view
        ):
            self._camera_combo.blockSignals(True)
            idx = self._camera_combo.findText(
                self._camera_view, QtCore.Qt.MatchFlag.MatchFixedString
            )
            if idx >= 0:
                self._camera_combo.setCurrentIndex(idx)
            self._camera_combo.blockSignals(False)
        self.render_frame(self._current_frame)

    def set_appearance_preset(self, preset: str) -> None:
        """Set appearance preset ('default', 'high_contrast', 'residual_vectors', 'dots_and_mesh')."""
        self._appearance_preset = preset.lower()
        if (
            hasattr(self, "_appearance_combo")
            and self._appearance_combo.currentText().lower() != self._appearance_preset
        ):
            self._appearance_combo.blockSignals(True)
            idx = self._appearance_combo.findText(
                self._appearance_preset, QtCore.Qt.MatchFlag.MatchFixedString
            )
            if idx >= 0:
                self._appearance_combo.setCurrentIndex(idx)
            self._appearance_combo.blockSignals(False)
        self.render_frame(self._current_frame)

    def _setup_3d_axes(self) -> None:
        self._ax.clear()
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")
        self._ax.set_xlim(-1.5, 1.5)
        self._ax.set_ylim(-1.5, 1.5)
        self._ax.set_zlim(0.0, 2.0)

        # Apply camera view preset (elev, azim)
        view = getattr(self, "_camera_view", "perspective")
        if view == "front":
            self._ax.view_init(elev=0, azim=-90)
        elif view == "side":
            self._ax.view_init(elev=0, azim=0)
        elif view == "top":
            self._ax.view_init(elev=90, azim=-90)
        elif view == "isometric":
            self._ax.view_init(elev=30, azim=45)
        else:  # "perspective"
            self._ax.view_init(elev=20, azim=45)

        # Draw ground grid at Z = 0
        gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 7), np.linspace(-1.5, 1.5, 7))
        gz = np.zeros_like(gx)
        grid_color = "lightgray"
        if getattr(self, "_appearance_preset", "default") == "high_contrast":
            grid_color = "dimgray"
        self._ax.plot_wireframe(gx, gy, gz, color=grid_color, linewidth=0.5, alpha=0.5)

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
        self._force_widget.set_candidate_session(session)

    def _setup_playback_transport(self, replay: ReplayData) -> None:
        """Initialize physical playback engine and configure transport controls."""
        time_s = np.asarray(replay.time_s, dtype=np.float64)
        n_frames = len(time_s)
        event_indices = {
            "Address": 0,
            "Top": max(0, min(int(0.35 * (n_frames - 1)), n_frames - 1)),
            "Impact": max(0, min(int(0.60 * (n_frames - 1)), n_frames - 1)),
            "Finish": max(0, n_frames - 1),
        }
        raw_coords = getattr(replay, "coordinates", getattr(replay, "q", None))
        coords = (
            np.asarray(raw_coords, dtype=np.float64)
            if raw_coords is not None
            else np.zeros((n_frames, 0), dtype=np.float64)
        )
        self._playback = PhysicalTimePlayback(
            times_s=time_s,
            q=coords,
            model_markers=replay.model_markers_m,
            target_markers=replay.target_markers_m,
            event_indices=event_indices,
        )
        duration_s = self._playback.duration_s
        events = self._playback.event_times
        event_times_s = (
            events["Address"],
            events["Top"],
            events["Impact"],
            events["Finish"],
        )
        self._transport.set_transport_timeline(duration_s, event_times_s)

    def load_multi_candidates(
        self,
        candidates: list[tuple[str, ReplayData]] | tuple[tuple[str, ReplayData], ...],
    ) -> None:
        """Load up to 4 candidate replays overlaid on a shared timeline."""
        self._multi_replay = MultiCandidateReplay(candidates=tuple(candidates))
        self._replay = self._multi_replay.candidates[0][1]
        self._engine_name = "multi"
        self._current_frame = 0

        self._setup_playback_transport(self._replay)

        names = ", ".join(eng.capitalize() for eng, _ in self._multi_replay.candidates)
        self._title_label.setText(
            f"Multi-Candidate Replay ({len(self._multi_replay.candidates)}) | Engines: {names}"
        )
        self.render_at_time(0.0)

    def load_replay_data(
        self,
        replay: ReplayData,
        *,
        candidate_hash: str = "returned81",
        engine_name: str = "mujoco",
        drive_mode: str | None = None,
    ) -> None:
        """Populate viewer with pre-loaded replay data."""
        self._multi_replay = None
        self._replay = replay
        self._candidate_hash = candidate_hash
        self._engine_name = engine_name.lower()
        self._drive_mode = (
            drive_mode
            if drive_mode is not None
            else getattr(replay, "drive_mode", "kinematic_prescribed")
        )
        self._current_frame = 0

        self._residual_summary = compute_residual_summary(replay)
        self._current_rms_error = self._residual_summary.mean_rms_m
        self._physics_score = float(
            max(0.0, 1.0 - self._residual_summary.mean_rms_m * 20.0)
        )

        self._setup_playback_transport(replay)

        mode_formatted = self._drive_mode.replace("_", " ").title()
        title = (
            f"Candidate: {self._candidate_hash} | Engine: {self._engine_name.capitalize()} | "
            f"Drive Mode: {mode_formatted}"
        )
        self._title_label.setText(title)

        forces_str = "Supported" if self._supports_forces else "Unsupported"
        cf_str = "Supported" if self._supports_counterfactuals else "Unsupported"
        self._capabilities_label.setText(
            f"Drive Mode: {mode_formatted} | Capabilities: Forces: {forces_str} | Counterfactuals: {cf_str}"
        )

        if self._residual_summary is not None:
            self._worst_residual_label.setText(
                f"Worst: Frame {self._residual_summary.worst_frame_idx + 1} "
                f"({self._residual_summary.worst_phase}) - "
                f"{self._residual_summary.max_marker_error_m * 1000.0:.1f} mm"
            )

        self.render_at_time(0.0)

    def load_file(
        self,
        path: Path | str,
        *,
        candidate_hash: str | None = None,
        engine_name: str | None = None,
        drive_mode: str | None = None,
        is_accepted: bool | None = None,
        rejection_reason: str = "",
    ) -> None:
        """Load a replay file (.npz or .mot).

        Filename-derived candidate/engine placeholders are fallbacks only;
        callers that know the selected ``LedgerRow``'s receipt provenance
        (candidate hash, engine, drive mode, verdict) must pass it here so
        captions and the rejection banner match the selected receipt.
        """
        p = Path(path)
        if self._spec is None:
            self._load_spec()
        replay = load_replay(p, self._spec)

        # Infer candidate hash and engine name from path (fallbacks only)
        if candidate_hash is None:
            candidate_hash = "returned81" if "returned81" in p.stem else p.stem[:12]
        if engine_name is None:
            engine_name = "default"
            for eng in ("mujoco", "pinocchio", "drake", "opensim"):
                if eng in p.name.lower() or eng in str(p.parent).lower():
                    engine_name = eng
                    break

        self.load_replay_data(
            replay,
            candidate_hash=candidate_hash,
            engine_name=engine_name,
            drive_mode=drive_mode,
        )
        if is_accepted is not None:
            self.set_acceptance(is_accepted, rejection_reason)

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
        self._current_rms_error = float(vframe.rms_error)
        self._rms_label.setText(f"Valid Marker RMS: {vframe.rms_error * 1000.0:.2f} mm")

        preset = getattr(self, "_appearance_preset", "default")

        # 1. Render visual skeleton segments as 3D lines
        skeleton_color = "cyan" if preset == "high_contrast" else "steelblue"
        lines = [[seg.start_m, seg.end_m] for seg in vframe.segments]
        if lines:
            line_coll = Line3DCollection(
                lines, colors=skeleton_color, linewidths=2.5, alpha=0.85
            )
            self._ax.add_collection3d(line_coll)

        # 2. Render target markers (black or magenta)
        tm_color = "magenta" if preset == "high_contrast" else "black"
        if vframe.target_markers is not None:
            valid = (
                vframe.valid_mask
                if vframe.valid_mask is not None
                else np.ones(len(vframe.target_markers), dtype=bool)
            )
            tm = vframe.target_markers[valid]
            if len(tm) > 0:
                s_size = 28 if preset == "dots_and_mesh" else 18
                self._ax.scatter(
                    tm[:, 0],
                    tm[:, 1],
                    tm[:, 2],
                    color=tm_color,
                    s=s_size,
                    label="Target Markers",
                    alpha=0.7,
                )

        # 3. Render model markers (engine colour)
        if vframe.model_markers is not None:
            eng_color = ENGINE_COLORS.get(self._engine_name, ENGINE_COLORS["default"])
            mm = vframe.model_markers
            s_size = 32 if preset == "dots_and_mesh" else 22
            self._ax.scatter(
                mm[:, 0],
                mm[:, 1],
                mm[:, 2],
                color=eng_color,
                s=s_size,
                label=f"Model ({self._engine_name})",
                alpha=0.9,
            )

        # 4. Render residual vectors (honest residuals: MMR-16)
        if preset == "residual_vectors" or getattr(
            self, "_show_residual_vectors", False
        ):
            if vframe.target_markers is not None and vframe.model_markers is not None:
                valid = (
                    vframe.valid_mask
                    if vframe.valid_mask is not None
                    else np.ones(len(vframe.target_markers), dtype=bool)
                )
                res_lines = [
                    [vframe.target_markers[i], vframe.model_markers[i]]
                    for i in range(
                        min(len(vframe.target_markers), len(vframe.model_markers))
                    )
                    if valid[i]
                ]
                if res_lines:
                    self._ax.add_collection3d(
                        Line3DCollection(
                            res_lines,
                            colors="firebrick",
                            linewidths=1.8,
                            linestyles="dashed",
                            alpha=0.9,
                        )
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
        ref_replay = self._get_active_candidate()
        t = float(ref_replay.time_s[frame_idx]) if ref_replay is not None else 0.0
        self._frame_label.setText(f"Frame: {frame_idx + 1} / {n_frames} ({t:.3f} s)")

        self._setup_3d_axes()
        if self._multi_replay is not None:
            self._render_multi_frame(frame_idx)
        else:
            self._render_single_frame(frame_idx)

        self._force_widget.update_frame(frame_idx)
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
        """Toggle animation playback using physical transport timer."""
        timer = self._transport.timer()
        if timer.isActive():
            self._transport.pause()
            self._is_playing = False
        else:
            self._transport.play()
            self._is_playing = True

    def _on_transport_time_changed(self, time_s: float) -> None:
        self.render_at_time(time_s)

    def render_at_time(self, time_s: float) -> None:
        """Evaluate continuous trajectory and render at physical time ``time_s``."""
        if self._playback is None:
            return
        state = self._playback.interpolate(time_s)
        self._current_frame = state.lower_index
        self.render_frame(state.lower_index)

    def _on_slider_changed(self, value: int) -> None:
        if self._playback is not None:
            t = self._playback.time_at_scrub(value)
            self.render_at_time(t)
        else:
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

    def _get_active_candidate(self) -> ReplayData | None:
        """Return the primary active replay candidate or None."""
        if self._multi_replay is not None and self._multi_replay.candidates:
            return self._multi_replay.candidates[0][1]
        return self._replay

    def _require_active_candidate(
        self, dialog_title: str = "Open Native Viewer"
    ) -> ReplayData | None:
        """Return active replay or display an informational dialog if none is loaded."""
        active = self._get_active_candidate()
        if active is None:
            QtWidgets.QMessageBox.information(
                self,
                dialog_title,
                "No candidate trajectory is currently loaded.",
            )
            return None
        return active

    def _show_recovery_message(self, message: str) -> None:
        """Display recovery / warning dialog when backend fails or is missing."""
        QtWidgets.QMessageBox.warning(self, "Native Viewer Recovery", message)

    def launch_native_backend(self, engine: str) -> None:
        """Launch specified native 3D engine backend with missing-engine recovery."""
        active = self._require_active_candidate("Open Native Viewer")
        if active is None:
            return

        from src.shared.python.motion_matching.native_viewers import (
            ViewerLaunchConfig,
            ViewerUnavailableError,
            open_in_native_viewer,
        )
        from src.shared.python.motion_matching.visualization.simulation_viewer import (
            SimulationData,
        )

        try:
            times = active.time_s
            q_matrix = active.coordinates
            sim_data = SimulationData(
                time_s=times,
                q=q_matrix,
                markers_m=active.model_markers_m,
                target_m=active.target_markers_m,
                valid=active.valid_mask,
                coordinate_order=list(active.coordinate_names)
                if active.coordinate_names
                else None,
            )
            cfg = ViewerLaunchConfig(speed=1.0, view_mode="fitted")
            res = open_in_native_viewer(sim_data, engine, config=cfg)
            if res.url:
                QtWidgets.QMessageBox.information(
                    self, "Native Viewer Launched", f"Viewer URL:\n{res.url}"
                )
        except ViewerUnavailableError as exc:
            logger.warning("Native viewer engine '%s' unavailable: %s", engine, exc)
            self._show_recovery_message(
                f"Native viewer engine '{engine.capitalize()}' is unavailable: {exc}"
            )
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("Failed to launch native viewer '%s': %s", engine, exc)
            self._show_recovery_message(
                f"Failed to launch native viewer '{engine.capitalize()}': {exc}"
            )

    def _on_open_native_clicked(self) -> None:
        """Launch the current candidate in a registered native viewer backend."""
        active = self._require_active_candidate("Open Native Viewer")
        if active is None:
            return

        from src.shared.python.motion_matching.native_viewers import (
            get_supported_backends,
        )

        backends = get_supported_backends()
        engine, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Select Native Viewer",
            "Choose 3D visualizer backend:",
            backends,
            0,
            False,
        )
        if not ok or not engine:
            return

        self.launch_native_backend(engine)

    def cleanup(self) -> None:
        """Halt playback and release resources."""
        self._transport.pause()
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
