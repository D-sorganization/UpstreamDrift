"""Embeddable widget shell for the Starting-Pose Matcher.

Subtask 5 / #4998 of EPIC #4993 refactors this tool so it can launch as
either a standalone :class:`QMainWindow` *or* an embedded ``QWidget``
inside the launcher host (tab / dock).

The historical entry point — :class:`StartingPoseMatcher` in
:mod:`gui` — still exists for back-compat, but its body now delegates
the actual UI to :class:`MainWidget` defined here. Embeddable hosts
construct :class:`MainWidget` directly and never see the
``QMainWindow`` shell.

The implementation is split across mixin modules to keep each file
under the 1200-line budget enforced by
``scripts/ci/check_file_size_budget.py``:

* :mod:`gui_render_mixin` — matplotlib drawing + residual helpers.
* :mod:`gui_builders_mixin` — the per-section ``_build_*_box`` helpers.
* :mod:`gui_session_mixin` — JSON save/load round-trip.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # noqa: F401 — preserved for downstream typing parity
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.tools.starting_pose_matcher.core import (
    DEFAULT_EVENT_PRESET as _DEFAULT_EVENT_PRESET,
    DEFAULT_PHASE as _DEFAULT_PHASE,
    EVENT_KEYS as _EVENT_KEYS,
    EVENT_LABEL_PRESETS as _EVENT_LABEL_PRESETS,
    PHASE_KEYS as _PHASE_KEYS,
    MocapEvents,
    PoseSlot,
    RigidTransform,
    SkeletonTrajectory,
    load_mocap_xlsx,
    load_simscape_trajectory_csv,
    load_skeleton,
    phase_display_label as _phase_display_label,
    phase_key_from_label as _phase_key_from_label,
    read_event_header,
    solve_shaft_rz_deg,
)
from src.tools.starting_pose_matcher.gui_source_panel import DataSourcesPanel
from src.tools.starting_pose_matcher.live_view_controller import LiveViewController
from src.tools.starting_pose_matcher.session_schema import (
    BODY_SKELETON_STYLES as _BODY_SKELETON_STYLES,
    BodySkeletonBlock,
    BodySkeletonStyleLiteral,
    DEFAULT_BODY_SKELETON_STYLE as _DEFAULT_BODY_SKELETON_STYLE,
    DataSourcesBlock,
    parse_body_skeleton,
    parse_data_sources,
    serialize_data_sources,
)
from src.tools.starting_pose_matcher.skeleton_extractor import (
    JsonSkeletonExtractor,
    SkeletonExtractor,
)

from ._gui_common import (
    _BODY_SKELETON_STYLE_LABELS,
    _BODY_SKELETON_STYLE_LABEL_BY_KEY,
    _DEFAULT_CAMERA,
    _S_RANGE,
    _S_SCALE,
)
from .gui_builders_mixin import _BuildersMixin
from .gui_render_mixin import _RenderMixin
from .gui_session_mixin import _SessionMixin

logger = logging.getLogger(__name__)

__all__ = ["MainWidget"]


class MainWidget(_RenderMixin, _BuildersMixin, _SessionMixin, QWidget):
    """Embeddable Starting-Pose Matcher widget.

    Composes the 3D matplotlib viewport, the per-section control panel
    (built by :class:`_BuildersMixin`), the rendering helpers
    (:class:`_RenderMixin`) and the session save/load helpers
    (:class:`_SessionMixin`).

    The widget can be parented to any :class:`QWidget`; when the
    launcher host embeds it as a tab or dock, no top-level window is
    created.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Data
        self.df: pd.DataFrame | None = None
        self.events = MocapEvents()
        self._xlsx_path: str | None = None

        here = Path(__file__).parent
        # Default extractor: JSON-based Simscape skeleton loader
        self.skeleton_extractor: SkeletonExtractor = JsonSkeletonExtractor(
            here, poses=("TopofBackswing", "Impact")
        )
        self.poses: dict[str, PoseSlot] = {
            "TopofBackswing": PoseSlot(
                name="TopofBackswing",
                skeleton=self.skeleton_extractor.get_skeleton("TopofBackswing"),
                color="#5b9eff",
                mocap_color="#ef4444",
                target_event="T",
            ),
            "Impact": PoseSlot(
                name="Impact",
                skeleton=self.skeleton_extractor.get_skeleton("Impact"),
                color="#10b981",
                mocap_color="#f59e0b",
                target_event="I",
            ),
        }
        self.transform = RigidTransform()
        for slot in self.poses.values():
            if "hub" in slot.skeleton.joints:
                self.transform.pivot = tuple(slot.skeleton.joints["hub"])
                break

        self.show_clubhead_trace = False
        self.show_midhands_trace = False
        self.show_ball = True
        self.show_ground = True
        self.show_torso_disk = True  # disc indicator at torso joint
        self.lock_xy_rotation = True  # Rx/Ry locked by default
        self.auto_fit_axes = True  # use shared equalize_3d_axes per redraw

        # Playback state
        self.current_frame: int = 0
        self.frame_override_active: bool = False  # use slider frame for mocap target?
        self.is_playing: bool = False
        self.loop_playback: bool = True
        self.event_overrides: dict[str, int] = {}  # user-set A/T/I/F sample numbers
        # Playback speed multiplier and marker-trail length for the animated
        # full-trajectory preview (issue #4482).
        self.playback_speed: float = 1.0
        self.trail_frames: int = 30
        self.show_trail: bool = True

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        # Phase window state keeps the user/session display label for legacy
        # compatibility; drawing paths normalize it back to a logical key.
        self.phase_window: str = _DEFAULT_PHASE
        self.manual_window_start: int = 0
        self.manual_window_end: int = 0

        # Playback target — what advances when the timer fires:
        #   "Mocap"     animate the mocap target only (skeleton stays static)
        #   "Skeleton"  animate the skeleton through its trajectory CSV
        #   "Both"      animate both, time-aligned at the impact frame
        self.playback_target: str = "Mocap"

        # Body-skeleton renderer style — "lines" | "library_shapes" (issue #4767).
        self.body_skeleton_style: BodySkeletonStyleLiteral = (
            _DEFAULT_BODY_SKELETON_STYLE
        )

        # Event labels (Address / Top of Backswing / Impact / Finish, or
        # author-specific conventions).  Mutated via the Event-Labels
        # group; persisted to session JSON.
        self.event_label_preset: str = _DEFAULT_EVENT_PRESET
        self.event_labels: dict[str, str] = dict(
            _EVENT_LABEL_PRESETS[_DEFAULT_EVENT_PRESET]
        )

        # Live multi-source state. The actual ``LiveViewController`` is
        # constructed at the end of :meth:`_build_ui` (it needs the
        # matplotlib axes), but ``_live_body_target`` is read by
        # :meth:`_build_align_box` while the section dict is being built,
        # so it has to exist with a sentinel value before that runs.
        self._live_body_target: Any | None = None

        self._build_ui()
        self._apply_camera_preset(_DEFAULT_CAMERA)

        default_xlsx = Path(__file__).with_name("Wiffle_ProV1_club_3D_data.xlsx")
        if default_xlsx.exists():
            self._load_xlsx(str(default_xlsx))

    # ===================================================================== #
    # UI                                                                    #
    # ===================================================================== #

    def _build_ui(self) -> None:
        """Build the main widget with QSplitters so the user can resize the
        control panel vs. the plot AND each section independently."""
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(0)

        # Outer horizontal splitter: control panel | plot
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self.h_splitter)

        # ---------- LEFT: scrollable column with vertical splitter ------- #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_widget = QWidget()
        scroll.setWidget(left_widget)
        left_col = QVBoxLayout(left_widget)
        left_col.setContentsMargins(4, 4, 4, 4)
        left_col.setSpacing(4)

        title = QLabel("Starting-Pose Matcher")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_col.addWidget(title)

        # Vertical splitter: every group can be resized.  Sections in order.
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.setChildrenCollapsible(True)
        # Multi-source toggle panel (issue #4480).  Lives alongside the
        # legacy Mocap-Source group; either may be used to drive the view.
        self.source_panel = DataSourcesPanel()
        self.source_panel.targets_changed.connect(self._on_multi_source_changed)
        self._latest_multi_source: object | None = None

        self._sections: dict[str, QGroupBox] = {
            "Mocap Source": self._build_file_box(),
            "Data sources": self.source_panel,
            "Event Labels": self._build_event_labels_box(),
            "Pose Slots": self._build_pose_box(),
            "Playback": self._build_playback_box(),
            "View / Mocap Traces": self._build_view_box(),
            "Auto-Align": self._build_align_box(),
            "Rigid Transform + Scale": self._build_transform_box(),
            "Output": self._build_save_box(),
        }
        for name, box in self._sections.items():
            self._attach_help_button(box, name)
            self.v_splitter.addWidget(box)
        # Reasonable starting heights (px) so big sections aren't squished.
        self.v_splitter.setSizes([90, 200, 160, 180, 220, 200, 180, 280, 140])
        left_col.addWidget(self.v_splitter, stretch=1)

        self.h_splitter.addWidget(scroll)

        # ---------- RIGHT: plot column ----------------------------------- #
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor="#1f242b")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#1f242b")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        self.toolbar.setStyleSheet("background:#2b2f36;color:#e6e6e6;")
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        self.h_splitter.addWidget(plot_widget)
        self.h_splitter.setSizes([440, 1200])
        self.h_splitter.setStretchFactor(0, 0)
        self.h_splitter.setStretchFactor(1, 1)

        self._setup_axes()

        # Live multi-source view controller (issue #4512). Owns the layer
        # stack that renders BodyTarget / ClubTarget / BallImpact data on
        # the same axes the static-pose path uses.
        self._live_view = LiveViewController(
            self.ax,
            self.canvas,
            body_skeleton_style=self.body_skeleton_style,
        )
        # ``_live_body_target`` was initialised in ``__init__`` so the
        # section builders could reference it; nothing more to set here.

    # ===================================================================== #
    # Event handlers — labels / phase / playback / transform                #
    # ===================================================================== #

    def _on_event_preset_changed(self, preset: str) -> None:
        if preset in _EVENT_LABEL_PRESETS:
            self.event_label_preset = preset
            self.event_labels = dict(_EVENT_LABEL_PRESETS[preset])
            for k, le in self._event_label_edits.items():
                with QSignalBlocker(le):
                    le.setText(self.event_labels[k])
        else:
            self.event_label_preset = "Custom"
        self._refresh_event_label_dependents()

    def _on_event_label_edited(self, key: str) -> None:
        text = self._event_label_edits[key].text().strip() or key
        self.event_labels[key] = text
        # Switch preset to Custom unless the new map matches a preset exactly.
        for preset, mapping in _EVENT_LABEL_PRESETS.items():
            if mapping == self.event_labels:
                self.event_label_preset = preset
                with QSignalBlocker(self.event_preset_combo):
                    self.event_preset_combo.setCurrentText(preset)
                break
        else:
            self.event_label_preset = "Custom"
            with QSignalBlocker(self.event_preset_combo):
                self.event_preset_combo.setCurrentText("Custom…")
        self._refresh_event_label_dependents()

    def _refresh_event_label_dependents(self) -> None:
        """Re-render anything that displays event labels."""
        # Pose-slot 'Event' combos: we keep the underlying key (A/T/I/F)
        # but show the display label.  Done via combo items.
        for combo in getattr(self, "_pose_event_combos", {}).values():
            current = combo.currentText().split()[0]  # original key
            with QSignalBlocker(combo):
                combo.clear()
                for k in _EVENT_KEYS:
                    combo.addItem(f"{k} - {self.event_labels[k]}")
                # restore selection
                idx = next(
                    (
                        i
                        for i in range(combo.count())
                        if combo.itemText(i).startswith(current + " ")
                    ),
                    0,
                )
                combo.setCurrentIndex(idx)
        # "Mark current frame as event" combo
        if hasattr(self, "combo_set_event"):
            current = self.combo_set_event.currentText().split(" ", 1)[0] or "T"
            with QSignalBlocker(self.combo_set_event):
                self.combo_set_event.clear()
                for k in _EVENT_KEYS:
                    self.combo_set_event.addItem(f"{k} - {self.event_labels[k]}")
                for i in range(self.combo_set_event.count()):
                    if self.combo_set_event.itemText(i).startswith(current + " "):
                        self.combo_set_event.setCurrentIndex(i)
                        break
        # Phase combo — re-render display labels (preserve selected key).
        if hasattr(self, "phase_combo"):
            current_key = self.phase_combo.currentData() or self.phase_window
            with QSignalBlocker(self.phase_combo):
                self.phase_combo.clear()
                for k in _PHASE_KEYS:
                    self.phase_combo.addItem(
                        _phase_display_label(k, self.event_labels), k
                    )
                for i in range(self.phase_combo.count()):
                    if self.phase_combo.itemData(i) == current_key:
                        self.phase_combo.setCurrentIndex(i)
                        break
        # Refresh events summary line
        self.lbl_event_info.setText(self._events_summary())
        self._redraw()

    # ---------- file / source loaders ------------------------------------- #

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Wiffle xlsx",
            str(Path(__file__).parent),
            "Excel files (*.xlsx *.xls)",
        )
        if path:
            self._load_xlsx(path)

    def _on_load_c3d_body_clicked(self) -> None:
        """Browse for a ``.c3d`` file and route it to the live view.

        Issue #4512: this is the user-facing entry point for the live
        body marker rendering — picking a file here causes 27 markers to
        appear on the existing 3D axes and start scrubbing with the
        timeline slider.
        """
        default_dir = str(Path(__file__).resolve().parents[3] / "data")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open C3D body file",
            default_dir,
            "C3D files (*.c3d)",
        )
        if not path:
            return
        try:
            from src.shared.python.motion_matching.load_body_target import (
                load_body_target,
            )

            body = load_body_target(path)
        except Exception as exc:  # noqa: BLE001 - surface loader errors
            logger.exception("failed to load C3D body file %s", path)
            QMessageBox.warning(self, "Load failed", f"Could not load C3D body:\n{exc}")
            return

        self._live_body_target = body
        if hasattr(self, "run_fit_button"):
            self.run_fit_button.set_inputs(
                target=body,
                engine_name=self.combo_fit_engine.currentText(),
            )
        n = int(body.marker_xyz.shape[0])
        m = int(body.marker_xyz.shape[1])
        self.lbl_c3d_body.setText(
            f"Live body: {Path(path).name}  ({m} markers, {n} samples)"
        )
        # Drive the existing slider/spin to the impact frame so the user
        # sees a recognisable pose immediately.
        self._live_view.set_target(body=body, club=None, ball=None)
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.spin_frame):
            self.frame_slider.setRange(0, max(0, n - 1))
            self.spin_frame.setRange(0, max(0, n - 1))
            self.frame_slider.setValue(0)
            self.spin_frame.setValue(0)
        self._live_view.set_frame(0)
        self.canvas.draw_idle()

    def _on_sheet_changed(self, _: str) -> None:
        if self._xlsx_path:
            self._load_xlsx(self._xlsx_path)

    def _on_pose_toggled(self, _state: int) -> None:
        for key, cb in self._pose_visible_checks.items():
            self.poses[key].visible = cb.isChecked()
        self._redraw()

    def _on_pose_event_changed(self, _: str) -> None:
        for key, combo in self._pose_event_combos.items():
            txt = combo.currentText().strip()
            # Combo items are now "K - Label"; extract the key.
            self.poses[key].target_event = txt.split(" ", 1)[0] if txt else "T"
        self._redraw()

    def _on_traces_toggled(self, _: int) -> None:
        self.show_clubhead_trace = self.cb_clubhead_trace.isChecked()
        self.show_midhands_trace = self.cb_midhands_trace.isChecked()
        self._redraw()

    def _on_playback_target_changed(self, target: str) -> None:
        if target not in ("Mocap", "Skeleton", "Both"):
            target = "Mocap"
        self.playback_target = target
        self._redraw()

    def _on_body_skeleton_style_changed(self, label: str) -> None:
        """Swap the body-skeleton renderer when the user picks a style.

        The swap is non-destructive: the loaded body target stays and
        the controller rebuilds the skeleton layer in place, preserving
        the current frame.
        """
        style = _BODY_SKELETON_STYLE_LABELS.get(label, _DEFAULT_BODY_SKELETON_STYLE)
        if style == self.body_skeleton_style:
            return
        self.body_skeleton_style = style
        live_view = getattr(self, "_live_view", None)
        if live_view is None:
            return
        live_view.set_body_skeleton_style(style)
        self.canvas.draw_idle()

    def _load_trajectory(self, slot_key: str) -> None:
        """Load a Simscape CSV trajectory for the given pose slot."""
        slot = self.poses.get(slot_key)
        if slot is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load Simscape trajectory CSV for {slot_key}",
            str(Path(__file__).parent),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            traj = load_simscape_trajectory_csv(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Trajectory load failed",
                f"Could not load {Path(path).name}:\n\n{exc}",
            )
            return
        if len(traj) == 0:
            QMessageBox.warning(
                self,
                "Empty trajectory",
                f"{Path(path).name} loaded but has no usable frames.",
            )
            return
        slot.trajectory = traj
        slot.trajectory_frame_index = 0
        # Update the button label so the user can see a trajectory is loaded.
        btn = self._pose_trajectory_buttons.get(slot_key)
        if btn is not None:
            btn.setText(f"✓ {len(traj)}f")
            btn.setToolTip(
                f"Loaded {len(traj)} frames from {Path(path).name}.\n"
                f"Time range: {traj.times[0]:.3f}s … {traj.times[-1]:.3f}s.\n"
                "Click to load a different file."
            )
        # First trajectory load auto-switches to 'Both' mode so the user
        # can immediately see the skeleton animate without having to find
        # the Playback target combo.
        if self.playback_target == "Mocap":
            with QSignalBlocker(self.combo_playback_target):
                self.combo_playback_target.setCurrentText("Both")
            self.playback_target = "Both"
        self._notify(
            f"Loaded {len(traj)}-frame trajectory for {slot_key} "
            f"from {Path(path).name}.  Playback target → Both."
        )
        self._redraw()

    def _toggle_play(self) -> None:
        """Override the parent toggle to surface a helpful message when the
        user presses Play in a target mode that won't visibly do anything.
        """
        if self.is_playing:
            self._timer.stop()
            self.is_playing = False
            self.btn_play.setText("▶ Play")
            return
        # About to start — sanity-check the chosen target.
        if self.playback_target == "Skeleton":
            visible_with_traj = [
                s for s in self.poses.values() if s.visible and s.trajectory is not None
            ]
            if not visible_with_traj:
                QMessageBox.information(
                    self,
                    "No skeleton trajectory loaded",
                    "Playback target is 'Skeleton' but no visible pose has\n"
                    "a trajectory CSV loaded yet.\n\n"
                    "Either:\n"
                    "  • Pose Slots → Trajectory Load… for one of the visible poses, or\n"
                    "  • Switch the Playback target back to 'Mocap'.",
                )
                return
        if self.df is None and self.playback_target in ("Mocap", "Both"):
            QMessageBox.information(
                self,
                "No mocap loaded",
                "Playback target is 'Mocap' or 'Both' but no xlsx file has\n"
                "been loaded yet.  Use Mocap Source → Load xlsx… first.",
            )
            return
        fps = max(1, int(self.spin_speed.value()))
        self._timer.start(int(round(1000.0 / fps)))
        self.is_playing = True
        self.btn_play.setText("⏸ Pause")

    def _on_phase_changed(self, phase: int | str) -> None:
        if isinstance(phase, str):
            label = phase
            key = _phase_key_from_label(label) or _DEFAULT_PHASE
        else:
            key = self.phase_combo.currentData()
            label = self.phase_combo.currentText()
            if not key:
                key = _phase_key_from_label(label) or _DEFAULT_PHASE
        self.phase_window = label
        self.manual_range_widget.setVisible(key == "manual")
        if key == "manual" and isinstance(phase, str) and not self.isVisible():
            self.show()
        self._redraw()

    def _on_manual_range_changed(self, _: int) -> None:
        self.manual_window_start = int(self.spin_phase_start.value())
        self.manual_window_end = int(self.spin_phase_end.value())
        if self.manual_window_end < self.manual_window_start:
            self.manual_window_end = self.manual_window_start
            with QSignalBlocker(self.spin_phase_end):
                self.spin_phase_end.setValue(self.manual_window_end)
        self._redraw()

    # ---- playback handlers ----
    def _on_frame_changed_slider(self, frame: int) -> None:
        with QSignalBlocker(self.spin_frame):
            self.spin_frame.setValue(int(frame))
        self.current_frame = int(frame)
        self._update_time_label()
        self._update_frame_counter()
        self._redraw()
        if getattr(self, "_live_view", None) is not None:
            self._live_view.set_frame(int(frame))

    def _on_frame_changed_spin(self, frame: int) -> None:
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setValue(int(frame))
        self.current_frame = int(frame)
        self._update_time_label()
        self._update_frame_counter()
        self._redraw()
        if getattr(self, "_live_view", None) is not None:
            self._live_view.set_frame(int(frame))

    def _update_frame_counter(self) -> None:
        """Refresh the ``12 / 301`` frame-counter label."""
        n = len(self.df) if self.df is not None else 0
        if hasattr(self, "lbl_frame_counter"):
            self.lbl_frame_counter.setText(
                f"{int(self.current_frame)} / {max(0, n - 1)}"
            )

    def _step_frame(self, delta: int) -> None:
        if self.df is None:
            return
        n = len(self.df)
        if delta <= -(10**8):
            self.spin_frame.setValue(0)
        elif delta >= 10**8:
            self.spin_frame.setValue(n - 1)
        else:
            new = max(0, min(n - 1, self.current_frame + delta))
            self.spin_frame.setValue(new)

    def _advance_frame(self) -> None:
        """Advance one playback step.

        Behaviour depends on `self.playback_target`:
            Mocap     - advance current_frame only
            Skeleton  - advance each visible pose's trajectory_frame_index
            Both      - advance both, time-aligned at impact
        """
        target = self.playback_target

        # 1. Mocap frame ----------------------------------------------------
        n = len(self.df) if self.df is not None else 0
        if target in ("Mocap", "Both") and n > 0:
            nxt = self.current_frame + 1
            if nxt >= n:
                if self.loop_playback:
                    nxt = 0
                else:
                    self._toggle_play()
                    return
            self.spin_frame.setValue(nxt)
        elif target == "Skeleton":
            # Without mocap advance, still consider stop condition based on
            # the longest visible trajectory.
            longest = max(
                (
                    len(s.trajectory)
                    for s in self.poses.values()
                    if s.visible and s.trajectory is not None
                ),
                default=0,
            )
            if longest == 0:
                self._toggle_play()
                return

        # 2. Skeleton trajectory frame -------------------------------------
        if target in ("Skeleton", "Both"):
            # In "Both" mode, time-align by mapping mocap_time -> sim_time
            # via the impact-frame offset.
            if target == "Both" and self.df is not None:
                self._sync_trajectory_indices_from_mocap()
            else:
                # Pure Skeleton: advance each visible trajectory by one frame.
                for slot in self.poses.values():
                    if not slot.visible or slot.trajectory is None:
                        continue
                    nxt = slot.trajectory_frame_index + 1
                    if nxt >= len(slot.trajectory):
                        nxt = 0 if self.loop_playback else (len(slot.trajectory) - 1)
                    slot.trajectory_frame_index = nxt
                self._redraw()  # redraw needed when only the skeleton moved

    def _sync_trajectory_indices_from_mocap(self) -> None:
        """In 'Both' mode, set each visible trajectory's frame index from the
        current mocap frame's time, aligned so the trajectory's first frame
        corresponds to the mocap address (A) frame and shafts hit at impact.

        Falls back to a linear stretch when impact times can't be resolved.
        """
        if self.df is None:
            return
        mocap_t = float(self.df.iloc[self.current_frame]["time"])
        a_idx = self._frame_for("A")
        i_idx = self._frame_for("I")
        if a_idx is None or i_idx is None or i_idx <= a_idx:
            # No valid window — pure linear stretch over [0, n_mocap-1].
            n_mocap = len(self.df)
            for slot in self.poses.values():
                if not slot.visible or slot.trajectory is None:
                    continue
                frac = self.current_frame / max(1, n_mocap - 1)
                slot.trajectory_frame_index = int(
                    np.clip(
                        frac * (len(slot.trajectory) - 1), 0, len(slot.trajectory) - 1
                    )
                )
            return
        mocap_t_a = float(self.df.iloc[a_idx]["time"])
        mocap_t_i = float(self.df.iloc[i_idx]["time"])
        # Map mocap_t into [0, 1] across A..I, then onto trajectory's time axis.
        for slot in self.poses.values():
            if not slot.visible or slot.trajectory is None:
                continue
            traj = slot.trajectory
            if len(traj.times) < 2:
                continue
            sim_t_a = float(traj.times[0])
            # Best impact estimate in trajectory: largest clubhead speed.
            sim_t_i = self._estimate_trajectory_impact_time(traj)
            if sim_t_i <= sim_t_a:
                # Fallback: align endpoints linearly.
                frac = (mocap_t - mocap_t_a) / max(1e-9, mocap_t_i - mocap_t_a)
                sim_t = sim_t_a + frac * (float(traj.times[-1]) - sim_t_a)
            else:
                # Linear map mocap_t -> sim_t through (A, I) anchor pair.
                slope = (sim_t_i - sim_t_a) / (mocap_t_i - mocap_t_a)
                sim_t = sim_t_a + slope * (mocap_t - mocap_t_a)
            slot.trajectory_frame_index = traj.frame_at_time(sim_t)

    def _estimate_trajectory_impact_time(self, traj: SkeletonTrajectory) -> float:
        """Return the time of peak |dCH/dt|^2 in the trajectory, or t[0] if
        clubhead positions aren't available.
        """
        if not traj.frames or "ch" not in traj.frames[0].joints:
            return float(traj.times[0]) if len(traj.times) else 0.0
        ch = np.array([f.joints["ch"] for f in traj.frames if "ch" in f.joints])
        if len(ch) < 3:
            return float(traj.times[0])
        # Forward-difference speed
        dt = np.diff(traj.times[: len(ch)])
        dt = np.where(dt == 0, 1e-6, dt)
        v = np.diff(ch, axis=0) / dt[:, None]
        speed = np.linalg.norm(v, axis=1)
        i = int(np.argmax(speed))
        return float(traj.times[i])

    def _on_frame_override_toggled(self, _state: int) -> None:
        self.frame_override_active = self.cb_use_current_frame.isChecked()
        self._redraw()

    def _set_event_to_current_frame(self) -> None:
        if self.df is None:
            return
        # Combo text is "K - Label"; key is first token.
        ev = self.combo_set_event.currentText().split(" ", 1)[0] or "T"
        # Store as "absolute sample number" (1-based) so it round-trips with
        # MocapEvents; current_frame is 0-based in the loaded data.
        self.event_overrides[ev] = self.current_frame + 1
        # Reflect in events struct for in-session use
        setattr(self.events, f"{ev}_sample", float(self.current_frame + 1))
        self.lbl_event_info.setText(self._events_summary() + "  (overrides active)")
        self._redraw()

    def _clear_event_overrides(self) -> None:
        if not self.event_overrides:
            return
        # Re-read events from the xlsx to undo overrides
        if self._xlsx_path:
            self.events = read_event_header(
                self._xlsx_path, self.sheet_combo.currentText()
            )
        self.event_overrides = {}
        self.lbl_event_info.setText(self._events_summary())
        self._redraw()

    def _update_time_label(self) -> None:
        if self.df is None or self.current_frame >= len(self.df):
            self.lbl_time.setText("t = — s")
            return
        t = float(self.df.iloc[self.current_frame]["time"])
        self.lbl_time.setText(f"t = {t:+.3f} s   (frame {self.current_frame})")

    def _on_lock_xy_toggled(self, _state: int) -> None:
        self.lock_xy_rotation = not self.cb_lock_xy.isChecked()
        if self.lock_xy_rotation:
            self.s_rx.set_value(0.0)
            self.s_ry.set_value(0.0)
        self.s_rx.setEnabled(not self.lock_xy_rotation)
        self.s_ry.setEnabled(not self.lock_xy_rotation)

    def _on_transform_changed(self, _: float) -> None:
        self.transform.tx = self.s_tx.value()
        self.transform.ty = self.s_ty.value()
        self.transform.tz = self.s_tz.value()
        if self.lock_xy_rotation:
            self.transform.rx = 0.0
            self.transform.ry = 0.0
        else:
            self.transform.rx = self.s_rx.value()
            self.transform.ry = self.s_ry.value()
        self.transform.rz = self.s_rz.value()
        self.transform.scale = max(1e-3, self.s_scale.value())
        self._redraw()

    # ---------- resets ---------------------------------------------------- #

    def _reset_translations(self) -> None:
        for s in (self.s_tx, self.s_ty, self.s_tz):
            s.set_value(0.0)

    def _reset_rotations(self) -> None:
        for s in (self.s_rx, self.s_ry, self.s_rz):
            s.set_value(0.0)

    def _reset_all(self) -> None:
        self._reset_translations()
        self._reset_rotations()
        self.s_scale.set_value(1.0)

    def _reload_pose(self, key: str) -> None:
        path = Path(__file__).parent / f"simscape_skeleton_{key}.json"
        self.poses[key].skeleton = load_skeleton(path, key)
        self._redraw()

    # ---------- snaps ----------------------------------------------------- #

    def _snap_mid_first_visible(self) -> None:
        slot = self._first_visible_pose()
        if slot is None or "mp" not in slot.skeleton.joints:
            return
        target = self._mocap_pos_for(slot, "mid")
        if target is None:
            return
        # Apply current rotation+scale (no translation) and compute delta.
        no_t = RigidTransform(
            rx=0.0 if self.lock_xy_rotation else self.s_rx.value(),
            ry=0.0 if self.lock_xy_rotation else self.s_ry.value(),
            rz=self.s_rz.value(),
            scale=max(1e-3, self.s_scale.value()),
            pivot=self.transform.pivot,
        )
        rotated_mp = no_t.apply(slot.skeleton.joints["mp"][None, :])[0]
        delta = target - rotated_mp
        self.s_tx.set_value(float(delta[0]))
        self.s_ty.set_value(float(delta[1]))
        self.s_tz.set_value(float(delta[2]))

    def _snap_shaft(self, slot_key: str) -> None:
        """Two-point shaft alignment for one pose.

        Keeps Z up (Rx=Ry=0).  Solves Rz so the model shaft (mp→ch) in the
        XY plane points the same way as the mocap shaft, then sets Tx/Ty/Tz
        so the model mid-hands lands on the mocap mid-hands.  Optionally
        sets scale = |shaft_target| / |shaft_model|.
        """
        slot = self.poses.get(slot_key)
        if slot is None:
            return
        sk = slot.skeleton
        if "mp" not in sk.joints or "ch" not in sk.joints:
            self._notify("Pose lacks mp/ch joints — cannot shaft-snap.")
            return
        mp_target = self._mocap_pos_for(slot, "mid")
        ch_target = self._mocap_pos_for(slot, "club")
        if mp_target is None or ch_target is None:
            self._notify(f"No mocap row for event '{slot.target_event}'.")
            return

        mp_skel = sk.joints["mp"]
        ch_skel = sk.joints["ch"]

        # Optional scale: ratio of shaft lengths.
        if self.cb_fit_scale.isChecked():
            shaft_t = ch_target - mp_target
            shaft_m = ch_skel - mp_skel
            len_t = float(np.linalg.norm(shaft_t))
            len_m = float(np.linalg.norm(shaft_m))
            if len_m > 1e-6 and len_t > 1e-6:
                new_scale = float(
                    np.clip(
                        len_t / len_m, _S_RANGE[0] * _S_SCALE, _S_RANGE[1] * _S_SCALE
                    )
                )
                self.s_scale.set_value(new_scale)

        scale = max(1e-3, self.s_scale.value())

        # Solve Rz from XY-plane shaft directions (delegated to core).
        nt = float(np.linalg.norm((ch_target - mp_target)[:2]))
        nm = float(np.linalg.norm((ch_skel - mp_skel)[:2]))
        if nt < 1e-6 or nm < 1e-6:
            self._notify(
                "Shaft projection onto XY plane is degenerate (vertical "
                "shaft) — Rz cannot be solved.  Adjust manually."
            )
            return
        rz_deg = solve_shaft_rz_deg(mp_target, ch_target, mp_skel, ch_skel)

        # Lock Rx/Ry to 0 for this snap (Z-up).
        if not self.lock_xy_rotation:
            self.cb_lock_xy.setChecked(False)  # leave as-is for user; we just zero
        self.s_rx.set_value(0.0)
        self.s_ry.set_value(0.0)
        self.s_rz.set_value(rz_deg)

        # Translation: rotate+scale mp_skel about pivot, then offset to land on mp_target.
        rotated = RigidTransform(
            rx=0.0, ry=0.0, rz=rz_deg, scale=scale, pivot=self.transform.pivot
        )
        rotated_mp = rotated.apply(mp_skel[None, :])[0]
        delta = mp_target - rotated_mp
        self.s_tx.set_value(float(delta[0]))
        self.s_ty.set_value(float(delta[1]))
        self.s_tz.set_value(float(delta[2]))
        self._notify(
            f"Snapped {slot_key}: Rz={rz_deg:+.1f}°, "
            f"|shaft_target|={nt:.3f}m, |shaft_model|={nm:.3f}m"
        )

    # ---------- file load ------------------------------------------------- #

    def _load_xlsx(self, path: str) -> None:
        sheet = self.sheet_combo.currentText()
        try:
            df = load_mocap_xlsx(path, sheet)
        except Exception as exc:  # noqa: BLE001
            logger.error("Load failed: %s", exc)
            self.lbl_file.setText(f"Load failed: {exc}")
            return
        if df is None or df.empty:
            self.lbl_file.setText(f"No data in sheet '{sheet}'")
            return
        self.df = df
        self._xlsx_path = path
        self.events = read_event_header(path, sheet)
        # Re-apply any event-override that survived from the previous load
        for ev, sample in list(self.event_overrides.items()):
            setattr(self.events, f"{ev}_sample", float(sample))
        n = len(df)
        self.lbl_file.setText(f"{Path(path).name}\nsheet={sheet}  frames={n}")
        self.lbl_event_info.setText(self._events_summary())
        # Configure playback widgets to the new range
        with QSignalBlocker(self.spin_frame):
            self.spin_frame.setRange(0, n - 1)
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setRange(0, n - 1)
        with QSignalBlocker(self.spin_phase_start):
            self.spin_phase_start.setRange(0, n - 1)
        with QSignalBlocker(self.spin_phase_end):
            self.spin_phase_end.setRange(0, n - 1)
            self.spin_phase_end.setValue(n - 1)
        self.manual_window_end = n - 1
        # Default initial frame to T (top of backswing) if available, but keep
        # enough room for ordinary step-forward playback on short canonical
        # windows.
        t_frame = self._frame_for("T")
        if t_frame is not None:
            t_frame = min(t_frame, max(0, n - 6))
            with QSignalBlocker(self.spin_frame):
                self.spin_frame.setValue(t_frame)
            with QSignalBlocker(self.frame_slider):
                self.frame_slider.setValue(t_frame)
            self.current_frame = t_frame
        self._update_time_label()
        self._redraw()

    def _events_summary(self) -> str:
        e = self.events
        parts = []
        for k in _EVENT_KEYS:
            v = getattr(e, f"{k}_sample")
            label = self.event_labels.get(k, k)
            sval = "?" if v != v else int(v)
            parts.append(f"{label} ({k})={sval}")
        if e.CHS_mph == e.CHS_mph:
            parts.append(f"CHS={e.CHS_mph:.1f}mph")
        return "Events:  " + "  ".join(parts)

    def _on_calibrate_subject_clicked(self) -> None:
        """Open the subject-anthropometrics calibration dialog (issue #4820)."""
        from .widgets.calibration_dialog import CalibrationDialog

        dlg = CalibrationDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            res = dlg.result_record
            if res is not None:
                self.lbl_subject.setText(
                    f"Subject: {res.record.subject_id} → {res.saved_path}"
                )

    # --------------------------------------------------------------------- #
    # Multi-source target panel hook (issue #4480)                          #
    # --------------------------------------------------------------------- #

    def _on_multi_source_changed(self, target: object) -> None:
        """Cache the latest ``MultiSourceTarget`` from the data-sources panel.

        Downstream consumers (cost/animation, landed in later issues) can
        read ``self._latest_multi_source`` to dispatch on whichever subset
        of targets the user toggled on.
        """
        self._latest_multi_source = target
        if target is None:
            logger.info("Data-sources panel cleared.")
        else:
            logger.info(
                "Data-sources panel: club=%s body=%s",
                getattr(target, "has_club", lambda: False)(),
                getattr(target, "has_body", lambda: False)(),
            )

    def _serialize_data_sources(self) -> dict[str, Any]:
        """Snapshot the data-sources panel for the session JSON."""
        return serialize_data_sources(self.source_panel.snapshot())

    def _apply_data_sources(self, block: dict[str, Any] | None) -> None:
        """Restore the data-sources panel from a (possibly missing) block."""
        parsed: DataSourcesBlock = parse_data_sources(block)
        self.source_panel.restore(parsed)

    def _apply_body_skeleton_block(self, block: dict[str, Any] | None) -> None:
        """Restore the body-skeleton renderer style from session JSON.

        Missing or unrecognised blocks fall back to the default style.
        Updates the combo widget without firing the change signal so a
        load does not double-trigger a rebuild.
        """
        parsed: BodySkeletonBlock = parse_body_skeleton(block)
        if parsed.style not in _BODY_SKELETON_STYLES:
            return
        self.body_skeleton_style = parsed.style
        if hasattr(self, "combo_body_skeleton_style"):
            label = _BODY_SKELETON_STYLE_LABEL_BY_KEY.get(parsed.style)
            if label is not None:
                with QSignalBlocker(self.combo_body_skeleton_style):
                    self.combo_body_skeleton_style.setCurrentText(label)
        live_view = getattr(self, "_live_view", None)
        if live_view is not None:
            live_view.set_body_skeleton_style(parsed.style)

    # --------------------------------------------------------------------- #
    # Embedded-tool lifecycle hooks                                         #
    # --------------------------------------------------------------------- #

    def cleanup(self) -> None:
        """Release timers and matplotlib resources before the host disposes us.

        Called by :class:`_MotionMatchPreviewEmbedAdapter.cleanup`. Safe
        to call more than once.
        """
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:  # pragma: no cover - defensive
                logger.debug("MainWidget timer.stop() raised", exc_info=True)
        # Drop the live-view controller's references so its artists can
        # be garbage-collected with the figure.
        live_view = getattr(self, "_live_view", None)
        if live_view is not None:
            try:
                live_view.set_target(body=None, club=None, ball=None)
            except Exception:  # pragma: no cover - defensive
                logger.debug("MainWidget live_view teardown raised", exc_info=True)
