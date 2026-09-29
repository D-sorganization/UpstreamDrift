"""UI-builder mixin for :class:`MainWidget`.

Holds every ``_build_<section>_box`` method plus the shared
``_attach_help_button`` helper. Split out of ``gui.py`` (Subtask 5 /
#4998 of EPIC #4993) so the per-file budget stays under 1200 lines.

The mixin assumes the host class has been initialised with the data
attributes used by the builders (``self.poses``, ``self.events``,
``self.event_labels``, ``self.transform``, etc.) and exposes the
handler slots referenced by ``connect`` calls.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.motion_matching import provider_registry
from src.tools.starting_pose_matcher.core import (
    DEFAULT_PHASE as _DEFAULT_PHASE,
    EVENT_KEYS as _EVENT_KEYS,
    EVENT_LABEL_PRESETS as _EVENT_LABEL_PRESETS,
    PHASE_KEYS as _PHASE_KEYS,
    phase_display_label as _phase_display_label,
)
from src.tools.starting_pose_matcher.session_schema import (
    DEFAULT_BODY_SKELETON_STYLE as _DEFAULT_BODY_SKELETON_STYLE,
)

from ._gui_common import (
    LabelledControl,
    _BODY_SKELETON_STYLE_LABEL_BY_KEY,
    _BODY_SKELETON_STYLE_LABELS,
    _CAMERA_PRESETS,
    _R_RANGE,
    _R_SCALE,
    _S_RANGE,
    _S_SCALE,
    _T_RANGE,
    _T_SCALE,
    _help_button,
    _hsep,
)


class _BuildersMixin:
    """All section ``_build_*`` methods used by :meth:`MainWidget._build_ui`."""

    def _attach_help_button(self, box: QGroupBox, section: str) -> None:
        """Place a small '?' help button at the top-right corner of a QGroupBox.

        Repositions itself on resize so it always tracks the corner.
        """
        btn = _help_button(section, box)
        btn.setParent(box)
        btn.show()
        btn.raise_()

        def _reposition() -> None:
            btn.move(max(0, box.width() - 30), 4)

        _reposition()
        # Chain into the existing resizeEvent without losing it
        original = box.resizeEvent

        def _on_resize(event):  # noqa: ANN001
            _reposition()
            original(event)

        box.resizeEvent = _on_resize  # type: ignore[method-assign]

    # ---------- builders --------------------------------------------------- #

    def _build_file_box(self) -> QGroupBox:
        box = QGroupBox("Mocap Source")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(6)
        self.btn_load = QPushButton("Load xlsx…")
        self.btn_load.setToolTip("Load a motion-capture xlsx target file")
        self.btn_load.setStatusTip("Loads motion-capture xlsx")
        self.btn_load.clicked.connect(self._on_load_clicked)
        gl.addWidget(self.btn_load, 0, 0, 1, 2)
        gl.addWidget(QLabel("Sheet:"), 1, 0)
        self.sheet_combo = QComboBox()
        self.sheet_combo.addItems(["TW_ProV1", "TW_wiffle", "GW_wiffle", "GW_ProV11"])
        self.sheet_combo.setToolTip("Select which sheet of the xlsx to load")
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
        gl.addWidget(self.sheet_combo, 1, 1)
        self.lbl_file = QLabel("(no file loaded)")
        self.lbl_file.setObjectName("status")
        self.lbl_file.setWordWrap(True)
        gl.addWidget(self.lbl_file, 2, 0, 1, 2)
        self.lbl_event_info = QLabel("Events: (none)")
        self.lbl_event_info.setObjectName("status")
        self.lbl_event_info.setWordWrap(True)
        gl.addWidget(self.lbl_event_info, 3, 0, 1, 2)
        # Live C3D body source row (issue #4512). Loads a BodyTarget and
        # routes per-frame marker positions to the matcher's existing 3D
        # axes via the LiveViewController.
        self.btn_load_c3d_body = QPushButton("Browse C3D Body…")
        self.btn_load_c3d_body.setToolTip(
            "Load a .c3d body-marker file and render its markers live on the 3D axes."
        )
        self.btn_load_c3d_body.setStatusTip(
            "Loads a C3D body target and wires it to the timeline slider."
        )
        self.btn_load_c3d_body.clicked.connect(self._on_load_c3d_body_clicked)
        gl.addWidget(self.btn_load_c3d_body, 4, 0, 1, 2)
        self.lbl_c3d_body = QLabel("Live body: (none)")
        self.lbl_c3d_body.setObjectName("status")
        self.lbl_c3d_body.setWordWrap(True)
        gl.addWidget(self.lbl_c3d_body, 5, 0, 1, 2)
        # Layer toggles for the live view.
        self.cb_show_body_markers = QCheckBox("Show body markers")
        self.cb_show_body_markers.setChecked(True)
        self.cb_show_body_markers.toggled.connect(
            lambda on: (
                self._live_view.set_layer_visible("body_markers", bool(on))
                if getattr(self, "_live_view", None) is not None
                else None
            )
        )
        gl.addWidget(self.cb_show_body_markers, 6, 0, 1, 1)
        self.cb_show_body_skeleton = QCheckBox("Show body skeleton")
        self.cb_show_body_skeleton.setChecked(True)
        self.cb_show_body_skeleton.toggled.connect(
            lambda on: (
                self._live_view.set_layer_visible("body_skeleton", bool(on))
                if getattr(self, "_live_view", None) is not None
                else None
            )
        )
        gl.addWidget(self.cb_show_body_skeleton, 6, 1, 1, 1)
        # Body skeleton style combo — switches between line segments
        # (legacy / fast) and body_part_viz library shapes (richer
        # figure). Issue #4767. Default tracks ``body_skeleton_style``
        # so a session-restored choice survives this build.
        gl.addWidget(QLabel("Body skeleton style:"), 7, 0)
        self.combo_body_skeleton_style = QComboBox()
        for label in _BODY_SKELETON_STYLE_LABELS:
            self.combo_body_skeleton_style.addItem(label)
        current_label = _BODY_SKELETON_STYLE_LABEL_BY_KEY.get(
            self.body_skeleton_style,
            _BODY_SKELETON_STYLE_LABEL_BY_KEY[_DEFAULT_BODY_SKELETON_STYLE],
        )
        self.combo_body_skeleton_style.setCurrentText(current_label)
        self.combo_body_skeleton_style.setToolTip(
            "Choose how the body skeleton is rendered. "
            "Lines (default) draws plain segments between marker pairs; "
            "Library shapes uses body_part_viz meshes (head, torso, "
            "upper_arm, ...) bound to canonical Plug-in-Gait markers."
        )
        self.combo_body_skeleton_style.currentTextChanged.connect(
            self._on_body_skeleton_style_changed
        )
        gl.addWidget(self.combo_body_skeleton_style, 7, 1, 1, 1)
        return box

    def _build_event_labels_box(self) -> QGroupBox:
        box = QGroupBox("Event Labels")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(4)

        gl.addWidget(QLabel("Convention:"), 0, 0)
        self.event_preset_combo = QComboBox()
        for preset in _EVENT_LABEL_PRESETS:
            self.event_preset_combo.addItem(preset)
        self.event_preset_combo.addItem("Custom…")
        self.event_preset_combo.setCurrentText(self.event_label_preset)
        self.event_preset_combo.setToolTip(
            "Event-label naming convention used in the legend and pose-event combos"
        )
        self.event_preset_combo.currentTextChanged.connect(
            self._on_event_preset_changed
        )
        gl.addWidget(self.event_preset_combo, 0, 1, 1, 3)

        # Editable entries for each event key
        from PyQt6.QtWidgets import QLineEdit

        self._event_label_edits: dict[str, QLineEdit] = {}
        for r, k in enumerate(_EVENT_KEYS, start=1):
            gl.addWidget(QLabel(f"{k}:"), r, 0)
            le = QLineEdit(self.event_labels[k])
            le.setMinimumWidth(160)
            le.editingFinished.connect(lambda key=k: self._on_event_label_edited(key))
            self._event_label_edits[k] = le
            gl.addWidget(le, r, 1, 1, 3)

        hint = QLabel(
            "Custom labels are saved with the session and shown in the legend / "
            "current-frame indicator."
        )
        hint.setObjectName("status")
        hint.setWordWrap(True)
        gl.addWidget(hint, len(_EVENT_KEYS) + 1, 0, 1, 4)
        return box

    def _build_pose_box(self) -> QGroupBox:
        box = QGroupBox("Pose Slots")
        gl = QGridLayout(box)
        gl.setVerticalSpacing(4)
        gl.addWidget(QLabel("Show"), 0, 0)
        gl.addWidget(QLabel("Pose"), 0, 1)
        gl.addWidget(QLabel("Event"), 0, 2)
        gl.addWidget(QLabel("Reload"), 0, 3)
        gl.addWidget(QLabel("Trajectory"), 0, 4)
        self._pose_visible_checks: dict[str, QCheckBox] = {}
        self._pose_event_combos: dict[str, QComboBox] = {}
        self._pose_trajectory_buttons: dict[str, QPushButton] = {}
        for r, (key, slot) in enumerate(self.poses.items(), start=1):
            cb = QCheckBox()
            cb.setChecked(slot.visible)
            cb.setToolTip(
                f"Show or hide the {key} skeleton overlay (color {slot.color})"
            )
            cb.stateChanged.connect(self._on_pose_toggled)
            self._pose_visible_checks[key] = cb
            gl.addWidget(cb, r, 0)
            color = slot.color
            tag = QLabel(f'<span style="color:{color};">●</span>  {key}')
            gl.addWidget(tag, r, 1)
            ec = QComboBox()
            ec.setToolTip(f"Mocap event the {key} pose snaps to when Auto-Align is run")
            for k in _EVENT_KEYS:
                ec.addItem(f"{k} - {self.event_labels[k]}")
            # Pick the item whose first token matches the slot's key
            for i in range(ec.count()):
                if ec.itemText(i).startswith(slot.target_event + " "):
                    ec.setCurrentIndex(i)
                    break
            ec.currentTextChanged.connect(self._on_pose_event_changed)
            self._pose_event_combos[key] = ec
            gl.addWidget(ec, r, 2)
            rbtn = QPushButton("⟳")
            rbtn.setObjectName("preset")
            rbtn.setMaximumWidth(40)
            rbtn.setToolTip(f"Reload simscape_skeleton_{key}.json")
            rbtn.clicked.connect(lambda _checked, k=key: self._reload_pose(k))
            gl.addWidget(rbtn, r, 3)
            tbtn = QPushButton("Load…")
            tbtn.setObjectName("preset")
            tbtn.setMaximumWidth(80)
            tbtn.setToolTip(
                "Load a Simscape forward-dynamics CSV so the\n"
                "skeleton can play back through its motion."
            )
            tbtn.clicked.connect(lambda _checked, k=key: self._load_trajectory(k))
            self._pose_trajectory_buttons[key] = tbtn
            gl.addWidget(tbtn, r, 4)
        return box

    def _build_view_box(self) -> QGroupBox:
        box = QGroupBox("View / Mocap Traces")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        # Camera presets
        cam_row = QHBoxLayout()
        cam_row.setSpacing(4)
        for name in _CAMERA_PRESETS:
            b = QPushButton(name)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, n=name: self._apply_camera_preset(n))
            cam_row.addWidget(b)
        v.addLayout(cam_row)

        v.addWidget(_hsep())

        # Trace toggles
        self.cb_clubhead_trace = QCheckBox("Show mocap clubhead path")
        self.cb_clubhead_trace.setToolTip(
            "Overlay the clubhead path from the mocap file"
        )
        self.cb_clubhead_trace.stateChanged.connect(self._on_traces_toggled)
        v.addWidget(self.cb_clubhead_trace)
        self.cb_midhands_trace = QCheckBox("Show mocap mid-hands path")
        self.cb_midhands_trace.setToolTip(
            "Overlay the mid-hands path from the mocap file"
        )
        self.cb_midhands_trace.stateChanged.connect(self._on_traces_toggled)
        v.addWidget(self.cb_midhands_trace)

        # Phase window combo (replaces the old simple "swing window" checkbox)
        ph_row = QHBoxLayout()
        ph_row.addWidget(QLabel("Phase:"))
        self.phase_combo = QComboBox()
        self.phase_combo.setToolTip(
            "Phase window for trace overlays; choose Manual range to set frames"
        )
        for key in _PHASE_KEYS:
            self.phase_combo.addItem(_phase_display_label(key, self.event_labels), key)
        # Select the default by KEY (currentData() lookup)
        for i in range(self.phase_combo.count()):
            if self.phase_combo.itemData(i) == _DEFAULT_PHASE:
                self.phase_combo.setCurrentIndex(i)
                break
        self.phase_combo.currentIndexChanged.connect(self._on_phase_changed)
        ph_row.addWidget(self.phase_combo, stretch=1)
        v.addLayout(ph_row)

        # Manual range (hidden until "Manual range" selected)
        self.manual_range_widget = QWidget()
        mr = QHBoxLayout(self.manual_range_widget)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.addWidget(QLabel("From:"))
        self.spin_phase_start = QSpinBox()
        self.spin_phase_start.setRange(0, 0)
        self.spin_phase_start.setToolTip("First frame index of the manual phase window")
        self.spin_phase_start.valueChanged.connect(self._on_manual_range_changed)
        mr.addWidget(self.spin_phase_start)
        mr.addWidget(QLabel("To:"))
        self.spin_phase_end = QSpinBox()
        self.spin_phase_end.setRange(0, 0)
        self.spin_phase_end.setToolTip("Last frame index of the manual phase window")
        self.spin_phase_end.valueChanged.connect(self._on_manual_range_changed)
        mr.addWidget(self.spin_phase_end)
        self.manual_range_widget.setVisible(False)
        v.addWidget(self.manual_range_widget)

        # Show current-frame marker on traces
        self.cb_frame_marker = QCheckBox("Show current-frame marker on traces")
        self.cb_frame_marker.setChecked(True)
        self.cb_frame_marker.setToolTip(
            "Render a marker at the current playback frame on each trace"
        )
        self.cb_frame_marker.stateChanged.connect(lambda _: self._redraw())
        v.addWidget(self.cb_frame_marker)

        v.addWidget(_hsep())

        # Scene element toggles
        self.cb_show_ball = QCheckBox("Show golf ball")
        self.cb_show_ball.setChecked(self.show_ball)
        self.cb_show_ball.setToolTip("Render the ball glyph at the address position")
        self.cb_show_ball.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_show_ball)

        self.cb_show_ground = QCheckBox("Show ground plane")
        self.cb_show_ground.setChecked(self.show_ground)
        self.cb_show_ground.setToolTip("Render a translucent ground plane at z=0")
        self.cb_show_ground.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_show_ground)

        self.cb_show_torso_disk = QCheckBox(
            "Show torso-twist indicator (disk at torso joint)"
        )
        self.cb_show_torso_disk.setChecked(self.show_torso_disk)
        self.cb_show_torso_disk.setToolTip(
            "Draws a small disc at the torso revolute joint between the\n"
            "spine and the hub.  The disc orientation reflects the body\n"
            "twist (LS-RS line direction) so the rotating-disk action of\n"
            "the model is visually obvious."
        )
        self.cb_show_torso_disk.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_show_torso_disk)

        self.cb_auto_fit_axes = QCheckBox("Auto-fit axes to data")
        self.cb_auto_fit_axes.setChecked(self.auto_fit_axes)
        self.cb_auto_fit_axes.setToolTip(
            "Re-fit the 3D axis bounds to whatever skeleton + mocap target\n"
            "is currently visible.  Uses the shared\n"
            "src/shared/python/motion_matching/diagnostics/\n"
            "_skeleton_render.equalize_3d_axes helper so the view always\n"
            "stays cropped tightly around the body.  Untick to keep fixed\n"
            "[-2, 2] x [-1.5, 2] x [-1.5, 2.5] m bounds (useful for\n"
            "comparing scale across loads)."
        )
        self.cb_auto_fit_axes.stateChanged.connect(self._on_scene_toggled)
        v.addWidget(self.cb_auto_fit_axes)
        return box

    def _on_scene_toggled(self, _: int) -> None:
        self.show_ball = self.cb_show_ball.isChecked()
        self.show_ground = self.cb_show_ground.isChecked()
        self.show_torso_disk = self.cb_show_torso_disk.isChecked()
        self.auto_fit_axes = self.cb_auto_fit_axes.isChecked()
        self._redraw()

    def _build_playback_box(self) -> QGroupBox:
        box = QGroupBox("Playback")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        # Frame slider + spinbox row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Frame:"))
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setMinimumWidth(80)
        self.spin_frame.setKeyboardTracking(False)
        self.spin_frame.setToolTip("Current playback frame index")
        self.spin_frame.valueChanged.connect(self._on_frame_changed_spin)
        row1.addWidget(self.spin_frame)
        self.lbl_time = QLabel("t = — s")
        self.lbl_time.setObjectName("status")
        self.lbl_time.setMinimumWidth(110)
        row1.addWidget(self.lbl_time)
        v.addLayout(row1)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setToolTip("Scrub through the loaded mocap or trajectory")
        self.frame_slider.valueChanged.connect(self._on_frame_changed_slider)
        v.addWidget(self.frame_slider)

        # Step buttons
        step_row = QHBoxLayout()
        step_row.setSpacing(2)
        for label, delta, tip in [
            ("⏮", -(10**9), "First frame"),
            ("⏪", -10, "−10 frames"),
            ("◀", -1, "−1 frame"),
            ("▶", +1, "+1 frame"),
            ("⏩", +10, "+10 frames"),
            ("⏭", +(10**9), "Last frame"),
        ]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.setToolTip(tip)
            b.setMaximumWidth(46)
            b.clicked.connect(lambda _checked, d=delta: self._step_frame(d))
            step_row.addWidget(b)
        v.addLayout(step_row)

        # Play/pause + speed
        play_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setObjectName("primary")
        self.btn_play.setToolTip("Start or stop animated playback (Space)")
        self.btn_play.setStatusTip("Toggles playback")
        self.btn_play.clicked.connect(self._toggle_play)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(QLabel("Speed:"))
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 240)
        self.spin_speed.setValue(30)
        self.spin_speed.setSuffix(" fps")
        self.spin_speed.setToolTip("Playback speed in frames per second (1-240)")
        play_row.addWidget(self.spin_speed)
        self.cb_loop = QCheckBox("Loop")
        self.cb_loop.setChecked(True)
        self.cb_loop.setToolTip("Restart playback from the first frame on overflow")
        self.cb_loop.stateChanged.connect(
            lambda _: setattr(self, "loop_playback", self.cb_loop.isChecked())
        )
        play_row.addWidget(self.cb_loop)
        v.addLayout(play_row)

        # Speed multiplier combo + frame counter (issue #4482).
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("× Speed:"))
        self.combo_speed = QComboBox()
        from .session_schema import ALLOWED_SPEEDS as _ALLOWED_SPEEDS

        for s in _ALLOWED_SPEEDS:
            self.combo_speed.addItem(f"{s}×", float(s))
        self.combo_speed.setCurrentText("1.0×")
        self.combo_speed.currentIndexChanged.connect(
            lambda _i: setattr(
                self,
                "playback_speed",
                float(self.combo_speed.currentData() or 1.0),
            )
        )
        scale_row.addWidget(self.combo_speed)
        scale_row.addStretch(1)
        self.lbl_frame_counter = QLabel("0 / 0")
        self.lbl_frame_counter.setObjectName("status")
        scale_row.addWidget(self.lbl_frame_counter)
        v.addLayout(scale_row)

        # Show-trail toggle (default on, fading polylines for last N frames).
        trail_row = QHBoxLayout()
        self.cb_show_trail = QCheckBox("Show trail")
        self.cb_show_trail.setChecked(True)
        self.cb_show_trail.stateChanged.connect(
            lambda _: setattr(self, "show_trail", self.cb_show_trail.isChecked())
        )
        trail_row.addWidget(self.cb_show_trail)
        trail_row.addWidget(QLabel("frames:"))
        self.spin_trail = QSpinBox()
        self.spin_trail.setRange(0, 600)
        self.spin_trail.setValue(int(self.trail_frames))
        self.spin_trail.valueChanged.connect(
            lambda v: setattr(self, "trail_frames", int(v))
        )
        trail_row.addWidget(self.spin_trail)
        trail_row.addStretch(1)
        v.addLayout(trail_row)

        # Playback target selector — what advances when Play is pressed.
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Playback target:"))
        self.combo_playback_target = QComboBox()
        self.combo_playback_target.addItems(["Mocap", "Skeleton", "Both"])
        self.combo_playback_target.setCurrentText(self.playback_target)
        self.combo_playback_target.currentTextChanged.connect(
            self._on_playback_target_changed
        )
        self.combo_playback_target.setToolTip(
            "Mocap: animate the mocap target.\n"
            "Skeleton: animate the model skeleton through its loaded\n"
            "  trajectory CSV (Pose Slot → Trajectory…).\n"
            "Both: animate both, time-aligned at impact."
        )
        target_row.addWidget(self.combo_playback_target, stretch=1)
        v.addLayout(target_row)

        # Use-current-frame override
        self.cb_use_current_frame = QCheckBox(
            "Use current frame for mocap target (override pose-slot events)"
        )
        self.cb_use_current_frame.setToolTip(
            "Use the slider's current frame as the mocap target rather than event keys"
        )
        self.cb_use_current_frame.stateChanged.connect(self._on_frame_override_toggled)
        v.addWidget(self.cb_use_current_frame)

        # "Set as event" row
        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Mark current frame as event:"))
        self.combo_set_event = QComboBox()
        self.combo_set_event.setToolTip(
            "Event key to assign to the current frame when Set is pressed"
        )
        for k in _EVENT_KEYS:
            self.combo_set_event.addItem(f"{k} - {self.event_labels[k]}")
        ev_row.addWidget(self.combo_set_event)
        b_set = QPushButton("Set")
        b_set.setObjectName("preset")
        b_set.setToolTip("Mark the current frame as the selected event")
        b_set.clicked.connect(self._set_event_to_current_frame)
        ev_row.addWidget(b_set)
        b_clear = QPushButton("Clear overrides")
        b_clear.setObjectName("preset")
        b_clear.setToolTip("Drop all user-assigned event-frame overrides")
        b_clear.clicked.connect(self._clear_event_overrides)
        ev_row.addWidget(b_clear)
        v.addLayout(ev_row)
        return box

    def _build_align_box(self) -> QGroupBox:
        box = QGroupBox("Auto-Align")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        hint = QLabel(
            "Solves Rz + Tx/Ty/Tz so the model SHAFT (mid-hands → clubhead) "
            "lines up with the mocap shaft at the chosen frame."
        )
        hint.setObjectName("status")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.cb_fit_scale = QCheckBox("Also fit scale (|shaft_target| / |shaft_model|)")
        self.cb_fit_scale.setToolTip(
            "Also solve a uniform scale so the model shaft length matches the target"
        )
        v.addWidget(self.cb_fit_scale)

        # Engine selector — populated live from the canonical fit_swing
        # provider registry (#4707 slice 1/3). The Run-fit QThread is
        # wired below (slice 2/3). Save-fit JSON serialization is the
        # remaining follow-up (slice 3/3).
        # TODO(#4707 slice 3/3): persist `RunFitButton.last_result` into
        # the save-fit JSON payload.
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("Fit engine:"))
        self.combo_fit_engine = QComboBox()
        self.combo_fit_engine.setToolTip(
            "Physics engine used by the Run-fit action. "
            "Populated live from motion_matching.provider_registry."
        )
        self._populate_engine_combo()
        self.combo_fit_engine.currentTextChanged.connect(self._on_fit_engine_changed)
        engine_row.addWidget(self.combo_fit_engine, stretch=1)
        v.addLayout(engine_row)

        # Run-fit QThread widget — slice 2/3 of #4707.
        from src.tools.starting_pose_matcher.widgets.run_fit_button import (
            RunFitButton,
        )

        self.run_fit_button = RunFitButton(self)
        self.run_fit_button.set_inputs(
            target=self._live_body_target,
            engine_name=self.combo_fit_engine.currentText(),
        )
        v.addWidget(self.run_fit_button)

        # One snap button per pose-slot
        for key, slot in self.poses.items():
            btn = QPushButton(
                f"Snap {key} pose → mocap @ {slot.target_event} (shaft-aligned)"
            )
            btn.setObjectName("primary")
            btn.clicked.connect(lambda _checked, k=key: self._snap_shaft(k))
            v.addWidget(btn)

        v.addWidget(_hsep())
        # Convenience: snap mid-hands only (legacy quick-snap)
        self.btn_snap_mid = QPushButton("Snap mid-hands only (no rotation)")
        self.btn_snap_mid.setToolTip(
            "Set Tx/Ty/Tz so the FIRST visible skeleton's "
            "mid-hands lands on its mocap target.  "
            "Rotations preserved."
        )
        self.btn_snap_mid.clicked.connect(self._snap_mid_first_visible)
        v.addWidget(self.btn_snap_mid)
        return box

    def _populate_engine_combo(self) -> None:
        """Refresh the engine combo from the canonical provider registry.

        Reads :func:`provider_registry.available_engines` live every call so
        late-registering providers (or test fixtures that mutate the
        registry) are reflected. Default selection is ``"mujoco"`` when
        present, otherwise the first registered engine; an empty registry
        leaves the combo empty.
        """
        engines = provider_registry.available_engines()
        combo = self.combo_fit_engine
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(engines)
            if engines:
                default = "mujoco" if "mujoco" in engines else engines[0]
                combo.setCurrentText(default)
        finally:
            combo.blockSignals(False)

    def _on_fit_engine_changed(self, engine_name: str) -> None:
        """Forward combo selection to the run-fit widget (#4707 slice 2)."""
        if hasattr(self, "run_fit_button"):
            self.run_fit_button.set_inputs(
                target=self._live_body_target,
                engine_name=engine_name,
            )

    @property
    def selected_engine(self) -> str:
        """Return the engine name currently chosen in the combo.

        Raises:
            RuntimeError: If the combo is empty (no providers registered).
        """
        text = self.combo_fit_engine.currentText()
        if not text:
            raise RuntimeError(
                "no fit_swing engine selected; provider registry is empty"
            )
        return text

    def _build_transform_box(self) -> QGroupBox:
        box = QGroupBox("Rigid Transform + Scale")
        v = QVBoxLayout(box)
        v.setSpacing(4)

        self.s_tx = LabelledControl("Tx", "m", _T_RANGE, _T_SCALE, 3)
        self.s_ty = LabelledControl("Ty", "m", _T_RANGE, _T_SCALE, 3)
        self.s_tz = LabelledControl("Tz", "m", _T_RANGE, _T_SCALE, 3)
        v.addWidget(self.s_tx)
        v.addWidget(self.s_ty)
        v.addWidget(self.s_tz)

        v.addWidget(_hsep())

        # Rz + presets (always enabled — Z is the heading axis)
        self.s_rz = LabelledControl("Rz", "°", _R_RANGE, _R_SCALE, 1)
        v.addWidget(self.s_rz)
        rz_row = QHBoxLayout()
        rz_row.setSpacing(4)
        rz_row.addWidget(QLabel("Presets:"))
        for label, deg in [
            ("-90°", -90),
            ("-45°", -45),
            ("0°", 0),
            ("+45°", 45),
            ("+90°", 90),
            ("180°", 180),
        ]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, d=deg: self.s_rz.set_value(d))
            rz_row.addWidget(b)
        v.addLayout(rz_row)

        v.addWidget(_hsep())

        # X/Y rotation lock
        self.cb_lock_xy = QCheckBox(
            "Allow Rx/Ry rotations (off by default — Z is up in both data and model)"
        )
        self.cb_lock_xy.setToolTip(
            "Unlock the Rx/Ry sliders. Both the mocap data and the model "
            "use Z-up, so leaving this off keeps the only physically "
            "meaningful rotation (Rz, the heading) plus translation."
        )
        self.cb_lock_xy.setChecked(False)
        self.cb_lock_xy.stateChanged.connect(self._on_lock_xy_toggled)
        v.addWidget(self.cb_lock_xy)

        self.s_rx = LabelledControl("Rx", "°", _R_RANGE, _R_SCALE, 1)
        self.s_ry = LabelledControl("Ry", "°", _R_RANGE, _R_SCALE, 1)
        v.addWidget(self.s_rx)
        v.addWidget(self.s_ry)
        self.s_rx.setEnabled(False)
        self.s_ry.setEnabled(False)

        v.addWidget(_hsep())

        # Scale + presets
        self.s_scale = LabelledControl("Scale", "×", _S_RANGE, _S_SCALE, 2, default=1.0)
        v.addWidget(self.s_scale)
        sc_row = QHBoxLayout()
        sc_row.setSpacing(4)
        sc_row.addWidget(QLabel("Presets:"))
        for label, val in [
            ("0.85", 0.85),
            ("0.95", 0.95),
            ("1.00", 1.00),
            ("1.05", 1.05),
            ("1.15", 1.15),
        ]:
            b = QPushButton(label)
            b.setObjectName("preset")
            b.clicked.connect(lambda _checked, x=val: self.s_scale.set_value(x))
            sc_row.addWidget(b)
        v.addLayout(sc_row)

        # Pivot info
        pi = QLabel(
            "Pivot @ first-pose hub: ({:.3f}, {:.3f}, {:.3f}) m".format(
                *self.transform.pivot
            )
        )
        pi.setObjectName("status")
        v.addWidget(pi)

        # Wire all the changes
        for s in (
            self.s_tx,
            self.s_ty,
            self.s_tz,
            self.s_rx,
            self.s_ry,
            self.s_rz,
            self.s_scale,
        ):
            s.spin.valueChanged.connect(self._on_transform_changed)

        # Reset row
        reset_row = QHBoxLayout()
        self.btn_reset_t = QPushButton("Reset translations")
        self.btn_reset_t.setToolTip("Set Tx, Ty, Tz back to zero")
        self.btn_reset_t.clicked.connect(self._reset_translations)
        reset_row.addWidget(self.btn_reset_t)
        self.btn_reset_r = QPushButton("Reset rotations")
        self.btn_reset_r.setToolTip("Set Rx, Ry, Rz back to zero")
        self.btn_reset_r.clicked.connect(self._reset_rotations)
        reset_row.addWidget(self.btn_reset_r)
        self.btn_reset_all = QPushButton("Reset all")
        self.btn_reset_all.setToolTip("Reset all translations, rotations, and scale")
        self.btn_reset_all.clicked.connect(self._reset_all)
        reset_row.addWidget(self.btn_reset_all)
        v.addLayout(reset_row)
        return box

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Output")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        self.btn_save = QPushButton("Save offsets to JSON…")
        self.btn_save.setObjectName("accent")
        self.btn_save.setToolTip("Write the current rigid transform offsets to JSON")
        self.btn_save.setStatusTip("Saves offsets to JSON")
        self.btn_save.clicked.connect(self._on_save_clicked)
        v.addWidget(self.btn_save)

        ses_row = QHBoxLayout()
        self.btn_save_session = QPushButton("Save session…")
        self.btn_save_session.setToolTip(
            "Save the full session (transform, events, view state) to JSON"
        )
        self.btn_save_session.clicked.connect(self._on_save_session_clicked)
        ses_row.addWidget(self.btn_save_session)
        self.btn_load_session = QPushButton("Load session…")
        self.btn_load_session.setToolTip("Load a previously-saved session JSON")
        self.btn_load_session.clicked.connect(self._on_load_session_clicked)
        ses_row.addWidget(self.btn_load_session)
        v.addLayout(ses_row)

        # Subject-anthropometrics calibration (issue #4820).
        self.btn_calibrate_subject = QPushButton("Calibrate subject…")
        self.btn_calibrate_subject.setToolTip(
            "Build a SubjectAnthropometrics record from height/mass/age/sex "
            "and persist it to ~/.golf_modeling_suite/subjects/<id>.json."
        )
        self.btn_calibrate_subject.setStatusTip(
            "Open the subject-anthropometrics calibration dialog."
        )
        self.btn_calibrate_subject.clicked.connect(self._on_calibrate_subject_clicked)
        v.addWidget(self.btn_calibrate_subject)
        self.lbl_subject = QLabel("Subject: (not calibrated)")
        self.lbl_subject.setObjectName("status")
        self.lbl_subject.setWordWrap(True)
        v.addWidget(self.lbl_subject)

        self.lbl_residual = QLabel("Residuals: (no data)")
        self.lbl_residual.setObjectName("residual")
        self.lbl_residual.setWordWrap(True)
        v.addWidget(self.lbl_residual)
        return box


__all__ = ["_BuildersMixin"]
