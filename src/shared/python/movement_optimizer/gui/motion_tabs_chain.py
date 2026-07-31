# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Chain dynamics analysis tab.

Extracted verbatim from :mod:`movement_optimizer.gui.motion_tabs` to keep each
module within the fleet 1500-line source budget. Behaviour is unchanged: the
shared palette globals, ``refresh_motion_palette`` and the reusable canvases
(``MotionCanvas`` etc.) remain in ``motion_tabs`` and are imported here, so
live theme recolouring continues to work exactly as before.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from movement_optimizer.models.chain_dynamics import (
    DEFAULT_BEND_DAMPING,
    DEFAULT_COUPLING,
    DEFAULT_DAMPING,
    ChainConfig,
    ChainRollout,
    ChainState,
    initial_catenary_angles,
    initial_tip_kick_velocities,
    random_wadded_chain_state,
    simulate_chain_for_duration,
    steps_for_duration,
)
from movement_optimizer.models.chain_forces import (
    ChainForceField,
    chain_force_fields,
    chain_force_history,
)

from . import plot_renderer
from .motion_analysis_panel import MotionAnalysisPanel
from .motion_controls import NumericControl, scrollable_control_panel
from .motion_tabs import (
    MotionCanvas,
    _chain_overlay_scene,
    _MotionViewMixin,
)
from .vector_overlay import OverlayScene


class ChainDynamicsTab(_MotionViewMixin, QWidget):
    """Interactive chain whip-motion analysis tab."""

    playbackStateChanged = pyqtSignal()  # noqa: N815 - Qt signal naming convention.

    def __init__(self) -> None:
        super().__init__()
        self.canvas = MotionCanvas()
        self.metric_label = QLabel()
        self.angle_edit = QLineEdit()
        self.tie_segments = QCheckBox("Tie segment starts with sag profile")
        self.tie_segments.setChecked(True)
        self.use_degrees = QCheckBox("Use degrees for typed segment angles")
        self.autoplay_checkbox = QCheckBox("Autoplay after simulation")
        self.autoplay_checkbox.setChecked(True)
        self.autoplay_checkbox.setToolTip(
            "Automatically play the chain animation when simulation finishes."
        )
        self.angle_edit.setMinimumHeight(28)
        self.analysis_panel = MotionAnalysisPanel(
            ["tension", "curvature", "energy", "tip_speed"],
            rows=2,
            cols=2,
        )
        self._controls: dict[str, NumericControl] = {}
        self._force_toggles: dict[str, QCheckBox] = {}
        self._layer_toggles: dict[str, QCheckBox] = {}
        self._force_fields: tuple[ChainForceField, ...] | None = None
        self._rollout: ChainRollout | None = None
        self._frame_index = 0
        self._dt_s = 0.01
        self._control_panel_visible = True
        self._control_scroll: QScrollArea | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self._build_animation_view(), "Animation")
        self.view_tabs.addTab(self._build_plots_view(), "Plots")
        layout.addWidget(self.view_tabs, 0, 0, 2, 1)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(8, 0, 8, 0)
        control_layout.setSpacing(10)
        controls = QGroupBox("Chain")
        form = QFormLayout(controls)
        form.setVerticalSpacing(8)
        self._add_control(
            form,
            "segments",
            "Segments",
            2,
            60,
            16,
            integer=True,
            tooltip="Number of links in the chain.",
        )
        self._add_control(
            form,
            "length",
            "Link length m",
            0.03,
            1.0,
            0.18,
            tooltip="Length of each chain link.",
        )
        self._add_control(
            form,
            "mass",
            "Link mass kg",
            0.01,
            4.0,
            0.12,
            tooltip="Mass of each chain link.",
        )
        self._add_control(
            form,
            "damping",
            "Joint damping",
            0.0,
            0.05,
            DEFAULT_DAMPING,
            tooltip="Viscous joint damping in N m s/rad.",
        )
        self._add_control(
            form,
            "bend_damping",
            "Bend damping",
            0.0,
            0.05,
            DEFAULT_BEND_DAMPING,
            tooltip="Neighbor bend-rate damping in N m s/rad.",
        )
        self._add_control(
            form,
            "coupling",
            "Bend stiffness",
            0.0,
            1.0,
            DEFAULT_COUPLING,
            tooltip="Spring stiffness coupling adjacent link angles in N m/rad.",
        )
        self._add_control(
            form,
            "sag",
            "Tied sag",
            0.0,
            180.0,
            0.35,
            tooltip="Initial catenary sag when 'Tie segment starts' is enabled.",
        )
        self._add_control(
            form,
            "kick",
            "Initial velocity",
            0.0,
            2.0,
            0.6,
            tooltip="Initial angular-velocity amplitude applied along the chain.",
        )
        self._add_control(
            form,
            "steps",
            "Computed steps",
            1,
            10000,
            180,
            integer=True,
            refresh=False,
            tooltip="Integration steps (auto-computed from simulation time / time step).",
        )
        self._add_control(
            form,
            "duration",
            "Simulation time s",
            0.05,
            20.0,
            1.8,
            tooltip="Total simulated duration of the whip motion.",
        )
        self._add_control(
            form,
            "dt",
            "Time step s",
            0.002,
            0.2,
            0.01,
            tooltip="Integration time step; smaller is more accurate but slower.",
        )
        self._add_control(
            form,
            "random_span",
            "Random angle span",
            0.0,
            360.0,
            np.pi,
            tooltip="Angle range for the 'Randomize Start' wadded configuration.",
        )
        self._add_control(
            form,
            "random_seed",
            "Random seed",
            0,
            9999,
            7,
            integer=True,
            tooltip="Seed for the random start; same seed reproduces the same start.",
        )
        self._add_control(
            form,
            "speed",
            "Playback speed",
            0.25,
            4.0,
            1.0,
            refresh=False,
            tooltip="Animation playback speed multiplier.",
        )
        self.tie_segments.stateChanged.connect(self._refresh)
        form.addRow("", self.tie_segments)
        self.use_degrees.stateChanged.connect(self._refresh_angle_placeholder)
        self.use_degrees.stateChanged.connect(self._refresh)
        form.addRow("", self.use_degrees)
        form.addRow("", self.autoplay_checkbox)
        self._refresh_angle_placeholder()
        self.angle_edit.editingFinished.connect(self._refresh)
        form.addRow("Segment angles", self.angle_edit)
        control_layout.addWidget(controls)
        # The chain tab draws no articulated rider, so omit that layer.
        control_layout.addWidget(
            self._build_layers_group(["grid", "chain", "markers", "forces"])
        )
        control_layout.addWidget(self._build_force_group())
        row = QHBoxLayout()
        simulate_button = QPushButton("Simulate Whip")
        simulate_button.setToolTip(
            "Simulate the chain whip motion, then plot tension/curvature/energy "
            "and draw per-link force vectors."
        )
        simulate_button.clicked.connect(self._simulate)
        randomize_button = QPushButton("Randomize Start")
        randomize_button.setToolTip(
            "Set a random 'wadded' starting configuration (seeded)."
        )
        randomize_button.clicked.connect(self._randomize_wadded_start)
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Play or pause the simulated whip animation.")
        self.play_button.clicked.connect(self._toggle_playback)
        row.addWidget(simulate_button)
        row.addWidget(randomize_button)
        row.addWidget(self.play_button)
        control_layout.addLayout(row)
        control_layout.addWidget(self.metric_label)
        control_layout.addStretch()
        self._control_scroll = scrollable_control_panel(control_panel)
        layout.addWidget(self._control_scroll, 0, 1, 2, 1)
        layout.setColumnStretch(0, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

    def _build_force_group(self) -> QGroupBox:
        group = QGroupBox("Force vectors")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        specs = [
            ("gravity", "Gravity", "Weight vector on each chain link."),
            ("tension", "Tension", "Estimated tension transmitted along each link."),
            ("net", "Net force", "Net force (mass x acceleration) on each link."),
        ]
        for key, label, tip in specs:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.setToolTip(tip)
            checkbox.stateChanged.connect(self._refresh_overlays)
            self._force_toggles[key] = checkbox
            layout.addWidget(checkbox)
        return group

    def _add_control(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        lower: float,
        upper: float,
        value: float,
        *,
        integer: bool = False,
        refresh: bool = True,
        tooltip: str = "",
    ) -> None:
        control = NumericControl(lower, upper, value, integer=integer)
        if refresh:
            control.valueChanged.connect(self._refresh)
        if tooltip:
            control.setToolTip(tooltip)
            control.slider.setToolTip(tooltip)
            control.edit.setToolTip(tooltip)
        self._controls[key] = control
        form.addRow(label, control)

    def _config(self) -> ChainConfig:
        return ChainConfig(
            segment_count=int(self._value("segments")),
            segment_length_m=self._value("length"),
            link_mass_kg=self._value("mass"),
            damping=self._value("damping"),
            coupling=self._value("coupling"),
            bend_damping=self._value("bend_damping"),
        )

    def _state(self) -> ChainState:
        config = self._config()
        angles = (
            initial_catenary_angles(
                config.segment_count, self._angle_to_rad(self._value("sag"))
            )
            if self.tie_segments.isChecked()
            else self._typed_angles(config.segment_count)
        )
        velocities = initial_tip_kick_velocities(
            config.segment_count, self._value("kick")
        )
        return ChainState(angles, velocities)

    def _typed_angles(self, segment_count: int) -> np.ndarray:
        raw = self.angle_edit.text().strip()
        if not raw:
            return np.zeros(segment_count, dtype=np.float64)
        values = np.asarray(
            [float(part.strip()) for part in raw.split(",")], dtype=np.float64
        )
        if values.size != segment_count:
            raise ValueError(f"Expected {segment_count} segment angles")
        return np.deg2rad(values) if self.use_degrees.isChecked() else values

    def _angle_to_rad(self, value: float) -> float:
        return float(np.deg2rad(value)) if self.use_degrees.isChecked() else value

    def _randomize_wadded_start(self) -> None:
        config = self._config()
        state = random_wadded_chain_state(
            config,
            angle_span_rad=self._angle_to_rad(self._value("random_span")),
            velocity_span_rad_s=self._value("kick"),
            seed=int(self._value("random_seed")),
        )
        self.tie_segments.setChecked(False)
        values = (
            np.rad2deg(state.angles_rad)
            if self.use_degrees.isChecked()
            else state.angles_rad
        )
        self.angle_edit.setText(", ".join(f"{value:.4f}" for value in values))
        self._refresh()

    def _refresh_angle_placeholder(self) -> None:
        unit = "degrees" if self.use_degrees.isChecked() else "radians"
        self._controls["sag"].set_value(20.0 if self.use_degrees.isChecked() else 0.35)
        self._controls["random_span"].set_value(
            180.0 if self.use_degrees.isChecked() else np.pi
        )
        self.angle_edit.setPlaceholderText(f"comma-separated {unit}, one per segment")

    def _value(self, key: str) -> float:
        return self._controls[key].value()

    def _refresh(self) -> None:
        self._timer.stop()
        self.play_button.setText("Play")
        self._rollout = None
        self._force_fields = None
        try:
            config = self._config()
            state = self._state()
            positions = state.node_positions(config)
            self.canvas.set_scene([tuple(point) for point in positions])
            metrics = state.metrics(config)
            curvature = (
                np.rad2deg(metrics.max_curvature_rad)
                if self.use_degrees.isChecked()
                else metrics.max_curvature_rad
            )
            unit = "deg" if self.use_degrees.isChecked() else "rad"
            self.metric_label.setText(
                f"Tip speed {metrics.tip_speed_m_s:.3f} m/s | curvature {curvature:.3f} {unit}"
            )
        except ValueError as exc:
            self.metric_label.setText(str(exc))
        self.canvas.set_overlays(OverlayScene())
        self.analysis_panel.clear()
        self.analysis_panel.draw()

    def _simulate(self) -> None:
        try:
            self._dt_s = self._value("dt")
            duration = self._value("duration")
            self._controls["steps"].set_value(steps_for_duration(duration, self._dt_s))
            self._rollout = simulate_chain_for_duration(
                self._config(),
                self._state(),
                duration_s=duration,
                dt_s=self._dt_s,
            )
        except ValueError as exc:
            self.metric_label.setText(str(exc))
            return
        self._force_fields = None
        self._frame_index = 0
        self._render_chain_frame()
        self._populate_analysis_panel()
        self.metric_label.setText(
            f"Frames {len(self._rollout.states)} | "
            f"peak tip speed {self._rollout.tip_speed_m_s.max():.3f} m/s | "
            f"real time {self._value('duration'):.2f} s"
        )
        if self.autoplay_checkbox.isChecked():
            self._start_playback()
        else:
            self._stop_playback()

    def _populate_analysis_panel(self) -> None:
        if self._rollout is None:
            return
        self._force_fields = chain_force_fields(
            self._config(), self._rollout, self._dt_s
        )
        history = chain_force_history(self._config(), self._rollout, self._dt_s)
        time_s = history.time_s
        count = len(time_s)
        panel = self.analysis_panel
        panel.clear()
        plot_renderer.plot_chain_tension(panel.axes["tension"], history)
        plot_renderer.plot_chain_curvature(panel.axes["curvature"], history)
        plot_renderer.plot_chain_energy(
            panel.axes["energy"], time_s, self._rollout.energy_j[:count]
        )
        plot_renderer.plot_chain_tip_speed(
            panel.axes["tip_speed"], time_s, self._rollout.tip_speed_m_s[:count]
        )
        self._apply_plot_legend_visibility()
        panel.draw()

    def _refresh_overlays(self, _state: int | None = None) -> None:
        """Rebuild the per-link force overlay for the current frame (no resimulation)."""
        if self._rollout is None:
            self.canvas.set_overlays(OverlayScene())
            return
        field = self._current_force_field()
        scene = _chain_overlay_scene(
            field,
            gravity=self._force_toggles["gravity"].isChecked(),
            tension=self._force_toggles["tension"].isChecked(),
            net=self._force_toggles["net"].isChecked(),
        )
        self.canvas.set_overlays(scene)

    def _current_force_field(self) -> ChainForceField:
        """Return the cached force field for the active chain frame.

        Preconditions:
            A rollout exists and ``_frame_index`` points at one of its frames.
        """
        if self._rollout is None:
            raise RuntimeError("DbC Blocked: force field requires a simulated rollout")
        frame_count = self._rollout.frame_count()
        if not 0 <= self._frame_index < frame_count:
            raise RuntimeError("DbC Blocked: frame index is outside the rollout")
        if self._force_fields is None or len(self._force_fields) != frame_count:
            self._force_fields = chain_force_fields(
                self._config(), self._rollout, self._dt_s
            )
        force_fields = self._force_fields
        if force_fields is None:
            raise RuntimeError("DbC Blocked: force-field cache was not initialized")
        return force_fields[self._frame_index]

    def _toggle_playback(self) -> None:
        created_rollout = self._rollout is None
        if self._rollout is None:
            self._simulate()
        if self._rollout is None:
            return
        if created_rollout and self._timer.isActive():
            return
        if self._timer.isActive():
            self._stop_playback()
            return
        self._start_playback()

    def playback_toggle(self) -> None:
        self._toggle_playback()

    def playback_step_forward(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = min(self._frame_index + 1, self._rollout.frame_count() - 1)
        self._render_chain_frame()
        self.playbackStateChanged.emit()

    def playback_step_back(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = max(self._frame_index - 1, 0)
        self._render_chain_frame()
        self.playbackStateChanged.emit()

    def playback_rewind(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = 0
        self._render_chain_frame()
        self.playbackStateChanged.emit()

    def playback_jump_to_end(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = self._rollout.frame_count() - 1
        self._render_chain_frame()
        self.playbackStateChanged.emit()

    def set_playback_speed(self, speed: float) -> None:
        self._controls["speed"].set_value(speed)
        if self._timer.isActive():
            self._timer.start(self._playback_interval_ms())
        self.playbackStateChanged.emit()

    def playback_status(self) -> tuple[int, int, bool]:
        total = self._rollout.frame_count() if self._rollout is not None else 0
        return self._frame_index + 1 if total else 0, total, self._timer.isActive()

    def _ensure_rollout(self) -> None:
        if self._rollout is None:
            self._simulate()

    def _advance_frame(self) -> None:
        if self._rollout is None:
            return
        self._frame_index = (self._frame_index + 1) % self._rollout.frame_count()
        self._render_chain_frame()
        self._timer.start(self._playback_interval_ms())
        self.playbackStateChanged.emit()

    def _start_playback(self) -> None:
        self.play_button.setText("Pause")
        self._timer.start(self._playback_interval_ms())
        self.playbackStateChanged.emit()

    def _stop_playback(self) -> None:
        self._timer.stop()
        self.play_button.setText("Play")
        self.playbackStateChanged.emit()

    def _render_chain_frame(self) -> None:
        if self._rollout is None:
            return
        self.canvas.set_scene(
            [tuple(point) for point in self._rollout.positions[self._frame_index]]
        )
        self._refresh_overlays()

    def _playback_interval_ms(self) -> int:
        speed = max(0.05, self._value("speed"))
        return max(10, round(1000.0 * self._dt_s / speed))

    def set_control_panel_visible(self, visible: bool) -> None:
        """Show or hide the right-side chain parameter panel."""
        if self._control_scroll is None:
            raise RuntimeError("Chain controls have not been built")
        self._control_panel_visible = bool(visible)
        self._control_scroll.setVisible(self._control_panel_visible)

    def control_panel_visible(self) -> bool:
        """Return whether the right-side chain parameter panel is expanded."""
        return self._control_panel_visible


def create_chain_tab() -> QWidget:
    return ChainDynamicsTab()
