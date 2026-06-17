# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""PyQt6 tabs for swingset policy search and chain dynamics analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
from PyQt6.QtCore import QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from movement_optimizer.models.chain_forces import ChainForceField
from movement_optimizer.models.swingset import (
    DEFAULT_OPTIMIZER_BUDGET,
    DEFAULT_POLICY_DT_S,
    MAX_OPTIMIZER_BUDGET,
    SWING_POLICY_JOINT_NAMES,
    CyclicPolicyBounds,
    CyclicPolicySearchResult,
    CyclicPolicySearchSpace,
    HumanSegmentSpec,
    SwingPose,
    SwingRollout,
    SwingSetConfig,
    build_swingset_snapshot,
    estimate_swingset_joint_torques,
)
from movement_optimizer.models.swingset_forces import (
    SwingForceField,
    swing_force_fields,
    swing_force_history,
)
from movement_optimizer.rendering import Palette, get_chart_color

from . import plot_renderer
from .motion_analysis_panel import MotionAnalysisPanel
from .motion_controls import NumericControl, scrollable_control_panel
from .policy_trace_canvas import PolicyTraceCanvas, refresh_policy_trace_palette
from .policy_worker import PolicyOptimizationWorker
from .vector_overlay import (
    ComMarker,
    ForceArrow,
    OverlayScene,
    TorqueArc,
    VectorStyle,
    auto_scale_factor,
    draw_overlay_scene,
)


# Canvas colours are sourced from the fleet shared theme (via rendering.Palette
# and the shared chart-colour cycle) so the swingset/chain canvases recolour with
# the rest of the app. ``refresh_motion_palette`` rebinds these on theme change.
def _build_motion_colors() -> dict[str, QColor]:
    return {
        "ACCENT": QColor(Palette.GREEN),
        "CHAIN": QColor(Palette.FG_DIM),
        "BODY": QColor(get_chart_color(0)),
        "LEG": QColor(get_chart_color(1)),
        "ARM": QColor(get_chart_color(2)),
        "SURFACE": QColor(Palette.BG),
        "GRID": QColor(Palette.BG_INPUT),
    }


_MOTION_COLORS = _build_motion_colors()
ACCENT = _MOTION_COLORS["ACCENT"]
CHAIN = _MOTION_COLORS["CHAIN"]
BODY = _MOTION_COLORS["BODY"]
LEG = _MOTION_COLORS["LEG"]
ARM = _MOTION_COLORS["ARM"]
SURFACE = _MOTION_COLORS["SURFACE"]
GRID = _MOTION_COLORS["GRID"]


def refresh_motion_palette() -> None:
    """Rebind the motion-canvas colours from the active theme palette."""
    global ACCENT, CHAIN, BODY, LEG, ARM, SURFACE, GRID
    colors = _build_motion_colors()
    ACCENT = colors["ACCENT"]
    CHAIN = colors["CHAIN"]
    BODY = colors["BODY"]
    LEG = colors["LEG"]
    ARM = colors["ARM"]
    SURFACE = colors["SURFACE"]
    GRID = colors["GRID"]
    refresh_policy_trace_palette()


def _swing_overlay_scene(
    field: SwingForceField,
    *,
    gravity: bool,
    tension: bool,
    torque: bool,
    com: bool,
) -> OverlayScene:
    """Build the swingset overlay scene from a force field, filtered by toggles."""
    arrows: list[ForceArrow] = []
    arcs: list[TorqueArc] = []
    markers: list[ComMarker] = []
    origin = (float(field.com_m[0]), float(field.com_m[1]))
    if gravity:
        gravity_vec = (float(field.gravity_n[0]), float(field.gravity_n[1]))
        arrows.append(ForceArrow(origin, gravity_vec, VectorStyle(LEG, label="gravity")))
    if tension:
        tension_vec = (float(field.chain_tension_n[0]), float(field.chain_tension_n[1]))
        arrows.append(ForceArrow(origin, tension_vec, VectorStyle(CHAIN, label="tension")))
    if torque:
        for joint, magnitude in zip(SWING_POLICY_JOINT_NAMES, field.joint_torque_nm, strict=True):
            point = field.joint_points_m[joint]
            arcs.append(
                TorqueArc((float(point[0]), float(point[1])), float(magnitude), VectorStyle(ARM))
            )
    if com:
        markers.append(ComMarker(origin, VectorStyle(ACCENT)))
    return OverlayScene(arrows=tuple(arrows), torque_arcs=tuple(arcs), com_markers=tuple(markers))


def _chain_overlay_scene(
    field: ChainForceField,
    *,
    gravity: bool,
    tension: bool,
    net: bool,
) -> OverlayScene:
    """Build the chain overlay scene from a per-link force field, filtered by toggles."""
    arrows: list[ForceArrow] = []
    for index in range(len(field.midpoints_m)):
        origin = (float(field.midpoints_m[index][0]), float(field.midpoints_m[index][1]))
        if gravity:
            vec = (float(field.gravity_n[index][0]), float(field.gravity_n[index][1]))
            arrows.append(ForceArrow(origin, vec, VectorStyle(LEG)))
        if tension:
            vec = (float(field.tension_n[index][0]), float(field.tension_n[index][1]))
            arrows.append(ForceArrow(origin, vec, VectorStyle(CHAIN)))
        if net:
            vec = (float(field.net_force_n[index][0]), float(field.net_force_n[index][1]))
            arrows.append(ForceArrow(origin, vec, VectorStyle(ARM)))
    return OverlayScene(arrows=tuple(arrows))


class MotionCanvas(QWidget):
    """Side-view renderer for chain and articulated rider snapshots."""

    #: Drawable layers, in paint order, with their display labels.
    LAYERS: tuple[tuple[str, str], ...] = (
        ("grid", "Grid"),
        ("chain", "Chain"),
        ("rider", "Rider"),
        ("markers", "Pivot markers"),
        ("forces", "Force vectors"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(360)
        self._chain_nodes: list[tuple[float, float]] = []
        self._body_points: dict[str, tuple[float, float]] = {}
        self._overlay = OverlayScene()
        self._path_length_m = 0.5
        self._layers: dict[str, bool] = {key: True for key, _ in self.LAYERS}

    def set_layer_visible(self, name: str, visible: bool) -> None:
        """Show or hide a single drawable layer and repaint.

        Raises ValueError for an unknown layer name so wiring mistakes
        surface immediately rather than silently doing nothing.
        """
        if name not in self._layers:
            raise ValueError(f"unknown motion-canvas layer: {name!r}")
        self._layers[name] = bool(visible)
        self.update()

    def is_layer_visible(self, name: str) -> bool:
        """Return whether a drawable layer is currently visible."""
        if name not in self._layers:
            raise ValueError(f"unknown motion-canvas layer: {name!r}")
        return self._layers[name]

    def set_scene(
        self,
        chain_nodes: list[tuple[float, float]],
        body_points: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._chain_nodes = chain_nodes
        self._body_points = body_points or {}
        self._path_length_m = self._compute_chain_path_length(chain_nodes)
        self.update()

    def set_overlays(self, scene: OverlayScene) -> None:
        """Set the force/torque overlay primitives and repaint (no recompute)."""
        self._overlay = scene
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), SURFACE)
        if not self._chain_nodes:
            return
        projector = self._projector()
        chain_points = [projector(point) for point in self._chain_nodes]
        if self._layers["grid"]:
            self._draw_grid(painter)
        if self._layers["chain"]:
            self._draw_polyline(painter, chain_points, CHAIN, 3)
        if self._layers["rider"]:
            self._draw_body(painter, projector)
        if self._layers["markers"]:
            painter.setBrush(ACCENT)
            painter.setPen(QPen(ACCENT, 1))
            for point in chain_points[:1] + chain_points[-1:]:
                painter.drawEllipse(point, 5, 5)
        if self._layers["forces"]:
            self._draw_overlay(painter, projector)

    def _draw_overlay(
        self,
        painter: QPainter,
        projector: Callable[[tuple[float, float]], QPointF],
    ) -> None:
        scene = self._overlay
        if not (scene.arrows or scene.torque_arcs or scene.com_markers):
            return
        target = 0.5 * self._chain_path_length()
        arrow_scale = auto_scale_factor(scene.arrows, target) if scene.arrows else 1.0
        torque_reference = max(
            (abs(arc.magnitude_nm) for arc in scene.torque_arcs),
            default=1.0,
        )
        draw_overlay_scene(
            painter,
            projector,
            scene,
            arrow_scale=arrow_scale,
            torque_reference_nm=torque_reference or 1.0,
        )

    def _projector(self) -> Callable[[tuple[float, float]], QPointF]:
        anchor_x, anchor_y = self._chain_nodes[0]
        chain_length = self._chain_path_length()
        scale = 0.84 * min(
            self.width() / max(2.0 * chain_length, 0.5),
            self.height() / max(1.12 * chain_length, 0.5),
        )
        offset_x = 0.5 * self.width() - scale * anchor_x
        offset_y = 32.0 - scale * anchor_y

        def _project(point: tuple[float, float]) -> QPointF:
            x, y = point
            return QPointF(offset_x + scale * x, offset_y + scale * y)

        return _project

    def _chain_path_length(self) -> float:
        return self._path_length_m

    @staticmethod
    def _compute_chain_path_length(chain_nodes: list[tuple[float, float]]) -> float:
        distances = [
            np.hypot(end[0] - start[0], end[1] - start[1]) for start, end in pairwise(chain_nodes)
        ]
        return max(float(sum(distances)), 0.5)

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(GRID, 1))
        step = 40
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)

    def _draw_polyline(
        self,
        painter: QPainter,
        points: list[QPointF],
        color: QColor,
        width: int,
    ) -> None:
        painter.setPen(QPen(color, width))
        for start, end in pairwise(points):
            painter.drawLine(start, end)

    def _draw_body(
        self,
        painter: QPainter,
        projector: Callable[[tuple[float, float]], QPointF],
    ) -> None:
        if not self._body_points:
            return
        pairs = [
            ("hand", "elbow", ARM, 4),
            ("elbow", "shoulder", ARM, 4),
            ("shoulder", "waist", BODY, 5),
            ("waist", "hip", BODY, 5),
            ("hip", "knee", LEG, 4),
            ("knee", "foot", LEG, 4),
        ]
        for start, end, color, width in pairs:
            painter.setPen(QPen(color, width))
            painter.drawLine(
                projector(self._body_points[start]),
                projector(self._body_points[end]),
            )


class _MotionViewMixin:
    """Shared Animation/Plots subtab scaffolding for motion-analysis tabs.

    Hosts the per-element animation layer toggles, the animation/plots
    sub-tab split, and the plot-legend visibility control. Concrete tabs
    must provide ``self.canvas`` (a :class:`MotionCanvas`),
    ``self.analysis_panel`` (a :class:`MotionAnalysisPanel`), and the
    ``self._layer_toggles`` / ``self._plot_legend_toggle`` attributes
    referenced below.
    """

    canvas: MotionCanvas
    analysis_panel: MotionAnalysisPanel
    _layer_toggles: dict[str, QCheckBox]

    def _build_animation_view(self) -> QWidget:
        """The animation subtab: the motion canvas with full vertical room."""
        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self.canvas)
        return view

    def _build_plots_view(self) -> QWidget:
        """The plots subtab: roomy analysis plots plus appearance controls."""
        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(6)
        appearance = QHBoxLayout()
        self._plot_legend_toggle = QCheckBox("Show plot legends")
        self._plot_legend_toggle.setChecked(True)
        self._plot_legend_toggle.setToolTip(
            "Show or hide the legends on the analysis plots so they do not "
            "obscure the plotted curves."
        )
        self._plot_legend_toggle.stateChanged.connect(self._refresh_plot_legends)
        appearance.addWidget(self._plot_legend_toggle)
        appearance.addStretch()
        view_layout.addLayout(appearance)
        view_layout.addWidget(self.analysis_panel, stretch=1)
        return view

    def _build_layers_group(self, layer_keys: Sequence[str] | None = None) -> QGroupBox:
        """Build the "Show in animation" checklist.

        ``layer_keys`` restricts the checklist to the layers a given tab
        actually draws (e.g. the chain tab has no articulated rider), so no
        inert toggles are shown. Defaults to every canvas layer.
        """
        allowed = set(layer_keys) if layer_keys is not None else None
        group = QGroupBox("Show in animation")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        tips = {
            "grid": "Background reference grid.",
            "chain": "Swing chain polyline.",
            "rider": "Articulated rider body segments.",
            "markers": "Anchor and seat pivot markers.",
            "forces": "All force and torque vector overlays.",
        }
        for key, label in MotionCanvas.LAYERS:
            if allowed is not None and key not in allowed:
                continue
            checkbox = QCheckBox(label)
            checkbox.setChecked(self.canvas.is_layer_visible(key))
            checkbox.setToolTip(tips.get(key, ""))
            checkbox.stateChanged.connect(
                lambda _state, name=key, box=checkbox: self.canvas.set_layer_visible(
                    name, box.isChecked()
                )
            )
            self._layer_toggles[key] = checkbox
            layout.addWidget(checkbox)
        return group

    def _apply_plot_legend_visibility(self) -> None:
        """Match analysis-plot legend visibility to the appearance toggle."""
        self.analysis_panel.set_legends_visible(self._plot_legend_toggle.isChecked())

    def _refresh_plot_legends(self, _state: int | None = None) -> None:
        self._apply_plot_legend_visibility()
        self.analysis_panel.draw()


class SwingsetTab(_MotionViewMixin, QWidget):
    """Interactive swingset model tab with cyclic policy optimization."""

    playbackStateChanged = pyqtSignal()  # noqa: N815 - Qt signal naming convention.

    def __init__(self) -> None:
        super().__init__()
        self.canvas = MotionCanvas()
        self.metric_label = QLabel()
        self.policy_status_label = QLabel()
        self.progress_bar = QProgressBar()
        self.policy_detail_label = QLabel("Policy not optimized.")
        self.policy_detail_label.setWordWrap(True)
        self.autoplay_checkbox = QCheckBox("Autoplay after optimization")
        self.autoplay_checkbox.setChecked(True)
        self.autoplay_checkbox.setToolTip(
            "Automatically play the optimized swingset simulation when policy search finishes."
        )
        self.policy_trace_canvas = PolicyTraceCanvas()
        self.analysis_panel = MotionAnalysisPanel(
            ["torques", "power", "angle", "com_height", "energy", "com_path"],
            rows=2,
            cols=3,
        )
        self._controls: dict[str, NumericControl] = {}
        self._force_toggles: dict[str, QCheckBox] = {}
        self._layer_toggles: dict[str, QCheckBox] = {}
        self._force_history: object | None = None
        self._force_fields: tuple[SwingForceField, ...] | None = None
        self._rollout: SwingRollout | None = None
        self._frame_index = 0
        self._control_panel_visible = True
        self._control_scroll: QScrollArea | None = None
        self._control_panel_widget: QWidget | None = None
        self._policy_worker: PolicyOptimizationWorker | None = None
        self._play_after_policy = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._build_ui()
        self._refresh_static()

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self._build_animation_view(), "Animation")
        self.view_tabs.addTab(self._build_plots_view(), "Plots")
        layout.addWidget(self.view_tabs, 0, 0, 2, 1)
        self._control_panel_widget = QWidget()
        right_layout = QVBoxLayout(self._control_panel_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._build_policy_toolbar())
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(8, 0, 8, 0)
        control_layout.setSpacing(10)
        control_layout.addWidget(self._build_chain_group())
        control_layout.addWidget(self._build_body_group())
        control_layout.addWidget(self._build_layers_group())
        control_layout.addWidget(self._build_force_group())
        control_layout.addWidget(self._build_policy_group())
        control_layout.addWidget(self._build_policy_telemetry_group())
        control_layout.addStretch()
        self._control_scroll = scrollable_control_panel(control_panel)
        right_layout.addWidget(self._control_scroll)
        layout.addWidget(self._control_panel_widget, 0, 1, 2, 1)
        layout.addWidget(self.metric_label, 2, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

    def _build_force_group(self) -> QGroupBox:
        group = QGroupBox("Force vectors")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        specs = [
            ("gravity", "Gravity", "Weight vector at the rider's centre of mass."),
            ("tension", "Chain tension", "Net chain reaction supporting the rider."),
            ("torque", "Joint torque", "Per-joint torque indicators (curved arrows)."),
            ("com", "Centre of mass", "Marker at the rider's centre of mass."),
        ]
        for key, label, tip in specs:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.setToolTip(tip)
            checkbox.stateChanged.connect(self._refresh_overlays)
            self._force_toggles[key] = checkbox
            layout.addWidget(checkbox)
        return group

    def _build_policy_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QVBoxLayout(toolbar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)
        row = QHBoxLayout()
        self.optimize_button = QPushButton("Optimize Swing Policy")
        self.optimize_button.setProperty("class", "primary")
        self.optimize_button.setMinimumHeight(42)
        self.optimize_button.setAccessibleName("Optimize swing policy")
        self.optimize_button.setAccessibleDescription(
            "Run the swingset policy optimizer and populate playback, charts, and force vectors."
        )
        self.optimize_button.setToolTip(
            "Search for the rider pumping policy that maximises swing height, "
            "then plot torques/power and draw force vectors."
        )
        self.optimize_button.clicked.connect(self._optimize_policy)
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Play or pause the optimised swing animation.")
        self.play_button.clicked.connect(self._toggle_playback)
        row.addWidget(self.optimize_button)
        row.addWidget(self.play_button)
        layout.addLayout(row)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.autoplay_checkbox)
        layout.addWidget(self.policy_status_label)
        return toolbar

    def _build_chain_group(self) -> QGroupBox:
        group = QGroupBox("Swingset")
        form = QFormLayout(group)
        form.setVerticalSpacing(8)
        self._add_control(
            form,
            "segments",
            "Chain segments",
            3,
            40,
            14,
            integer=True,
            tooltip="Number of links the swing chain is divided into.",
        )
        self._add_control(
            form,
            "chain_length",
            "Chain length m",
            1.0,
            5.0,
            2.4,
            tooltip="Total length of the swing chain in metres.",
        )
        self._add_control(
            form,
            "link_mass",
            "Link mass kg",
            0.01,
            2.0,
            0.16,
            tooltip="Mass of each individual chain link.",
        )
        self._add_control(
            form,
            "seat_mass",
            "Seat mass kg",
            0.5,
            25.0,
            4.5,
            tooltip="Mass of the swing seat.",
        )
        self._add_control(
            form,
            "seat_placement",
            "Seat placement %",
            1.0,
            100.0,
            35.0,
            tooltip="Where along the thigh the seat sits, as a percentage.",
        )
        return group

    def _build_body_group(self) -> QGroupBox:
        group = QGroupBox("Rider")
        form = QFormLayout(group)
        form.setVerticalSpacing(8)
        self._add_control(
            form,
            "torso_len",
            "Torso length m",
            0.2,
            1.2,
            0.62,
            tooltip="Rider torso segment length.",
        )
        self._add_control(
            form,
            "torso_mass",
            "Torso mass kg",
            5.0,
            80.0,
            28.0,
            tooltip="Rider torso segment mass.",
        )
        self._add_control(
            form,
            "thigh_len",
            "Thigh length m",
            0.15,
            0.9,
            0.46,
            tooltip="Rider thigh segment length.",
        )
        self._add_control(
            form,
            "thigh_mass",
            "Thigh mass kg",
            1.0,
            25.0,
            8.0,
            tooltip="Rider thigh segment mass (per leg).",
        )
        self._add_control(
            form,
            "shank_len",
            "Shank length m",
            0.15,
            0.9,
            0.45,
            tooltip="Rider shank (lower leg) segment length.",
        )
        self._add_control(
            form,
            "shank_mass",
            "Shank mass kg",
            1.0,
            20.0,
            5.5,
            tooltip="Rider shank segment mass (per leg).",
        )
        self._add_control(
            form,
            "arm_len",
            "Arm segment m",
            0.1,
            0.8,
            0.30,
            tooltip="Rider arm segment length (upper arm and forearm).",
        )
        self._add_control(
            form, "arm_mass", "Arm segment kg", 0.2, 10.0, 2.0, tooltip="Rider arm segment mass."
        )
        return group

    def _build_policy_group(self) -> QGroupBox:
        group = QGroupBox("Policy")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self.iterative_checkbox = QCheckBox("Iterative optimizer")
        self.iterative_checkbox.setChecked(True)
        self.iterative_checkbox.setToolTip(
            "Use the seeded differential-evolution + local-refine optimizer "
            "(uncheck to fall back to the coarse grid search)."
        )
        layout.addWidget(self.iterative_checkbox)
        form = QFormLayout()
        form.setVerticalSpacing(8)
        self._add_control(
            form,
            "budget",
            "Optimizer budget",
            50,
            MAX_OPTIMIZER_BUDGET,
            DEFAULT_OPTIMIZER_BUDGET,
            integer=True,
            refresh=False,
            tooltip="Maximum number of policy evaluations the optimizer may spend (up to 2000).",
        )
        self._add_control(
            form,
            "seed",
            "Random seed",
            0,
            9999,
            0,
            integer=True,
            refresh=False,
            tooltip="Seed for the optimizer; identical seeds give identical, repeatable results.",
        )
        self._add_control(
            form,
            "cycles",
            "Swing cycles",
            1,
            12,
            2,
            integer=True,
            refresh=False,
            tooltip="Number of pump cycles to simulate per evaluation.",
        )
        self._add_control(
            form,
            "policy_steps",
            "Rollout steps",
            60,
            MAX_OPTIMIZER_BUDGET,
            220,
            integer=True,
            tooltip="Time steps simulated per evaluation when cycles are not used (up to 2000).",
        )
        self._add_control(form, "freq_min", "Freq min Hz", 0.2, 2.0, 0.45, refresh=False)
        self._add_control(form, "freq_max", "Freq max Hz", 0.2, 2.0, 0.75, refresh=False)
        self._add_control(
            form, "freq_samples", "Freq samples", 1, 8, 3, integer=True, refresh=False
        )
        self._add_control(form, "hip_rate_min", "Hip min rad/s", 0.0, 3.0, 0.5, refresh=False)
        self._add_control(form, "hip_rate_max", "Hip max rad/s", 0.0, 3.0, 1.3, refresh=False)
        self._add_control(form, "hip_samples", "Hip samples", 1, 8, 2, integer=True, refresh=False)
        self._add_control(form, "torso_rate_min", "Torso min rad/s", 0.0, 3.0, 0.3, refresh=False)
        self._add_control(form, "torso_rate_max", "Torso max rad/s", 0.0, 3.0, 1.1, refresh=False)
        self._add_control(
            form,
            "torso_samples",
            "Torso samples",
            1,
            8,
            2,
            integer=True,
            refresh=False,
        )
        self._add_control(form, "knee_ratio_min", "Knee ratio min", 0.0, 1.5, 0.25, refresh=False)
        self._add_control(form, "knee_ratio_max", "Knee ratio max", 0.0, 1.5, 0.65, refresh=False)
        self._add_control(
            form, "knee_samples", "Knee samples", 1, 8, 2, integer=True, refresh=False
        )
        self._add_control(
            form, "phase_samples", "Phase samples", 1, 12, 2, integer=True, refresh=False
        )
        self._add_control(form, "speed", "Playback speed", 0.25, 4.0, 1.0, refresh=False)
        layout.addLayout(form)
        return group

    def _build_policy_telemetry_group(self) -> QGroupBox:
        group = QGroupBox("Policy Telemetry")
        layout = QVBoxLayout(group)
        self._trace_legend_toggle = QCheckBox("Show trace legend")
        self._trace_legend_toggle.setChecked(self.policy_trace_canvas.legend_visible())
        self._trace_legend_toggle.setToolTip(
            "Show or hide the legend drawn over the policy-trace plot."
        )
        self._trace_legend_toggle.stateChanged.connect(
            lambda _state: self.policy_trace_canvas.set_legend_visible(
                self._trace_legend_toggle.isChecked()
            )
        )
        layout.addWidget(self.policy_trace_canvas)
        layout.addWidget(self._trace_legend_toggle)
        layout.addWidget(self.policy_detail_label)
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
            control.valueChanged.connect(self._refresh_static)
        if tooltip:
            control.setToolTip(tooltip)
            control.slider.setToolTip(tooltip)
            control.edit.setToolTip(tooltip)
        self._controls[key] = control
        form.addRow(label, control)

    def _config(self) -> SwingSetConfig:
        arm = HumanSegmentSpec(self._value("arm_len"), self._value("arm_mass"))
        return SwingSetConfig(
            chain_segments=int(self._value("segments")),
            chain_length_m=self._value("chain_length"),
            chain_link_mass_kg=self._value("link_mass"),
            seat_mass_kg=self._value("seat_mass"),
            seat_placement_thigh_fraction=self._value("seat_placement") / 100.0,
            torso=HumanSegmentSpec(self._value("torso_len"), self._value("torso_mass")),
            thigh=HumanSegmentSpec(self._value("thigh_len"), self._value("thigh_mass")),
            shank=HumanSegmentSpec(self._value("shank_len"), self._value("shank_mass")),
            upper_arm=arm,
            forearm=arm,
        )

    def _value(self, key: str) -> float:
        return self._controls[key].value()

    def _refresh_static(self) -> None:
        self._timer.stop()
        self.play_button.setText("Play")
        self._rollout = None
        self._force_fields = None
        self.policy_trace_canvas.set_trace(())
        self.policy_detail_label.setText("Policy not optimized.")
        config = self._config()
        pose = SwingPose(
            swing_angle_rad=0.12,
            torso_lean_rad=0.0,
            hip_angle_rad=0.2,
            knee_angle_rad=-0.2,
            shoulder_angle_rad=-0.35,
            elbow_angle_rad=0.08,
        )
        snapshot = build_swingset_snapshot(config, pose)
        self._render_snapshot(snapshot)
        self._force_history = None
        self.canvas.set_overlays(OverlayScene())
        self.analysis_panel.clear()
        self.analysis_panel.draw()
        self.metric_label.setText(
            f"Rider mass {config.rider_mass_kg:.1f} kg | "
            f"hand constraint {snapshot.hand_chain_error_m:.3f} m | "
            f"seat constraint {snapshot.seat_chain_error_m:.3f} m"
        )

    def _search_space(self) -> CyclicPolicySearchSpace:
        return CyclicPolicySearchSpace(
            frequency_hz_min=self._value("freq_min"),
            frequency_hz_max=self._value("freq_max"),
            frequency_samples=int(self._value("freq_samples")),
            hip_rate_min_rad_s=self._value("hip_rate_min"),
            hip_rate_max_rad_s=self._value("hip_rate_max"),
            hip_rate_samples=int(self._value("hip_samples")),
            torso_rate_min_rad_s=self._value("torso_rate_min"),
            torso_rate_max_rad_s=self._value("torso_rate_max"),
            torso_rate_samples=int(self._value("torso_samples")),
            knee_ratio_min=self._value("knee_ratio_min"),
            knee_ratio_max=self._value("knee_ratio_max"),
            knee_ratio_samples=int(self._value("knee_samples")),
            phase_samples=int(self._value("phase_samples")),
        )

    def _optimize_policy(self) -> None:
        if self._policy_worker is not None and self._policy_worker.isRunning():
            return
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.policy_status_label.setText("Evaluating swing-policy candidates")
        self.optimize_button.setEnabled(False)
        self.optimize_button.setText("Optimizing...")
        self._timer.stop()

        worker = PolicyOptimizationWorker(
            config=self._config(),
            iterative=self.iterative_checkbox.isChecked(),
            steps=int(self._value("policy_steps")),
            cycles=self._value("cycles"),
            bounds=self._policy_bounds(),
            budget=int(self._value("budget")),
            seed=int(self._value("seed")),
            search_space=self._search_space(),
            parent=self,
        )
        worker.progress.connect(self._on_policy_progress)
        worker.succeeded.connect(self._on_policy_success)
        worker.failed.connect(self._on_policy_error)
        worker.finished.connect(lambda: self._on_policy_finished(worker))
        worker.finished.connect(worker.deleteLater)
        self._policy_worker = worker
        worker.start()

    def _on_policy_progress(
        self,
        completed: int,
        total: int,
        best_score: float,
        params: object,
    ) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        frequency = getattr(params, "frequency_hz", 0.0)
        self.policy_status_label.setText(
            f"{completed}/{total} candidates | best {best_score:.3f} m | {float(frequency):.2f} Hz"
        )

    def _on_policy_success(self, result: object) -> None:
        self._set_policy_result(result)  # type: ignore[arg-type]
        if self._play_after_policy or self.autoplay_checkbox.isChecked():
            self.play_button.setText("Pause")
            self._timer.start(self._playback_interval_ms(DEFAULT_POLICY_DT_S))
        else:
            self.play_button.setText("Play")
            self._timer.stop()
        self.playbackStateChanged.emit()

    def _on_policy_error(self, message: str) -> None:
        self._play_after_policy = False
        self.policy_status_label.setText("Policy optimization failed.")
        self.policy_detail_label.setText(message)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Policy Optimization Failed", message)
        self.playbackStateChanged.emit()

    def _on_policy_finished(self, worker: PolicyOptimizationWorker) -> None:
        if self._policy_worker is worker:
            self._policy_worker = None
        self._play_after_policy = False
        self.optimize_button.setEnabled(True)
        self.optimize_button.setText("Optimize Swing Policy")

    def _policy_bounds(self) -> CyclicPolicyBounds:
        return CyclicPolicyBounds(
            frequency_hz=(self._value("freq_min"), self._value("freq_max")),
            hip_rate_rad_s=(self._value("hip_rate_min"), self._value("hip_rate_max")),
            torso_rate_rad_s=(self._value("torso_rate_min"), self._value("torso_rate_max")),
            knee_ratio=(self._value("knee_ratio_min"), self._value("knee_ratio_max")),
        )

    def _set_policy_result(self, result: CyclicPolicySearchResult) -> None:
        rollout = result.rollout
        self._rollout = rollout
        self._force_fields = None
        self._frame_index = 0
        self._render_snapshot(rollout.snapshots[0])
        self._populate_analysis_panel()
        self._refresh_overlays()
        self.policy_trace_canvas.set_trace(result.trace)
        self._set_policy_detail(result)
        params = result.parameters
        cycle_text = (
            f"{result.optimized_cycles:.1f} cycles"
            if result.optimized_cycles is not None
            else f"{len(rollout.states) - 1} steps"
        )
        self.metric_label.setText(
            f"Best height {result.objective_height_m:.3f} m | "
            f"peak angle {np.rad2deg(rollout.max_abs_swing_angle_rad()):.1f} deg | "
            f"freq {params.frequency_hz:.2f} Hz | "
            f"{result.evaluated_candidates} candidates | "
            f"{cycle_text}"
        )

    def _set_policy_detail(self, result: CyclicPolicySearchResult) -> None:
        params = result.parameters
        torques = estimate_swingset_joint_torques(
            self._config(),
            result.rollout,
            DEFAULT_POLICY_DT_S,
        )
        peak = np.max(np.abs(torques), axis=0)
        rms = np.sqrt(np.mean(np.square(torques), axis=0))
        torque_text = ", ".join(
            f"{joint} {peak_value:.1f}/{rms_value:.1f} Nm"
            for joint, peak_value, rms_value in zip(
                SWING_POLICY_JOINT_NAMES,
                peak,
                rms,
                strict=True,
            )
        )
        self.policy_detail_label.setText(
            "Policy: "
            f"frequency {params.frequency_hz:.2f} Hz, "
            f"hip rate {params.hip_rate_amplitude_rad_s:.2f} rad/s, "
            f"torso rate {params.torso_rate_amplitude_rad_s:.2f} rad/s, "
            f"knee ratio {params.knee_rate_ratio:.2f}, "
            f"phase {np.rad2deg(params.phase_rad):.1f} deg. "
            f"Peak torque/RMS: {torque_text}."
        )

    def _render_snapshot(self, snapshot) -> None:
        self.canvas.set_scene(
            [tuple(point) for point in snapshot.chain_nodes],
            {key: tuple(value) for key, value in snapshot.points.items()},
        )

    def _populate_analysis_panel(self) -> None:
        if self._rollout is None:
            return
        history = swing_force_history(self._config(), self._rollout, DEFAULT_POLICY_DT_S)
        self._force_history = history
        self._force_fields = swing_force_fields(self._config(), self._rollout, DEFAULT_POLICY_DT_S)
        panel = self.analysis_panel
        panel.clear()
        plot_renderer.plot_swing_joint_torques(panel.axes["torques"], history)
        plot_renderer.plot_swing_joint_power(panel.axes["power"], history)
        plot_renderer.plot_swing_angle(panel.axes["angle"], history)
        plot_renderer.plot_swing_com_height(panel.axes["com_height"], history)
        plot_renderer.plot_swing_energy(panel.axes["energy"], history)
        plot_renderer.plot_swing_com_path(panel.axes["com_path"], history)
        self._apply_plot_legend_visibility()
        panel.draw()

    def _refresh_overlays(self, _state: int | None = None) -> None:
        """Rebuild the canvas force overlay from the current frame and toggles.

        Reads the cached rollout only -- never re-runs the optimizer -- so
        toggling a checkbox is a cheap redraw.
        """
        if self._rollout is None:
            self.canvas.set_overlays(OverlayScene())
            return
        field = self._current_force_field()
        scene = _swing_overlay_scene(
            field,
            gravity=self._force_toggles["gravity"].isChecked(),
            tension=self._force_toggles["tension"].isChecked(),
            torque=self._force_toggles["torque"].isChecked(),
            com=self._force_toggles["com"].isChecked(),
        )
        self.canvas.set_overlays(scene)

    def _current_force_field(self) -> SwingForceField:
        """Return the cached force field for the active rollout frame.

        Preconditions:
            A rollout exists and ``_frame_index`` points at one of its snapshots.
        """
        if self._rollout is None:
            raise RuntimeError("DbC Blocked: force field requires an optimized rollout")
        frame_count = len(self._rollout.snapshots)
        if not 0 <= self._frame_index < frame_count:
            raise RuntimeError("DbC Blocked: frame index is outside the rollout")
        if self._force_fields is None or len(self._force_fields) != frame_count:
            self._force_fields = swing_force_fields(
                self._config(),
                self._rollout,
                DEFAULT_POLICY_DT_S,
            )
        force_fields = self._force_fields
        if force_fields is None:
            raise RuntimeError("DbC Blocked: force-field cache was not initialized")
        return force_fields[self._frame_index]

    def _toggle_playback(self) -> None:
        if self._rollout is None:
            self._play_after_policy = True
            self._optimize_policy()
            self.play_button.setText("Pause")
            self.playbackStateChanged.emit()
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("Play")
            self.playbackStateChanged.emit()
            return
        self.play_button.setText("Pause")
        self._timer.start(self._playback_interval_ms(DEFAULT_POLICY_DT_S))
        self.playbackStateChanged.emit()

    def playback_toggle(self) -> None:
        self._toggle_playback()

    def playback_step_forward(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = min(self._frame_index + 1, len(self._rollout.snapshots) - 1)
        self._render_snapshot(self._rollout.snapshots[self._frame_index])
        self._refresh_overlays()
        self.playbackStateChanged.emit()

    def playback_step_back(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = max(self._frame_index - 1, 0)
        self._render_snapshot(self._rollout.snapshots[self._frame_index])
        self._refresh_overlays()
        self.playbackStateChanged.emit()

    def playback_rewind(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = 0
        self._render_snapshot(self._rollout.snapshots[self._frame_index])
        self._refresh_overlays()
        self.playbackStateChanged.emit()

    def playback_jump_to_end(self) -> None:
        self._ensure_rollout()
        if self._rollout is None:
            return
        self._timer.stop()
        self.play_button.setText("Play")
        self._frame_index = len(self._rollout.snapshots) - 1
        self._render_snapshot(self._rollout.snapshots[self._frame_index])
        self._refresh_overlays()
        self.playbackStateChanged.emit()

    def set_playback_speed(self, speed: float) -> None:
        self._controls["speed"].set_value(speed)
        if self._timer.isActive():
            self._timer.start(self._playback_interval_ms(DEFAULT_POLICY_DT_S))

    def playback_status(self) -> tuple[int, int, bool]:
        total = len(self._rollout.snapshots) if self._rollout is not None else 0
        return self._frame_index + 1 if total else 0, total, self._timer.isActive()

    def _ensure_rollout(self) -> None:
        if self._rollout is None:
            self._optimize_policy()

    def _advance_frame(self) -> None:
        if self._rollout is None:
            return
        self._frame_index = (self._frame_index + 1) % len(self._rollout.snapshots)
        self._render_snapshot(self._rollout.snapshots[self._frame_index])
        self._refresh_overlays()
        self._timer.start(self._playback_interval_ms(DEFAULT_POLICY_DT_S))
        self.playbackStateChanged.emit()

    def _playback_interval_ms(self, dt_s: float) -> int:
        speed = max(0.05, self._value("speed"))
        return max(10, round(1000.0 * dt_s / speed))

    def set_control_panel_visible(self, visible: bool) -> None:
        """Show or hide the right-side swingset parameter panel."""
        if self._control_panel_widget is None:
            raise RuntimeError("Swingset controls have not been built")
        self._control_panel_visible = bool(visible)
        self._control_panel_widget.setVisible(self._control_panel_visible)

    def control_panel_visible(self) -> bool:
        """Return whether the right-side swingset parameter panel is expanded."""
        return self._control_panel_visible


def create_swingset_tab() -> QWidget:
    return SwingsetTab()


from .motion_tabs_chain import ChainDynamicsTab as ChainDynamicsTab  # noqa: E402
from .motion_tabs_chain import create_chain_tab as create_chain_tab  # noqa: E402
