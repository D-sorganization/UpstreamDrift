# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
ToolStrip — persistent header bar with stacked overlay controls.

Row 1 (Actions):   Title | Run  Reset  ▶Play | Speed | [frame slider] | Frame# | ⤢ Reset View
Row 2 (Overlays):  Stacked vertical overlay controls (Force/Mobility/Force Ell each with
                   checkbox + scale slider) | ☑ Zero-τ | ☑ COM | Status

Design by Contract
------------------
Stateless: emits signals only; does not own simulation data.

Closes #1098: Stack overlay sliders vertically in toolbar section
Closes #1099: Force vector checkbox is toolstrip-only
Closes #1134: Font sizes increased for visibility
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.pendulum_simulator.gui.responsive import (
    apply_overlay_checkbox_sizing,
)

# ---------------------------------------------------------------------------
# Stylesheet constants
# ---------------------------------------------------------------------------

_STYLE_STRIP = (
    "QWidget#toolstrip {background: #16162e;border-bottom: 1px solid #2a2a50;}"
)
_BTN_RUN = (
    "QPushButton{"
    "background:#1e5c30;color:#a8f0b8;border:none;border-radius:5px;"
    "padding:3px 12px;font-weight:bold;font-size:11px;"
    "}"
    "QPushButton:hover{background:#28784a;}"
    "QPushButton:pressed{background:#133d20;}"
    "QPushButton:disabled{background:#243424;color:#507050;}"
)
_BTN_RESET = (
    "QPushButton{"
    "background:#50282a;color:#f0b0b0;border:none;border-radius:5px;"
    "padding:3px 9px;font-size:10px;"
    "}"
    "QPushButton:hover{background:#6a3838;}"
    "QPushButton:disabled{background:#332020;color:#604040;}"
)
_BTN_PLAY = (
    "QPushButton{"
    "background:#1e2a50;color:#b0bce0;border:1px solid #303060;"
    "border-radius:5px;padding:3px 9px;font-size:10px;"
    "}"
    "QPushButton:checked{background:#50402a;color:#f0d070;border-color:#706030;}"
    "QPushButton:hover{background:#282860;}"
)
_BTN_SMALL = (
    "QPushButton{background:#1e2440;color:#b0b0d8;border:1px solid #303060;"
    "border-radius:4px;padding:3px 10px;font-size:10px;font-weight:bold;}"
    "QPushButton:hover{background:#252860;color:#d0d0f0;}"
)
_CHK_FORCE = (
    "QCheckBox{color:#a0e0b0;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #405068;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#285040;border-color:#60a060;}"
)
_CHK_MOB = (
    "QCheckBox{color:#80c8f0;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #304060;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#204060;border-color:#4080c0;}"
)
_CHK_FELL = (
    "QCheckBox{color:#f0b880;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #604020;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#604020;border-color:#c08040;}"
)
_CHK_ZERO = (
    "QCheckBox{color:#d0a0e0;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #604080;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#602080;border-color:#a060c0;}"
)
_CHK_COM = (
    "QCheckBox{color:#e0e060;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #606020;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#505010;border-color:#a0a030;}"
)
_CHK_TORQUE = (
    "QCheckBox{color:#f08080;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #604040;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#604040;border-color:#c06060;}"
)
_CHK_MOF = (
    "QCheckBox{color:#8080f0;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #404060;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#304080;border-color:#6060c0;}"
)
_CHK_SUM = (
    "QCheckBox{color:#60c060;font-size:11px;spacing:3px;}"
    "QCheckBox::indicator{width:14px;height:14px;border:1px solid #406040;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#305030;border-color:#60a060;}"
)
_SLIDER_FORCE = (
    "QSlider::groove:horizontal{height:3px;background:#203028;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#60b070;border:none;"
    "width:10px;height:10px;margin:-4px 0;border-radius:5px;}"
    "QSlider::sub-page:horizontal{background:#406850;border-radius:2px;}"
)
_SLIDER_MOB = (
    "QSlider::groove:horizontal{height:3px;background:#202840;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#5898d0;border:none;"
    "width:10px;height:10px;margin:-4px 0;border-radius:5px;}"
    "QSlider::sub-page:horizontal{background:#305880;border-radius:2px;}"
)
_SLIDER_FELL = (
    "QSlider::groove:horizontal{height:3px;background:#402814;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#d08848;border:none;"
    "width:10px;height:10px;margin:-4px 0;border-radius:5px;}"
    "QSlider::sub-page:horizontal{background:#805028;border-radius:2px;}"
)
_PROGRESS_BAR = (
    "QSlider::groove:horizontal{height:10px;background:#1a1a36;border-radius:5px;"
    "border:1px solid #3a3a5a;}"
    "QSlider::sub-page:horizontal{background:qlineargradient("
    "x1:0,y1:0,x2:1,y2:0,stop:0 #2a5080,stop:1 #4888c8);"
    "border-radius:5px;}"
    "QSlider::handle:horizontal{background:#70a8e0;border:2px solid #4080c0;"
    "width:16px;height:18px;margin:-5px 0;border-radius:8px;}"
    "QSlider::handle:horizontal:hover{background:#90c0f0;"
    "border-color:#60a0e0;}"
)
_SEP_STYLE = "QFrame{color:#2a2a50;border:none;}"
_SEP_H_STYLE = "QFrame{color:#2a2a50;border:none;max-height:1px;}"
_LABEL = "color:#606080;font-size:12px;"
_VAL_LBL = "color:#8080b0;font-size:12px;font-family:monospace;min-width:32px;"
_FRAME_LBL = "color:#6060a0;font-size:12px;font-family:monospace;"
_TITLE = "color:#9090c8;font-size:14px;font-weight:bold;letter-spacing:1px;padding-right:4px;"
_OVERLAY_SECTION = (
    "QFrame#overlay_section {"
    "background: #12122a;"
    "border: 1px solid #2a2a4a;"
    "border-radius: 4px;"
    "padding: 2px;"
    "}"
)


def _vline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFixedWidth(1)
    sep.setStyleSheet(_SEP_STYLE)
    return sep


def _hline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFixedHeight(1)
    sep.setStyleSheet(_SEP_H_STYLE)
    return sep


def _make_scale_slider(style: str, default: int = 10, max_val: int = 1000) -> QSlider:
    """Create a compact scale slider (1–max_val, default=10 → 1.0×).

    max_val=1000 → 0.1×…100× (force vectors, which can be very large)
    max_val=100  → 0.1×…10×  (ellipsoids, more subtle visual scaling)
    """
    if not (style is not None):
        raise ValueError("style must be provided")
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(1, max_val)
    s.setValue(default)
    s.setStyleSheet(style)
    s.setFixedHeight(14)
    s.setMaximumWidth(160)
    return s


def _fmt_scale(raw: int) -> str:
    v = raw / 10.0
    return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×"


def _overlay_row(
    checkbox: QCheckBox,
    slider: QSlider,
    label: QLabel,
) -> QHBoxLayout:
    """Build a single overlay row: [☑ Name] [---slider---] [value]."""
    if not (checkbox is not None):
        raise ValueError("checkbox must be provided")
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(3)
    apply_overlay_checkbox_sizing(checkbox)
    row.addWidget(checkbox)
    row.addWidget(slider, stretch=1)
    row.addWidget(label)
    return row


class ToolStrip(QWidget):
    """Persistent header toolbar with stacked overlay controls.

    Row 1: simulation actions + playback.
    Row 2: Stacked overlay controls (force/mob/force-ell) + extra toggles.
    """

    # Simulation lifecycle
    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_scrubbed = pyqtSignal(int)

    # Overlay toggles
    forces_toggled = pyqtSignal(bool)
    zero_torque_toggled = pyqtSignal(bool)
    mob_ellipsoid_toggled = pyqtSignal(bool)
    force_ellipsoid_toggled = pyqtSignal(bool)
    com_toggled = pyqtSignal(bool)

    # Export actions (#1141)
    export_data_requested = pyqtSignal()
    export_video_requested = pyqtSignal()

    # Pop-out chart (#1135)
    popout_chart_requested = pyqtSignal()

    # Per-segment visibility: emits set[str] | None (#1100, #1101, #1102)
    segment_visibility_changed = pyqtSignal(object)

    # 3D mode toggle (#1155)
    mode_3d_toggled = pyqtSignal(bool)

    # Scale changes
    force_scale_changed = pyqtSignal(float)
    mob_scale_changed = pyqtSignal(float)
    force_ell_scale_changed = pyqtSignal(float)

    # View controls
    reset_view_requested = pyqtSignal()

    # Rotation controls (#1146)
    azimuth_changed = pyqtSignal(float)  # radians
    tilt_changed = pyqtSignal(float)  # radians

    # Physics display toggles (#1208)
    torque_vectors_toggled = pyqtSignal(bool)
    moment_of_force_toggled = pyqtSignal(bool)
    sum_moments_toggled = pyqtSignal(bool)

    # Model selection (#1149)
    model_changed = pyqtSignal(int)

    # Loop playback toggle
    loop_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolstrip")
        self.setFixedHeight(136)
        self.setStyleSheet(_STYLE_STRIP)
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.setSpacing(1)

        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self._build_row1(row1)
        outer.addLayout(row1)

        outer.addWidget(_hline())

        # Row 2: overlay section (stacked vertically) + extra toggles
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self._build_overlay_section(row2)
        outer.addLayout(row2)

    def _add_separator(self, layout: QHBoxLayout) -> None:
        layout.addWidget(_vline())

        if not (layout is not None):
            raise ValueError("layout must be provided")
        title = QLabel("Pendulums")
        title.setStyleSheet(_TITLE)
        title.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["Double Pendulum", "Triple Pendulum", "Upper Body"])
        self.cmb_model.setToolTip("Switch between pendulum models")
        self.cmb_model.setStyleSheet(
            "QComboBox { background: #303050; color: #c0c0d8; border: 1px solid #505070;"
            "  border-radius: 3px; padding: 2px 6px; font-size: 11px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #252540; color: #c0c0d8;"
            "  selection-background-color: #3b6eb0; }"
        )
        self.cmb_model.currentIndexChanged.connect(self.model_changed.emit)
        layout.addWidget(self.cmb_model)

    def _build_row1_transport_controls(self, layout: QHBoxLayout) -> None:
        self.btn_run = QPushButton("▶ Run")
        self.btn_run.setStyleSheet(_BTN_RUN)
        self.btn_run.setToolTip("Run simulation")
        self.btn_run.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.btn_run)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setStyleSheet(_BTN_RESET)
        self.btn_reset.setToolTip("Reset to initial state")
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.btn_reset)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(_BTN_PLAY)
        self.btn_play.setToolTip("Play / Pause animation")
        self.btn_play.toggled.connect(self._on_play_toggled)
        layout.addWidget(self.btn_play)

        self.chk_loop = QCheckBox("🔁")
        self.chk_loop.setToolTip("Loop animation")
        self.chk_loop.setStyleSheet(
            "QCheckBox{color:#8080b0;font-size:13px;spacing:2px;}"
            "QCheckBox::indicator{width:14px;height:14px;border:1px solid #404060;"
            "border-radius:3px;background:#1a1a2a;}"
            "QCheckBox::indicator:checked{background:#304060;border-color:#5080c0;}"
        )
        self.chk_loop.toggled.connect(self.loop_toggled.emit)
        layout.addWidget(self.chk_loop)

        spd_lbl = QLabel("Speed:")
        spd_lbl.setStyleSheet(_LABEL)
        layout.addWidget(spd_lbl)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.05, 20.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setFixedWidth(58)
        self.speed_spin.setStyleSheet(
            "QDoubleSpinBox{background:#1a1a2e;color:#b0b0d8;"
            "border:1px solid #303050;border-radius:4px;padding:1px 2px;"
            "font-size:10px;}"
        )
        self.speed_spin.setToolTip("Playback speed multiplier (0.05× – 20×)")
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        layout.addWidget(self.speed_spin)

        scrub_lbl = QLabel("Playback:")
        scrub_lbl.setStyleSheet(_LABEL)
        layout.addWidget(scrub_lbl)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setStyleSheet(_PROGRESS_BAR)
        self._frame_slider.setToolTip("Drag to scrub through animation frames")
        self._frame_slider.setMinimumWidth(200)
        self._frame_slider.setFixedHeight(20)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        layout.addWidget(self._frame_slider, stretch=1)

        self._frame_lbl = QLabel("0%")
        self._frame_lbl.setStyleSheet(_FRAME_LBL)
        self._frame_lbl.setFixedWidth(90)
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._frame_lbl)

    def _build_row1_secondary_actions(self, layout: QHBoxLayout) -> None:
        self.btn_reset_view = QPushButton("⤢ Reset View")
        self.btn_reset_view.setStyleSheet(_BTN_SMALL)
        self.btn_reset_view.setToolTip(
            "Reset zoom & pan to default\n(shortcut: double-click canvas)"
        )
        self.btn_reset_view.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(self.btn_reset_view)

        self.btn_export_csv = QPushButton("📄 Export CSV")
        self.btn_export_csv.setStyleSheet(_BTN_SMALL)
        self.btn_export_csv.setToolTip("Export simulation data to CSV")
        self.btn_export_csv.clicked.connect(self.export_data_requested.emit)
        layout.addWidget(self.btn_export_csv)

        self.btn_export_video = QPushButton("🎬 Export Video")
        self.btn_export_video.setStyleSheet(_BTN_SMALL)
        self.btn_export_video.setToolTip("Export animation as video")
        self.btn_export_video.clicked.connect(self.export_video_requested.emit)
        layout.addWidget(self.btn_export_video)

        self.btn_eom = QPushButton("📐 Equations of Motion")
        self.btn_eom.setStyleSheet(_BTN_SMALL)
        self.btn_eom.setToolTip("Show Equations of Motion derivation")
        self.btn_eom.clicked.connect(self._show_eom_popup)
        layout.addWidget(self.btn_eom)

        self.btn_mass_matrix = QPushButton("📊 Mass Matrix")
        self.btn_mass_matrix.setStyleSheet(_BTN_SMALL)
        self.btn_mass_matrix.setToolTip("Show Mass Matrix explanation")
        self.btn_mass_matrix.clicked.connect(self._show_mass_matrix_popup)
        layout.addWidget(self.btn_mass_matrix)

        self.btn_popout = QPushButton("📈 Pop-Out Chart")
        self.btn_popout.setStyleSheet(_BTN_SMALL)
        self.btn_popout.setToolTip(
            "Pop out current simulation data as a\ndetachable chart with regression fitting"
        )
        self.btn_popout.clicked.connect(self.popout_chart_requested.emit)
        layout.addWidget(self.btn_popout)

    def _build_row1_diagnostics_button(self, layout: QHBoxLayout) -> None:
        self.btn_diagnostics = QPushButton("🔍 Diagnostics")
        self.btn_diagnostics.setStyleSheet(
            "QPushButton{background:#2a1a2a;color:#d0a0d0;border:1px solid #503060;"
            "border-radius:4px;padding:3px 10px;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#3a2a3a;color:#e0b0e0;}"
        )
        self.btn_diagnostics.setToolTip(
            "Open diagnostics tracker — view all errors,\n"
            "warnings, and system events for troubleshooting"
        )
        self.btn_diagnostics.clicked.connect(self._show_diagnostics)
        layout.addWidget(self.btn_diagnostics)

    def _make_segment_checkbox(self, label: str, checked: bool = True) -> QCheckBox:
        chk = QCheckBox(label)
        chk.setChecked(checked)
        chk.setStyleSheet(
            "QCheckBox{color:#707090;font-size:11px;spacing:2px;}"
            "QCheckBox::indicator{width:11px;height:11px;border:1px solid #404060;"
            "border-radius:2px;background:#1a1a2a;}"
            "QCheckBox::indicator:checked{background:#303068;border-color:#5050a0;}"
        )
        chk.toggled.connect(self._on_segment_toggled)
        return chk

    def _clear_segment_checks(self) -> None:
        for chk in self._segment_checks.values():
            chk.setParent(None)
            chk.deleteLater()
        self._segment_checks.clear()

    def _segment_row_layout(self) -> QHBoxLayout | None:
        overlay_frame: QFrame | None = self.findChild(QFrame, "overlay_section")
        if overlay_frame is None:
            return None
        overlay_layout = overlay_frame.layout()
        if overlay_layout is None:
            return None
        seg_item = overlay_layout.itemAt(overlay_layout.count() - 1)
        if seg_item is None:
            return None
        return seg_item.layout()  # type: ignore[return-value]

    def _build_row1(self, layout: QHBoxLayout) -> None:
        """Actions row: Title | Run Reset Play | Speed | [frame slider] | Frame# | Reset View"""

        if layout is None:
            raise ValueError("layout must be provided")
        self._build_row1_title_and_model(layout)
        self._add_separator(layout)
        self._build_row1_transport_controls(layout)
        self._add_separator(layout)
        self._build_row1_secondary_actions(layout)
        self._add_separator(layout)
        self._build_row1_diagnostics_button(layout)

    def _show_eom_popup(self) -> None:
        """Open the Equations of Motion popup (#1144)."""
        from .equations_popup import EquationTopic, show_equations_popup

        show_equations_popup(self, EquationTopic.EQUATIONS_OF_MOTION)

    def _show_mass_matrix_popup(self) -> None:
        """Open the Mass Matrix popup (#1136)."""
        from .equations_popup import EquationTopic, show_equations_popup

        show_equations_popup(self, EquationTopic.MASS_MATRIX)

    def _show_diagnostics(self) -> None:
        """Open the diagnostics tracker viewer."""
        from .diagnostics import get_tracker

        get_tracker().show_viewer(self)

    def _overlay_build_frame_rows(self, overlay_layout: QVBoxLayout) -> None:
        """Build rows A-D: force/mobility/force-ellipsoid checkboxes and segment row."""
        # Row A: Force Vectors
        self.chk_forces = QCheckBox("Force Vectors")
        self.chk_forces.setStyleSheet(_CHK_FORCE)
        self.chk_forces.setToolTip(
            "Show net joint force vectors at each joint.\n"
            "Arrow length scales with force magnitude."
        )
        self.chk_forces.toggled.connect(self.forces_toggled.emit)
        self._sld_force = _make_scale_slider(_SLIDER_FORCE, default=10)
        self._sld_force.setToolTip("Force vector display scale (0.1x - 100x)")
        self._sld_force.valueChanged.connect(self._on_force_scale)
        self._lbl_force_scale = QLabel("1.0x")
        self._lbl_force_scale.setStyleSheet(_VAL_LBL)
        overlay_layout.addLayout(
            _overlay_row(self.chk_forces, self._sld_force, self._lbl_force_scale)
        )
        # Row B: Mobility Ellipsoids
        self.chk_mob = QCheckBox("Mobility Ellipsoids")
        self.chk_mob.setStyleSheet(_CHK_MOB)
        self.chk_mob.setToolTip(
            "Show mobility ellipsoids at segment endpoints.\n"
            "Cyan = achievable velocity; large = high dexterity."
        )
        self.chk_mob.toggled.connect(self.mob_ellipsoid_toggled.emit)
        self._sld_mob = _make_scale_slider(_SLIDER_MOB, default=10, max_val=100)
        self._sld_mob.setToolTip("Mobility ellipsoid display scale (0.1x - 10x)")
        self._sld_mob.valueChanged.connect(self._on_mob_scale)
        self._lbl_mob_scale = QLabel("1.0x")
        self._lbl_mob_scale.setStyleSheet(_VAL_LBL)
        overlay_layout.addLayout(
            _overlay_row(self.chk_mob, self._sld_mob, self._lbl_mob_scale)
        )
        # Row C: Force Ellipsoids
        self.chk_force_ell = QCheckBox("Force Ellipsoids")
        self.chk_force_ell.setStyleSheet(_CHK_FELL)
        self.chk_force_ell.setToolTip(
            "Show force ellipsoids at segment endpoints.\n"
            "Orange = achievable endpoint force; small = near-singular."
        )
        self.chk_force_ell.toggled.connect(self.force_ellipsoid_toggled.emit)
        self._sld_force_ell = _make_scale_slider(_SLIDER_FELL, default=10, max_val=100)
        self._sld_force_ell.setToolTip("Force ellipsoid display scale (0.1x - 10x)")
        self._sld_force_ell.valueChanged.connect(self._on_force_ell_scale)
        self._lbl_force_ell_scale = QLabel("1.0x")
        self._lbl_force_ell_scale.setStyleSheet(_VAL_LBL)
        overlay_layout.addLayout(
            _overlay_row(
                self.chk_force_ell, self._sld_force_ell, self._lbl_force_ell_scale
            )
        )
        # Row D: Per-segment visibility sub-checkboxes (#1100, #1101, #1102)
        seg_row = QHBoxLayout()
        seg_row.setContentsMargins(0, 1, 0, 0)
        seg_row.setSpacing(2)
        seg_lbl = QLabel("Segments:")
        seg_lbl.setStyleSheet("color:#505070;font-size:11px;")
        seg_row.addWidget(seg_lbl)
        self._segment_checks: dict[str, QCheckBox] = {}
        self._segment_names: list[str] = ["shoulder", "wrist", "tip"]
        for name in self._segment_names:
            chk = QCheckBox(name[:6])
            chk.setChecked(True)
            chk.setStyleSheet(
                "QCheckBox{color:#707090;font-size:11px;spacing:2px;}"
                "QCheckBox::indicator{width:11px;height:11px;border:1px solid #404060;"
                "border-radius:2px;background:#1a1a2a;}"
                "QCheckBox::indicator:checked{background:#303068;border-color:#5050a0;}"
            )
            chk.toggled.connect(self._on_segment_toggled)
            seg_row.addWidget(chk)
            self._segment_checks[name] = chk
        seg_row.addStretch()
        overlay_layout.addLayout(seg_row)

    def _overlay_build_extra_col(self, layout: QHBoxLayout) -> None:
        """Build the extra toggles column (right of the overlay frame)."""
        extra_col = QVBoxLayout()
        extra_col.setContentsMargins(0, 0, 0, 0)
        extra_col.setSpacing(2)
        self.chk_zero_torque = QCheckBox("Zero-τ Forces")
        self.chk_zero_torque.setStyleSheet(_CHK_ZERO)
        self.chk_zero_torque.setToolTip(
            "Show zero-torque counterfactual forces (dashed vectors).\n"
            "These represent joint forces if all driving torques were removed\u2014\n"
            "the passive drift due to gravity and inertia alone."
        )
        self.chk_zero_torque.toggled.connect(self.zero_torque_toggled.emit)
        extra_col.addWidget(self.chk_zero_torque)
        self.chk_com = QCheckBox("Center of Mass")
        self.chk_com.setStyleSheet(_CHK_COM)
        self.chk_com.setToolTip("Show the combined center of mass of the whole system.")
        self.chk_com.toggled.connect(self.com_toggled.emit)
        extra_col.addWidget(self.chk_com)

        self.chk_torque = QCheckBox("Torque Vectors")
        self.chk_torque.setStyleSheet(_CHK_TORQUE)
        self.chk_torque.setToolTip(
            "Show applied torque as curved arrows at each joint.\n"
            "Red arrows \u2014 magnitude scales with torque value."
        )
        self.chk_torque.toggled.connect(self.torque_vectors_toggled.emit)
        extra_col.addWidget(self.chk_torque)

        self.chk_mof = QCheckBox("Moment of Force")
        self.chk_mof.setStyleSheet(_CHK_MOF)
        self.chk_mof.setToolTip(
            "Show moment of force from proximal segment on distal.\n"
            "Blue arrows \u2014 proximal-on-distal convention."
        )
        self.chk_mof.toggled.connect(self.moment_of_force_toggled.emit)
        extra_col.addWidget(self.chk_mof)
        self.chk_sum_moments = QCheckBox("Sum of Moments")
        self.chk_sum_moments.setStyleSheet(_CHK_SUM)
        self.chk_sum_moments.setToolTip(
            "Show sum of all moments (torque + moment of force)\n"
            "Green arrows \u2014 resultant moment at each joint."
        )
        self.chk_sum_moments.toggled.connect(self.sum_moments_toggled.emit)
        extra_col.addWidget(self.chk_sum_moments)
        self.chk_3d = QCheckBox("3D Segments")
        self.chk_3d.setStyleSheet(_CHK_COM)
        self.chk_3d.setToolTip(
            "Toggle 3D tapered segment rendering (#1155).\n"
            "Shows segments as gradient-shaded cylinders."
        )
        self.chk_3d.toggled.connect(self.mode_3d_toggled.emit)
        extra_col.addWidget(self.chk_3d)
        azimuth_row = QHBoxLayout()
        azimuth_row.setContentsMargins(0, 0, 0, 0)
        azimuth_row.setSpacing(2)
        az_lbl = QLabel("Az:")
        az_lbl.setStyleSheet("color:#606080;font-size:10px;")
        az_lbl.setToolTip("View azimuth rotation (0-360 degrees)")
        azimuth_row.addWidget(az_lbl)
        self._sld_azimuth = QSlider(Qt.Orientation.Horizontal)
        self._sld_azimuth.setRange(0, 360)
        self._sld_azimuth.setValue(0)
        self._sld_azimuth.setFixedWidth(80)
        self._sld_azimuth.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#252540;"
            "border-radius:2px;}"
            "QSlider::handle:horizontal{width:10px;margin:-3px 0;"
            "background:#6080b0;border-radius:5px;}"
        )
        self._sld_azimuth.valueChanged.connect(self._on_azimuth_slider)
        azimuth_row.addWidget(self._sld_azimuth)
        self._lbl_azimuth = QLabel("0 deg")
        self._lbl_azimuth.setStyleSheet("color:#606080;font-size:10px;min-width:30px;")
        azimuth_row.addWidget(self._lbl_azimuth)
        extra_col.addLayout(azimuth_row)
        tilt_row = QHBoxLayout()
        tilt_row.setContentsMargins(0, 0, 0, 0)
        tilt_row.setSpacing(2)
        tilt_lbl = QLabel("Tilt:")
        tilt_lbl.setStyleSheet("color:#606080;font-size:10px;")
        tilt_lbl.setToolTip("Swing plane tilt from vertical (0-90 degrees)")
        tilt_row.addWidget(tilt_lbl)
        self._sld_tilt = QSlider(Qt.Orientation.Horizontal)
        self._sld_tilt.setRange(0, 90)
        self._sld_tilt.setValue(0)
        self._sld_tilt.setFixedWidth(80)
        self._sld_tilt.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#252540;"
            "border-radius:2px;}"
            "QSlider::handle:horizontal{width:10px;margin:-3px 0;"
            "background:#608050;border-radius:5px;}"
        )
        self._sld_tilt.valueChanged.connect(self._on_tilt_slider)
        tilt_row.addWidget(self._sld_tilt)
        self._lbl_tilt = QLabel("0 deg")
        self._lbl_tilt.setStyleSheet("color:#606080;font-size:10px;min-width:30px;")
        tilt_row.addWidget(self._lbl_tilt)
        extra_col.addLayout(tilt_row)
        extra_col.addStretch()
        layout.addLayout(extra_col)

    def _overlay_build_frame_rows(self, overlay_layout: QVBoxLayout) -> None:  # type: ignore[no-redef]
        """Build rows A-D: force/mobility/force-ellipsoid checkboxes and segment row."""
        # Row A: Force Vectors
        self.chk_forces = QCheckBox("Force Vectors")
        self.chk_forces.setStyleSheet(_CHK_FORCE)
        self.chk_forces.setToolTip(
            "Show net joint force vectors at each joint.\n"
            "Arrow length scales with force magnitude."
        )
        self.chk_forces.toggled.connect(self.forces_toggled.emit)
        self._sld_force = _make_scale_slider(_SLIDER_FORCE, default=10)
        self._sld_force.setToolTip("Force vector display scale (0.1× – 100×)")
        self._sld_force.valueChanged.connect(self._on_force_scale)
        self._lbl_force_scale = QLabel("1.0×")
        self._lbl_force_scale.setStyleSheet(_VAL_LBL)
        overlay_layout.addLayout(
            _overlay_row(self.chk_forces, self._sld_force, self._lbl_force_scale)
        )
        # Row B: Mobility Ellipsoids
        self.chk_mob = QCheckBox("Mobility Ellipsoids")
        self.chk_mob.setStyleSheet(_CHK_MOB)
        self.chk_mob.setToolTip(
            "Show mobility ellipsoids at segment endpoints.\n"
            "Cyan = achievable velocity; large = high dexterity."
        )
        self.chk_mob.toggled.connect(self.mob_ellipsoid_toggled.emit)
        self._sld_mob = _make_scale_slider(_SLIDER_MOB, default=10, max_val=100)
        self._sld_mob.setToolTip("Mobility ellipsoid display scale (0.1× – 10×)")
        self._sld_mob.valueChanged.connect(self._on_mob_scale)
        self._lbl_mob_scale = QLabel("1.0×")
        self._lbl_mob_scale.setStyleSheet(_VAL_LBL)
        overlay_layout.addLayout(
            _overlay_row(self.chk_mob, self._sld_mob, self._lbl_mob_scale)
        )
        # Row C: Force Ellipsoids
        self.chk_force_ell = QCheckBox("Force Ellipsoids")
        self.chk_force_ell.setStyleSheet(_CHK_FELL)
        self.chk_force_ell.setToolTip(
            "Show force ellipsoids at segment endpoints.\n"
            "Orange = achievable endpoint force; small = near-singular."
        )
        self.chk_force_ell.toggled.connect(self.force_ellipsoid_toggled.emit)
        self._sld_force_ell = _make_scale_slider(_SLIDER_FELL, default=10, max_val=100)
        self._sld_force_ell.setToolTip("Force ellipsoid display scale (0.1× – 10×)")
        self._sld_force_ell.valueChanged.connect(self._on_force_ell_scale)
        self._lbl_force_ell_scale = QLabel("1.0×")
        self._lbl_force_ell_scale.setStyleSheet(_VAL_LBL)
        overlay_layout.addLayout(
            _overlay_row(
                self.chk_force_ell, self._sld_force_ell, self._lbl_force_ell_scale
            )
        )
        # Row D: Per-segment visibility sub-checkboxes (#1100, #1101, #1102)
        seg_row = QHBoxLayout()
        seg_row.setContentsMargins(0, 1, 0, 0)
        seg_row.setSpacing(2)
        seg_lbl = QLabel("Segments:")
        seg_lbl.setStyleSheet("color:#505070;font-size:11px;")
        seg_row.addWidget(seg_lbl)
        self._segment_checks: dict[str, QCheckBox] = {}  # type: ignore[no-redef]
        self._segment_names: list[str] = ["shoulder", "wrist", "tip"]  # type: ignore[no-redef]
        for name in self._segment_names:
            chk = QCheckBox(name[:6])
            chk.setChecked(True)
            chk.setStyleSheet(
                "QCheckBox{color:#707090;font-size:11px;spacing:2px;}"
                "QCheckBox::indicator{width:11px;height:11px;border:1px solid #404060;"
                "border-radius:2px;background:#1a1a2a;}"
                "QCheckBox::indicator:checked{background:#303068;border-color:#5050a0;}"
            )
            chk.toggled.connect(self._on_segment_toggled)
            seg_row.addWidget(chk)
            self._segment_checks[name] = chk
        seg_row.addStretch()
        overlay_layout.addLayout(seg_row)

    def _overlay_build_extra_col(self, layout: QHBoxLayout) -> None:  # type: ignore[no-redef]
        """Build the extra toggles column (right of the overlay frame)."""
        extra_col = QVBoxLayout()
        extra_col.setContentsMargins(0, 0, 0, 0)
        extra_col.setSpacing(2)
        self.chk_zero_torque = QCheckBox("Zero-\u03c4 Forces")
        self.chk_zero_torque.setStyleSheet(_CHK_ZERO)
        self.chk_zero_torque.setToolTip(
            "Show zero-torque counterfactual forces (dashed vectors).\n"
            "These represent joint forces if all driving torques were removed—\n"
            "the passive drift due to gravity and inertia alone."
        )
        self.chk_zero_torque.toggled.connect(self.zero_torque_toggled.emit)
        extra_col.addWidget(self.chk_zero_torque)
        self.chk_com = QCheckBox("Center of Mass")
        self.chk_com.setStyleSheet(_CHK_COM)
        self.chk_com.setToolTip("Show the combined center of mass of the whole system.")
        self.chk_com.toggled.connect(self.com_toggled.emit)
        extra_col.addWidget(self.chk_com)
        self.chk_torque = QCheckBox("Torque Vectors")
        self.chk_torque.setStyleSheet(_CHK_TORQUE)
        self.chk_torque.setToolTip(
            "Show applied torque as curved arrows at each joint.\n"
            "Red arrows — magnitude scales with torque value."
        )
        self.chk_torque.toggled.connect(self.torque_vectors_toggled.emit)
        extra_col.addWidget(self.chk_torque)
        self.chk_mof = QCheckBox("Moment of Force")
        self.chk_mof.setStyleSheet(_CHK_MOF)
        self.chk_mof.setToolTip(
            "Show moment of force from proximal segment on distal.\n"
            "Blue arrows — proximal-on-distal convention."
        )
        self.chk_mof.toggled.connect(self.moment_of_force_toggled.emit)
        extra_col.addWidget(self.chk_mof)
        self.chk_sum_moments = QCheckBox("Sum of Moments")
        self.chk_sum_moments.setStyleSheet(_CHK_SUM)
        self.chk_sum_moments.setToolTip(
            "Show sum of all moments (torque + moment of force)\n"
            "Green arrows — resultant moment at each joint."
        )
        self.chk_sum_moments.toggled.connect(self.sum_moments_toggled.emit)
        extra_col.addWidget(self.chk_sum_moments)
        self.chk_3d = QCheckBox("3D Segments")
        self.chk_3d.setStyleSheet(_CHK_COM)
        self.chk_3d.setToolTip(
            "Toggle 3D tapered segment rendering (#1155).\n"
            "Shows segments as gradient-shaded cylinders."
        )
        self.chk_3d.toggled.connect(self.mode_3d_toggled.emit)
        extra_col.addWidget(self.chk_3d)
        # Rotation controls (#1146)
        azimuth_row = QHBoxLayout()
        azimuth_row.setContentsMargins(0, 0, 0, 0)
        azimuth_row.setSpacing(2)
        az_lbl = QLabel("Az:")
        az_lbl.setStyleSheet("color:#606080;font-size:10px;")
        az_lbl.setToolTip("View azimuth rotation (0°-360°)")
        azimuth_row.addWidget(az_lbl)
        self._sld_azimuth = QSlider(Qt.Orientation.Horizontal)
        self._sld_azimuth.setRange(0, 360)
        self._sld_azimuth.setValue(0)
        self._sld_azimuth.setFixedWidth(80)
        self._sld_azimuth.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#252540;"
            "border-radius:2px;}"
            "QSlider::handle:horizontal{width:10px;margin:-3px 0;"
            "background:#6080b0;border-radius:5px;}"
        )
        self._sld_azimuth.valueChanged.connect(self._on_azimuth_slider)
        azimuth_row.addWidget(self._sld_azimuth)
        self._lbl_azimuth = QLabel("0°")
        self._lbl_azimuth.setStyleSheet("color:#606080;font-size:10px;min-width:30px;")
        azimuth_row.addWidget(self._lbl_azimuth)
        extra_col.addLayout(azimuth_row)
        tilt_row = QHBoxLayout()
        tilt_row.setContentsMargins(0, 0, 0, 0)
        tilt_row.setSpacing(2)
        tilt_lbl = QLabel("Tilt:")
        tilt_lbl.setStyleSheet("color:#606080;font-size:10px;")
        tilt_lbl.setToolTip("Swing plane tilt from vertical (0°-90°)")
        tilt_row.addWidget(tilt_lbl)
        self._sld_tilt = QSlider(Qt.Orientation.Horizontal)
        self._sld_tilt.setRange(0, 90)
        self._sld_tilt.setValue(0)
        self._sld_tilt.setFixedWidth(80)
        self._sld_tilt.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#252540;"
            "border-radius:2px;}"
            "QSlider::handle:horizontal{width:10px;margin:-3px 0;"
            "background:#608050;border-radius:5px;}"
        )
        self._sld_tilt.valueChanged.connect(self._on_tilt_slider)
        tilt_row.addWidget(self._sld_tilt)
        self._lbl_tilt = QLabel("0°")
        self._lbl_tilt.setStyleSheet("color:#606080;font-size:10px;min-width:30px;")
        tilt_row.addWidget(self._lbl_tilt)
        extra_col.addLayout(tilt_row)
        extra_col.addStretch()
        layout.addLayout(extra_col)

    def _build_overlay_section(self, layout: QHBoxLayout) -> None:
        """Build stacked overlay controls: three rows of [☑ checkbox] [slider] [value].

        All three overlay types (Force Vectors, Mobility Ellipsoids, Force Ellipsoids)
        are stacked vertically in a compact section.
        """
        if not (layout is not None):
            raise ValueError("layout must be provided")
        overlay_frame = QFrame()
        overlay_frame.setObjectName("overlay_section")
        overlay_frame.setStyleSheet(_OVERLAY_SECTION)
        overlay_layout = QVBoxLayout(overlay_frame)
        overlay_layout.setContentsMargins(4, 2, 4, 2)
        overlay_layout.setSpacing(1)
        self._overlay_build_frame_rows(overlay_layout)
        layout.addWidget(overlay_frame)
        layout.addWidget(_vline())
        self._overlay_build_extra_col(layout)
        layout.addWidget(_vline())
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color:#404060;font-size:11px;")
        layout.addWidget(self._status_lbl)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Slots / public API
    # ------------------------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        self.play_toggled.emit(checked)

    def _on_frame_slider_changed(self, val: int) -> None:
        if not (val is not None):
            raise ValueError("val must be provided")
        total = self._frame_slider.maximum()
        pct = int(100 * val / max(total, 1))
        self._frame_lbl.setText(f"{pct}% ({val}/{total})")
        self.frame_scrubbed.emit(val)

    def _on_force_scale(self, raw: int) -> None:
        self._lbl_force_scale.setText(_fmt_scale(raw))
        self.force_scale_changed.emit(raw / 10.0)

    def _on_mob_scale(self, raw: int) -> None:
        self._lbl_mob_scale.setText(_fmt_scale(raw))
        self.mob_scale_changed.emit(raw / 10.0)

    def _on_force_ell_scale(self, raw: int) -> None:
        self._lbl_force_ell_scale.setText(_fmt_scale(raw))
        self.force_ell_scale_changed.emit(raw / 10.0)

    def _on_azimuth_slider(self, deg: int) -> None:
        """Emit azimuth rotation in radians from slider value (#1146)."""
        if not (deg is not None):
            raise ValueError("deg must be provided")
        import numpy as np

        self._lbl_azimuth.setText(f"{deg}°")
        self.azimuth_changed.emit(np.radians(float(deg)))

    def _on_tilt_slider(self, deg: int) -> None:
        """Emit tilt rotation in radians from slider value (#1146)."""
        if not (deg is not None):
            raise ValueError("deg must be provided")
        import numpy as np

        self._lbl_tilt.setText(f"{deg}°")
        self.tilt_changed.emit(np.radians(float(deg)))

    def set_status(self, msg: str) -> None:
        """Update the right-hand status label."""
        self._status_lbl.setText(msg)

    def set_running(self, running: bool) -> None:
        """Disable run/reset while simulation is computing."""
        if not (running is not None):
            raise ValueError("running must be provided")
        self.btn_run.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.set_status("Simulating…" if running else "Ready")

    def set_frame_range(self, n_steps: int) -> None:
        """Set the playback slider maximum after simulation completes."""
        if not (n_steps >= 0):
            raise ValueError("DbC Blocked: Precondition failed.")
        self._frame_slider.setRange(0, max(0, n_steps - 1))
        self._frame_slider.setValue(0)
        self._frame_lbl.setText(f"0% (0/{max(0, n_steps - 1)})")

    def set_frame(self, idx: int) -> None:
        """Update slider + label to reflect current frame (no re-emission)."""
        if not (idx is not None):
            raise ValueError("idx must be provided")
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)
        total = self._frame_slider.maximum()
        pct = int(100 * idx / max(total, 1))
        self._frame_lbl.setText(f"{pct}% ({idx}/{total})")

    def _on_segment_toggled(self) -> None:
        """Recompute visible segments and emit the signal.

        If all segments are checked, emit None (show all).
        Otherwise emit the set of checked segment names.
        """
        checked = {
            name for name, chk in self._segment_checks.items() if chk.isChecked()
        }
        if len(checked) == len(self._segment_checks):
            self.segment_visibility_changed.emit(None)  # all visible
        else:
            self.segment_visibility_changed.emit(checked)

    def set_segment_names(self, names: list[tuple[str, str]]) -> None:  # noqa: C901
        """Update segment sub-checkboxes for the active model tab.

        Called by MainWindow when the tab index changes.

        Parameters
        ----------
        names : list[tuple[str, str]]
            List of (internal_key, display_label) tuples for the current model.
            e.g. [("shoulder", "Shoulder"), ("wrist", "Wrist")] for double.
        """
        # Remove old checkboxes
        if not (names is not None):
            raise ValueError("names must be provided")
        for chk in self._segment_checks.values():
            chk.setParent(None)
            chk.deleteLater()
        self._segment_checks.clear()
        self._segment_names = [key for key, _label in names]

        seg_layout = self._segment_row_layout()
        if seg_layout is None:
            return
        overlay_frame: QFrame | None = self.findChild(QFrame, "overlay_section")
        if overlay_frame is None:
            return
        overlay_layout = overlay_frame.layout()
        if overlay_layout is None:
            return
        seg_item = overlay_layout.itemAt(overlay_layout.count() - 1)
        if seg_item is not None and seg_item.layout() is not None:
            layout_candidate = seg_item.layout()
            if isinstance(layout_candidate, QHBoxLayout):
                seg_layout = layout_candidate
            if not (seg_layout is not None):  # narrowing for mypy
                raise ValueError("DbC Blocked: Precondition failed.")
            # Clear old widgets (keep "Segments:" label at position 0)
            while seg_layout.count() > 1:
                item = seg_layout.takeAt(1)
                w = item.widget() if item is not None else None
                if w is not None:
                    w.deleteLater()
            # Add new checkboxes — label for display, key for internal tracking
            for key, label in names:
                chk = QCheckBox(label)
                chk.setChecked(True)
                chk.setStyleSheet(
                    "QCheckBox{color:#707090;font-size:10px;spacing:2px;}"
                    "QCheckBox::indicator{width:11px;height:11px;border:1px solid #404060;"
                    "border-radius:2px;background:#1a1a2a;}"
                    "QCheckBox::indicator:checked{background:#303068;border-color:#5050a0;}"
                )
                chk.toggled.connect(self._on_segment_toggled)
                seg_layout.addWidget(chk)
                self._segment_checks[key] = chk
            if hasattr(seg_layout, "addStretch"):
                seg_layout.addStretch()

        # Emit all-visible since we just reset
        self.segment_visibility_changed.emit(None)

    def stop_play(self) -> None:
        """Force the Play button to the uncheck (stopped) state."""
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶ Play")
        self.btn_play.blockSignals(False)
