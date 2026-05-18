# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Control panel widget (double pendulum) — parameter inputs,
torque polynomial editors, gravity toggle.

Refactored to extend ControlsWidgetBase (DRY).
Model-specific: UnitAwareInput support, joint limits, torque clamps,
tilt/azimuth rotation controls.

NOTE: ``LabeledInput`` and ``_row`` are defined here and imported
by controls_widget_triple.py and controls_widget_golfer.py.
"""

from __future__ import annotations

import contextlib

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from .controls_utils import (
    STYLE_CHECK,
    STYLE_EDIT,
    STYLE_GROUP,
    STYLE_LABEL,
    clamp_dt,
    parse_coeffs,
    parse_coeffs_lenient,
    parse_float,
    require_non_negative,
    require_positive,
)
from .controls_widget_base import ControlsWidgetBase
from .torque_preview_widget import TorquePreviewWidget

try:
    from sidekick.ui.widgets.unit_aware_input import UnitAwareInput

    _HAS_UAI = True
except ImportError:
    _HAS_UAI = False


# ---------------------------------------------------------------------------
# Reusable widgets (imported by triple / golfer variants)
# ---------------------------------------------------------------------------


class LabeledInput(QWidget):
    """A label + line-edit pair used throughout the control panel."""

    def __init__(
        self,
        label: str,
        default: str,
        tooltip: str = "",
        label_width: int = 80,
        parent: QWidget | None = None,
    ) -> None:
        if not (label is not None):
            raise ValueError("label must be provided")
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        lbl = QLabel(label)
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet(STYLE_LABEL)
        layout.addWidget(lbl)

        self.edit = QLineEdit(default)
        self.edit.setStyleSheet(STYLE_EDIT)
        self.edit.setMinimumHeight(22)
        if tooltip:
            self.edit.setToolTip(tooltip)
        layout.addWidget(self.edit)

    @property
    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, text: str) -> None:
        self.edit.setText(text)


def _row(*widgets: QWidget) -> QHBoxLayout:
    """Helper: pack widgets into a horizontal row with no margin."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    for w in widgets:
        row.addWidget(w, stretch=1)
    return row


# ---------------------------------------------------------------------------
# Double pendulum controls
# ---------------------------------------------------------------------------


class ControlsWidget(ControlsWidgetBase):
    """Parameter input panel for the double pendulum model.

    Extra signals beyond base:
    - tilt_changed / azimuth_changed — real-time view rotation
    - force_scale_changed — force vector display scale
    """

    # Extra signals specific to the double pendulum
    tilt_changed = pyqtSignal(float)  # radians
    azimuth_changed = pyqtSignal(float)  # radians
    force_scale_changed = pyqtSignal(float)

    # Preset: (theta1°, phi°, dth, dph, tau_sh, tau_wr, tend, m1, m2, mClub, L1, L2)
    PRESETS = {
        "Golf Swing (passive wrist)": (
            120.0,
            -90.0,
            0.0,
            0.0,
            "-25, 10",
            "0",
            2.0,
            5.0,
            0.30,
            0.20,
            0.65,
            1.10,
        ),
        "Golf Swing (active wrist)": (
            120.0,
            -90.0,
            0.0,
            0.0,
            "-25, 10",
            "-2, 3",
            2.0,
            5.0,
            0.30,
            0.20,
            0.65,
            1.10,
        ),
        "Heavy Clubhead": (
            120.0,
            -90.0,
            0.0,
            0.0,
            "-30, 12",
            "0",
            2.0,
            5.0,
            0.30,
            0.35,
            0.65,
            1.10,
        ),
        "Free Double Pendulum": (
            90.0,
            90.0,
            0.0,
            0.0,
            "0",
            "0",
            5.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
        ),
        "Straight Drop": (
            0.1,
            0.1,
            0.0,
            0.0,
            "0",
            "0",
            3.0,
            1.0,
            1.0,
            0.0,
            0.8,
            0.8,
        ),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(270)
        self.setMaximumWidth(370)
        self._build_ui()
        self._apply_preset("Golf Swing (passive wrist)")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble control panel from focused section builders."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        main_layout.addWidget(self._build_preset_section())
        main_layout.addWidget(self._build_physics_section())
        main_layout.addWidget(self._build_joint_limits_section())
        main_layout.addWidget(self._build_torque_clamp_section())
        main_layout.addWidget(self._build_ic_section())
        main_layout.addWidget(self._build_torque_section())
        main_layout.addWidget(self._build_sim_section())
        main_layout.addWidget(self._build_dissipation_section())
        main_layout.addWidget(self._build_rotation_section())
        self._build_hidden_compat_widgets()
        self._build_extra_hidden_widgets()
        main_layout.addWidget(self._build_export_section())
        main_layout.addStretch()

        # Wire torque preview
        self.inp_tau_shoulder.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tau_wrist.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tend.edit.textChanged.connect(self._update_torque_preview)
        self.chk_clamp.toggled.connect(lambda _: self._update_torque_preview())

    def _build_extra_hidden_widgets(self) -> None:
        """Extra hidden compat widgets specific to double pendulum."""
        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.chk_forces = QCheckBox()
        self.chk_forces.toggled.connect(self.forces_changed.emit)

    # ---- Model-specific section builders --------------------------------

    def _build_physics_section(self) -> QGroupBox:
        lw = 56
        box = QGroupBox("Arms & Shaft")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        if _HAS_UAI:
            self.inp_m1 = UnitAwareInput(
                category="mass",
                default_value=5.0,
                default_unit="kg",
                min_value=0,
                max_value=500,
                decimals=3,
                compact=True,
            )
            self.inp_m2 = UnitAwareInput(
                category="mass",
                default_value=0.30,
                default_unit="kg",
                min_value=0,
                max_value=500,
                decimals=3,
                compact=True,
            )
            self.inp_mClub = UnitAwareInput(
                category="mass",
                default_value=0.20,
                default_unit="kg",
                min_value=0,
                max_value=500,
                decimals=3,
                compact=True,
            )
            self.inp_L1 = UnitAwareInput(
                category="length",
                default_value=0.65,
                default_unit="m",
                min_value=0,
                max_value=100,
                decimals=4,
                compact=True,
            )
            self.inp_L2 = UnitAwareInput(
                category="length",
                default_value=1.10,
                default_unit="m",
                min_value=0,
                max_value=100,
                decimals=4,
                compact=True,
            )
            for lbl_text, widget in [
                ("m1", self.inp_m1),
                ("m2", self.inp_m2),
                ("mC", self.inp_mClub),
                ("L1", self.inp_L1),
                ("L2", self.inp_L2),
            ]:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(3)
                lbl = QLabel(lbl_text)
                lbl.setFixedWidth(24)
                lbl.setStyleSheet(STYLE_LABEL)
                row.addWidget(lbl)
                row.addWidget(widget)
                layout.addLayout(row)
        else:
            self.inp_m1 = LabeledInput("m1 (kg)", "5.0", "Arms mass (kg)", lw)  # type: ignore[assignment]
            self.inp_m2 = LabeledInput("m2 (kg)", "0.30", "Shaft mass (kg)", lw)  # type: ignore[assignment]
            self.inp_mClub = LabeledInput("mC (kg)", "0.20", "Clubhead mass (kg)", lw)  # type: ignore[assignment]
            self.inp_L1 = LabeledInput("L1 (m)", "0.65", "Arms length (m)", lw)  # type: ignore[assignment]
            self.inp_L2 = LabeledInput("L2 (m)", "1.10", "Shaft length (m)", lw)  # type: ignore[assignment]
            layout.addLayout(_row(self.inp_m1, self.inp_m2))
            layout.addLayout(_row(self.inp_L1, self.inp_L2))
            layout.addWidget(self.inp_mClub)
        return box

    def _build_joint_limits_section(self) -> QGroupBox:
        lw = 56
        box = QGroupBox("Joint Limits")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.chk_limits = QCheckBox("Enable joint limits")
        self.chk_limits.setStyleSheet(STYLE_CHECK)
        layout.addWidget(self.chk_limits)
        self.inp_theta1_min = LabeledInput(
            "θ1 min°", "-180", "Min shoulder angle (deg)", lw
        )
        self.inp_theta1_max = LabeledInput(
            "θ1 max°", "180", "Max shoulder angle (deg)", lw
        )
        layout.addLayout(_row(self.inp_theta1_min, self.inp_theta1_max))
        self.inp_phi_min = LabeledInput("φ min°", "-90", "Min wrist angle (deg)", lw)
        self.inp_phi_max = LabeledInput("φ max°", "90", "Max wrist angle (deg)", lw)
        layout.addLayout(_row(self.inp_phi_min, self.inp_phi_max))
        self.inp_limit_k = LabeledInput("K (N·m/rad)", "500", "Penalty stiffness", lw)
        layout.addWidget(self.inp_limit_k)
        return box

    def _build_torque_clamp_section(self) -> QGroupBox:
        lw = 56
        box = QGroupBox("Torque Saturation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.chk_clamp = QCheckBox("Enable torque clamping")
        self.chk_clamp.setStyleSheet(STYLE_CHECK)
        layout.addWidget(self.chk_clamp)
        if _HAS_UAI:
            self.inp_max_tau1 = UnitAwareInput(
                category="torque",
                default_value=50.0,
                default_unit="N\u00b7m",
                min_value=0,
                max_value=10000,
                decimals=1,
                compact=True,
            )
            self.inp_max_tau2 = UnitAwareInput(
                category="torque",
                default_value=20.0,
                default_unit="N\u00b7m",
                min_value=0,
                max_value=10000,
                decimals=1,
                compact=True,
            )
            for lbl_text, widget in [
                ("Max|\u03c41|", self.inp_max_tau1),
                ("Max|\u03c42|", self.inp_max_tau2),
            ]:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(3)
                lbl = QLabel(lbl_text)
                lbl.setFixedWidth(48)
                lbl.setStyleSheet(STYLE_LABEL)
                row.addWidget(lbl)
                row.addWidget(widget)
                layout.addLayout(row)
        else:
            self.inp_max_tau1 = LabeledInput(  # type: ignore[assignment]
                "Max |\u03c41|",
                "50",
                "Max shoulder torque magnitude ±(N·m)",
                lw,
            )
            self.inp_max_tau2 = LabeledInput(  # type: ignore[assignment]
                "Max |\u03c42|",
                "20",
                "Max wrist torque magnitude ±(N·m)",
                lw,
            )
            layout.addLayout(_row(self.inp_max_tau1, self.inp_max_tau2))
        return box

    def _build_ic_section(self) -> QGroupBox:
        lw = 56
        box = QGroupBox("Initial Conditions")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_theta1 = LabeledInput("θ1°", "120", "Arm angle from vertical", lw)
        self.inp_phi = LabeledInput("φ°", "-90", "Club angle relative to arm", lw)
        if _HAS_UAI:
            self.inp_dtheta1 = UnitAwareInput(
                category="angular_velocity",
                default_value=0.0,
                default_unit="rad/s",
                min_value=-1000,
                max_value=1000,
                decimals=3,
                compact=True,
            )
            self.inp_dphi = UnitAwareInput(
                category="angular_velocity",
                default_value=0.0,
                default_unit="rad/s",
                min_value=-1000,
                max_value=1000,
                decimals=3,
                compact=True,
            )
            layout.addLayout(_row(self.inp_theta1, self.inp_phi))
            for lbl_text, widget in [("dθ1", self.inp_dtheta1), ("dφ", self.inp_dphi)]:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(3)
                lbl = QLabel(lbl_text)
                lbl.setFixedWidth(24)
                lbl.setStyleSheet(STYLE_LABEL)
                row.addWidget(lbl)
                row.addWidget(widget)
                layout.addLayout(row)
        else:
            self.inp_dtheta1 = LabeledInput(
                "dθ1", "0", "Arm angular velocity rad/s", lw
            )  # type: ignore[assignment]
            self.inp_dphi = LabeledInput("dφ", "0", "Club angular velocity rad/s", lw)  # type: ignore[assignment]
            layout.addLayout(_row(self.inp_theta1, self.inp_phi))
            layout.addLayout(_row(self.inp_dtheta1, self.inp_dphi))
        return box

    def _build_torque_section(self) -> QGroupBox:
        box = QGroupBox("Torque Polynomials  (c0, c1, c2 …)")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_tau_shoulder = LabeledInput(
            "Shoulder", "-25, 10", "τ(t)=c0+c1·t+…", 56
        )
        self.inp_tau_wrist = LabeledInput("Wrist", "0", "τ(t)=c0+c1·t+…", 56)
        layout.addWidget(self.inp_tau_shoulder)
        layout.addWidget(self.inp_tau_wrist)
        self.btn_funcgen = self._build_funcgen_button()
        layout.addWidget(self.btn_funcgen)
        preview_box = QGroupBox("Torque Preview")
        preview_box.setStyleSheet(STYLE_GROUP)
        pv_layout = QVBoxLayout(preview_box)
        pv_layout.setContentsMargins(4, 12, 4, 4)
        self.torque_preview = TorquePreviewWidget()
        pv_layout.addWidget(self.torque_preview)
        layout.addWidget(preview_box)
        return box

    def _build_sim_section(self) -> QGroupBox:
        lw = 56
        box = QGroupBox("Simulation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_tend = LabeledInput("Duration s", "2.0", "Total simulation time", lw)
        self.inp_dt = LabeledInput(
            "dt (s)",
            "0.005",
            "Step size.\nSmaller = more accurate, slower.\nTypical: 0.001 – 0.02",
            lw,
        )
        layout.addLayout(_row(self.inp_tend, self.inp_dt))
        return box

    def _build_dissipation_section(self) -> QGroupBox:
        lw = 56
        box = QGroupBox("Dissipation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_b1 = LabeledInput("b1", "0.0", "Viscous damping shoulder (N·m·s)", lw)
        self.inp_b2 = LabeledInput("b2", "0.0", "Viscous damping wrist (N·m·s)", lw)
        self.inp_mu1 = LabeledInput("μ1", "0.0", "Coulomb friction shoulder (N·m)", lw)
        self.inp_mu2 = LabeledInput("μ2", "0.0", "Coulomb friction wrist (N·m)", lw)
        layout.addLayout(_row(self.inp_b1, self.inp_b2))
        layout.addLayout(_row(self.inp_mu1, self.inp_mu2))
        return box

    def _build_rotation_section(self) -> QGroupBox:
        """Swing plane tilt — gravity is always on (#1209)."""
        box = QGroupBox("Physics")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(4)

        self.inp_tilt = LabeledInput(
            "Tilt °",
            "0",
            "Swing plane tilt from vertical (0°=vertical, 90°=horizontal).\n"
            "Effective gravity = g·cos(tilt). A typical golfer's\n"
            "swing plane is tilted ~30–60° from vertical.",
            56,
        )
        layout.addWidget(self.inp_tilt)

        self.inp_azimuth = LabeledInput(
            "Azimuth °",
            "0",
            "View rotation around vertical axis (0°=front, 90°=side).\n"
            "Rotate the canvas to see the tilted swing plane\n"
            "from different angles.",
            56,
        )
        layout.addWidget(self.inp_azimuth)

        self.inp_tilt.edit.textChanged.connect(self._on_tilt_edited)
        self.inp_azimuth.edit.textChanged.connect(self._on_azimuth_edited)
        return box

    def _on_tilt_edited(self, text: str) -> None:
        """Emit tilt_changed in real-time when Tilt input is edited."""
        with contextlib.suppress(ValueError):
            self.tilt_changed.emit(np.radians(float(text)))

    def _on_azimuth_edited(self, text: str) -> None:
        """Emit azimuth_changed in real-time when Azimuth input is edited."""
        with contextlib.suppress(ValueError):
            self.azimuth_changed.emit(np.radians(float(text)))

    # ── Override base set_slider_range/value to also update lbl_frame ─
    def set_slider_range(self, max_val: int) -> None:
        super().set_slider_range(max_val)
        self.lbl_frame.setText(f"Frame: 0 / {max_val}")

    def set_slider_value(self, val: int) -> None:
        super().set_slider_value(val)
        self.lbl_frame.setText(f"Frame: {val} / {self.slider.maximum()}")

    # ── Abstract interface implementation ────────────────────────────

    def _get_joint_names(self) -> list[str]:
        return ["Shoulder", "Wrist"]

    def _get_torque_inputs(self) -> dict[str, LabeledInput]:
        return {
            "Shoulder": self.inp_tau_shoulder,
            "Wrist": self.inp_tau_wrist,
        }

    def _apply_preset(self, name: str) -> None:
        if not (name is not None):
            raise ValueError("name must be provided")
        if name not in self.PRESETS:
            return
        theta1, phi, dth, dph, tau_sh, tau_wr, tend, m1, m2, mClub, L1, L2 = (
            self.PRESETS[name]
        )
        self.inp_theta1.set_value(str(theta1))
        self.inp_phi.set_value(str(phi))
        self.inp_tau_shoulder.set_value(tau_sh)
        self.inp_tau_wrist.set_value(tau_wr)
        self.inp_tend.set_value(str(tend))

        if _HAS_UAI:
            self.inp_m1.set_value(m1, is_si=True)
            self.inp_m2.set_value(m2, is_si=True)
            self.inp_mClub.set_value(mClub, is_si=True)
            self.inp_L1.set_value(L1, is_si=True)
            self.inp_L2.set_value(L2, is_si=True)
            self.inp_dtheta1.set_value(dth, is_si=True)
            self.inp_dphi.set_value(dph, is_si=True)
        else:
            self.inp_dtheta1.set_value(str(dth))  # type: ignore[arg-type]
            self.inp_dphi.set_value(str(dph))  # type: ignore[arg-type]
            self.inp_m1.set_value(str(m1))  # type: ignore[arg-type]
            self.inp_m2.set_value(str(m2))  # type: ignore[arg-type]
            self.inp_mClub.set_value(str(mClub))  # type: ignore[arg-type]
            self.inp_L1.set_value(str(L1))  # type: ignore[arg-type]
            self.inp_L2.set_value(str(L2))  # type: ignore[arg-type]
        self._update_torque_preview()

    # ------------------------------------------------------------------
    # Value parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _uai_or_parse(widget: object, label: str) -> float:
        """Extract SI value from UnitAwareInput or parse from LabeledInput."""
        if not (widget is not None):
            raise ValueError("widget must be provided")
        if _HAS_UAI and isinstance(widget, UnitAwareInput):
            return widget.value_si()
        return parse_float(widget, label)  # type: ignore[arg-type]

    def get_params(self) -> dict:
        """Parse all input fields and return as a simulation parameter dict."""
        dt_raw = parse_float(self.inp_dt, "dt")
        dt = clamp_dt(dt_raw)
        m1 = self._uai_or_parse(self.inp_m1, "m1")
        m2 = self._uai_or_parse(self.inp_m2, "m2")
        mClub = self._uai_or_parse(self.inp_mClub, "mClub")
        L1 = self._uai_or_parse(self.inp_L1, "L1")
        L2 = self._uai_or_parse(self.inp_L2, "L2")
        b1 = require_non_negative(parse_float(self.inp_b1, "b1"), "b1")
        b2 = require_non_negative(parse_float(self.inp_b2, "b2"), "b2")
        mu1 = require_non_negative(parse_float(self.inp_mu1, "μ1"), "μ1")
        mu2 = require_non_negative(parse_float(self.inp_mu2, "μ2"), "μ2")
        require_positive(m1, "m1")
        require_positive(m2, "m2")
        require_non_negative(mClub, "mClub")
        require_positive(L1, "L1")
        require_positive(L2, "L2")

        return {
            "m1": m1,
            "m2": m2,
            "mClub": mClub,
            "L1": L1,
            "L2": L2,
            "theta1_rad": np.radians(parse_float(self.inp_theta1, "θ1")),
            "phi_rad": np.radians(parse_float(self.inp_phi, "φ")),
            "dtheta1": self._uai_or_parse(self.inp_dtheta1, "dθ1"),
            "dphi": self._uai_or_parse(self.inp_dphi, "dφ"),
            "shoulder_coeffs": parse_coeffs(self.inp_tau_shoulder, "Shoulder torque"),
            "wrist_coeffs": parse_coeffs(self.inp_tau_wrist, "Wrist torque"),
            "t_end": parse_float(self.inp_tend, "Duration"),
            "dt": dt,
            "b1": b1,
            "b2": b2,
            "mu1": mu1,
            "mu2": mu2,
            "gravity_on": True,  # Gravity always on (#1209)
            "enable_limits": self.chk_limits.isChecked(),
            "theta1_min_rad": np.radians(parse_float(self.inp_theta1_min, "θ1 min")),
            "theta1_max_rad": np.radians(parse_float(self.inp_theta1_max, "θ1 max")),
            "phi_min_rad": np.radians(parse_float(self.inp_phi_min, "φ min")),
            "phi_max_rad": np.radians(parse_float(self.inp_phi_max, "φ max")),
            "limit_stiffness": parse_float(self.inp_limit_k, "Limit K"),
            "enable_clamp": self.chk_clamp.isChecked(),
            "max_torque1": self._uai_or_parse(self.inp_max_tau1, "Max τ1"),
            "max_torque2": self._uai_or_parse(self.inp_max_tau2, "Max τ2"),
            "tilt_deg": parse_float(self.inp_tilt, "Tilt"),
            "azimuth_deg": parse_float(self.inp_azimuth, "Azimuth"),
        }

    def _update_torque_preview(self) -> None:
        try:
            t_end = float(self.inp_tend.value)
        except ValueError:
            t_end = 2.0
        self.torque_preview.set_duration(t_end)

        clamp_limits: list[float | None] | None = None
        if self.chk_clamp.isChecked():
            try:
                tau1 = self._uai_or_parse(self.inp_max_tau1, "Max τ1")
            except (ValueError, AttributeError):
                tau1 = float("inf")
            try:
                tau2 = self._uai_or_parse(self.inp_max_tau2, "Max τ2")
            except (ValueError, AttributeError):
                tau2 = float("inf")
            clamp_limits = [tau1, tau2]

        self.torque_preview.set_profiles(
            [
                (
                    "Shoulder",
                    parse_coeffs_lenient(self.inp_tau_shoulder),
                    QColor(230, 120, 50),
                ),
                (
                    "Wrist",
                    parse_coeffs_lenient(self.inp_tau_wrist),
                    QColor(120, 180, 230),
                ),
            ],
            clamp_limits=clamp_limits,
        )

    # ------------------------------------------------------------------
    # Playback — override base to use styled text
    # ------------------------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        if not (checked is not None):
            raise ValueError("checked must be provided")
        self._is_playing = checked
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        self.play_toggled.emit(checked)
