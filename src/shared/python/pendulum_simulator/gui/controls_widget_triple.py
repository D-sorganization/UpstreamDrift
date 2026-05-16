"""
Control panel widget for triple pendulum inputs.

Refactored to extend ControlsWidgetBase (DRY).
Only model-specific sections remain here.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from .controls_utils import (
    STYLE_GROUP,
    parse_coeffs,
    parse_coeffs_lenient,
    parse_float,
    require_non_negative,
    require_positive,
)
from .controls_widget import LabeledInput
from .controls_widget_base import ControlsWidgetBase
from .torque_preview_widget import TorquePreviewWidget

try:
    from sidekick.ui.widgets.unit_aware_input import UnitAwareInput

    _HAS_UAI = True
except ImportError:
    _HAS_UAI = False


class ControlsWidgetTriple(ControlsWidgetBase):
    """Parameter input panel for the triple pendulum."""

    PRESETS = {
        "Triple Swing": (
            120.0,
            -60.0,
            -30.0,
            0.0,
            0.0,
            0.0,
            "-25, 10",
            "0",
            "0",
            2.0,
            5.0,
            0.5,
            0.4,
            0.20,  # L1: Hub (sternum → shoulder)
            0.65,  # L2: Arm
            1.10,  # L3: Club
        ),
        "Free Triple Pendulum": (
            90.0,
            60.0,
            -45.0,
            0.0,
            0.0,
            0.0,
            "0",
            "0",
            "0",
            5.0,
            1.0,
            0.5,
            0.2,
            0.50,  # L1
            0.50,  # L2
            0.50,  # L3
        ),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(360)
        self._build_ui()
        self._apply_preset("Triple Swing")

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        main_layout.addWidget(self._build_preset_section())
        main_layout.addWidget(self._build_physics_section())
        main_layout.addWidget(self._build_ic_section())
        main_layout.addWidget(self._build_torque_section())
        main_layout.addWidget(self._build_torque_preview_section())
        main_layout.addWidget(
            self._build_torque_clamp_section_ndof(
                ["Shoulder", "Elbow", "Wrist"],
                [50.0, 30.0, 15.0],
            )
        )
        main_layout.addWidget(
            self._build_joint_limits_section_ndof(
                ["Shoulder", "Elbow", "Wrist"],
                [-180.0, -150.0, -90.0],
                [180.0, 150.0, 90.0],
            )
        )
        main_layout.addWidget(self._build_sim_section())
        main_layout.addWidget(self._build_dissipation_section())
        main_layout.addLayout(self._build_run_reset_buttons())
        self._build_hidden_compat_widgets()
        main_layout.addWidget(self._build_export_section())
        main_layout.addWidget(self._build_gravity_section())
        main_layout.addStretch()

        # Wire torque preview
        self.inp_tau_shoulder.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tau_elbow.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tau_wrist.edit.textChanged.connect(self._update_torque_preview)
        self.inp_tend.edit.textChanged.connect(self._update_torque_preview)

    # ── Model-specific section builders ──────────────────────────────

    def _build_physics_section(self) -> QGroupBox:
        from PyQt6.QtWidgets import QHBoxLayout, QLabel

        from .controls_utils import STYLE_LABEL

        box = QGroupBox("Physical Parameters")
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
                default_value=0.5,
                default_unit="kg",
                min_value=0,
                max_value=500,
                decimals=3,
                compact=True,
            )
            self.inp_m3 = UnitAwareInput(
                category="mass",
                default_value=0.4,
                default_unit="kg",
                min_value=0,
                max_value=500,
                decimals=3,
                compact=True,
            )
            self.inp_L1 = UnitAwareInput(
                category="length",
                default_value=0.20,
                default_unit="m",
                min_value=0,
                max_value=100,
                decimals=4,
                compact=True,
            )
            self.inp_L2 = UnitAwareInput(
                category="length",
                default_value=0.65,
                default_unit="m",
                min_value=0,
                max_value=100,
                decimals=4,
                compact=True,
            )
            self.inp_L3 = UnitAwareInput(
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
                ("m3", self.inp_m3),
                ("L1 Hub", self.inp_L1),
                ("L2 Arm", self.inp_L2),
                ("L3 Club", self.inp_L3),
            ]:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(3)
                lbl = QLabel(lbl_text)
                lbl.setFixedWidth(50)
                lbl.setStyleSheet(STYLE_LABEL)
                row.addWidget(lbl)
                row.addWidget(widget)
                layout.addLayout(row)
        else:
            self.inp_m1 = LabeledInput("m1 (kg)", "5.0", "Mass of segment 1")  # type: ignore[assignment]
            self.inp_m2 = LabeledInput("m2 (kg)", "0.5", "Mass of segment 2")  # type: ignore[assignment]
            self.inp_m3 = LabeledInput("m3 (kg)", "0.4", "Mass of segment 3")  # type: ignore[assignment]
            self.inp_L1 = LabeledInput(  # type: ignore[assignment]
                "L1 (m) — Hub", "0.20", "Length of segment 1: Hub (sternum → shoulder)"
            )
            self.inp_L2 = LabeledInput(  # type: ignore[assignment]
                "L2 (m) — Arm", "0.65", "Length of segment 2: Arm"
            )
            self.inp_L3 = LabeledInput(  # type: ignore[assignment]
                "L3 (m) — Club", "1.10", "Length of segment 3: Club"
            )
            for w in [  # type: ignore[assignment]
                self.inp_m1,
                self.inp_m2,
                self.inp_m3,
                self.inp_L1,
                self.inp_L2,
                self.inp_L3,
            ]:
                layout.addWidget(w)

        self.inp_scapula = LabeledInput(
            "Scapula °",
            "0",
            "Scapula protraction/retraction offset angle (#1152).\n"
            "0° = neutral, positive = protracted (forward).",
        )
        layout.addWidget(self.inp_scapula)
        return box

    def _build_ic_section(self) -> QGroupBox:
        box = QGroupBox("Initial Conditions")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_theta1 = LabeledInput("theta1 (deg)", "120", "Segment 1 angle")
        self.inp_phi1 = LabeledInput("phi1 (deg)", "-60", "Segment 2 relative angle")
        self.inp_phi2 = LabeledInput("phi2 (deg)", "-30", "Segment 3 relative angle")
        self.inp_dtheta1 = LabeledInput("dtheta1 (rad/s)", "0", "Segment 1 velocity")
        self.inp_dphi1 = LabeledInput("dphi1 (rad/s)", "0", "Segment 2 velocity")
        self.inp_dphi2 = LabeledInput("dphi2 (rad/s)", "0", "Segment 3 velocity")
        for w in [
            self.inp_theta1,
            self.inp_phi1,
            self.inp_phi2,
            self.inp_dtheta1,
            self.inp_dphi1,
            self.inp_dphi2,
        ]:
            layout.addWidget(w)
        return box

    def _build_torque_section(self) -> QGroupBox:
        box = QGroupBox("Torque Polynomials (c0, c1, c2, ...)")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_tau_shoulder = LabeledInput(
            "Shoulder", "-25, 10", "τ(t) = c0 + c1*t + c2*t^2 + ..."
        )
        self.inp_tau_elbow = LabeledInput(
            "Elbow", "0", "τ(t) = c0 + c1*t + c2*t^2 + ..."
        )
        self.inp_tau_wrist = LabeledInput(
            "Wrist", "0", "τ(t) = c0 + c1*t + c2*t^2 + ..."
        )
        layout.addWidget(self.inp_tau_shoulder)
        layout.addWidget(self.inp_tau_elbow)
        layout.addWidget(self.inp_tau_wrist)
        self.btn_funcgen = self._build_funcgen_button()
        layout.addWidget(self.btn_funcgen)
        return box

    def _build_torque_preview_section(self) -> QGroupBox:
        box = QGroupBox("Torque Preview")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        self.torque_preview = TorquePreviewWidget()
        layout.addWidget(self.torque_preview)
        return box

    def _build_sim_section(self) -> QGroupBox:
        box = QGroupBox("Simulation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        self.inp_tend = LabeledInput("Duration (s)", "2.0", "Total simulation time")
        layout.addWidget(self.inp_tend)
        return box

    def _build_dissipation_section(self) -> QGroupBox:
        box = QGroupBox("Dissipation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_b1 = LabeledInput("b1", "0.0", "Viscous damping shoulder (N·m·s)")
        self.inp_b2 = LabeledInput("b2", "0.0", "Viscous damping elbow (N·m·s)")
        self.inp_b3 = LabeledInput("b3", "0.0", "Viscous damping wrist (N·m·s)")
        self.inp_mu1 = LabeledInput("μ1", "0.0", "Coulomb friction shoulder (N·m)")
        self.inp_mu2 = LabeledInput("μ2", "0.0", "Coulomb friction elbow (N·m)")
        self.inp_mu3 = LabeledInput("μ3", "0.0", "Coulomb friction wrist (N·m)")
        for w in [
            self.inp_b1,
            self.inp_b2,
            self.inp_b3,
            self.inp_mu1,
            self.inp_mu2,
            self.inp_mu3,
        ]:
            layout.addWidget(w)
        return box

    # ── Abstract interface implementation ────────────────────────────

    def _get_joint_names(self) -> list[str]:
        return ["Shoulder", "Elbow", "Wrist"]

    def _get_torque_inputs(self) -> dict[str, LabeledInput]:
        return {
            "Shoulder": self.inp_tau_shoulder,
            "Elbow": self.inp_tau_elbow,
            "Wrist": self.inp_tau_wrist,
        }

    def _apply_preset(self, name: str) -> None:
        if not (name is not None):
            raise ValueError("name must be provided")
        if name not in self.PRESETS:
            return
        (
            theta1,
            phi1,
            phi2,
            dth,
            dph1,
            dph2,
            tau_sh,
            tau_el,
            tau_wr,
            tend,
            m1,
            m2,
            m3,
            L1,
            L2,
            L3,
        ) = self.PRESETS[name]

        self.inp_theta1.set_value(str(theta1))
        self.inp_phi1.set_value(str(phi1))
        self.inp_phi2.set_value(str(phi2))
        self.inp_dtheta1.set_value(str(dth))
        self.inp_dphi1.set_value(str(dph1))
        self.inp_dphi2.set_value(str(dph2))
        self.inp_tau_shoulder.set_value(tau_sh)
        self.inp_tau_elbow.set_value(tau_el)
        self.inp_tau_wrist.set_value(tau_wr)
        self.inp_tend.set_value(str(tend))
        # Use float API for UnitAwareInput, fall back to str for LabeledInput
        for widget, val in [
            (self.inp_m1, m1),
            (self.inp_m2, m2),
            (self.inp_m3, m3),
            (self.inp_L1, L1),
            (self.inp_L2, L2),
            (self.inp_L3, L3),
        ]:
            try:
                widget.set_value(val, is_si=True)
            except TypeError:
                widget.set_value(str(val))  # type: ignore[arg-type]
        self._update_torque_preview()

    def get_params(self) -> dict:
        """Parse all inputs and return a simulation parameter dict.

        Raises ValueError if any field cannot be parsed or violates contracts.
        """
        m1 = self._uai_or_parse(self.inp_m1, "m1")
        m2 = self._uai_or_parse(self.inp_m2, "m2")
        m3 = self._uai_or_parse(self.inp_m3, "m3")
        L1 = self._uai_or_parse(self.inp_L1, "L1")
        L2 = self._uai_or_parse(self.inp_L2, "L2")
        L3 = self._uai_or_parse(self.inp_L3, "L3")
        b1 = require_non_negative(parse_float(self.inp_b1, "b1"), "b1")
        b2 = require_non_negative(parse_float(self.inp_b2, "b2"), "b2")
        b3 = require_non_negative(parse_float(self.inp_b3, "b3"), "b3")
        mu1 = require_non_negative(parse_float(self.inp_mu1, "μ1"), "μ1")
        mu2 = require_non_negative(parse_float(self.inp_mu2, "μ2"), "μ2")
        mu3 = require_non_negative(parse_float(self.inp_mu3, "μ3"), "μ3")
        require_positive(m1, "m1")
        require_positive(m2, "m2")
        require_positive(m3, "m3")
        require_positive(L1, "L1")
        require_positive(L2, "L2")
        require_positive(L3, "L3")

        # Torque clamp / joint limits (N-DOF base class helpers)
        torque_lims = self._parse_torque_limits()
        joint_lims = self._parse_joint_limits()

        result = {
            "m1": m1,
            "m2": m2,
            "m3": m3,
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "theta1_rad": np.radians(parse_float(self.inp_theta1, "theta1")),
            "phi1_rad": np.radians(parse_float(self.inp_phi1, "phi1")),
            "phi2_rad": np.radians(parse_float(self.inp_phi2, "phi2")),
            "dtheta1": parse_float(self.inp_dtheta1, "dtheta1"),
            "dphi1": parse_float(self.inp_dphi1, "dphi1"),
            "dphi2": parse_float(self.inp_dphi2, "dphi2"),
            "shoulder_coeffs": parse_coeffs(self.inp_tau_shoulder, "Shoulder torque"),
            "elbow_coeffs": parse_coeffs(self.inp_tau_elbow, "Elbow torque"),
            "wrist_coeffs": parse_coeffs(self.inp_tau_wrist, "Wrist torque"),
            "t_end": parse_float(self.inp_tend, "Duration"),
            "gravity_on": True,  # Gravity always on (#1209)
            "b1": b1,
            "b2": b2,
            "b3": b3,
            "mu1": mu1,
            "mu2": mu2,
            "mu3": mu3,
            "scapula_deg": parse_float(self.inp_scapula, "Scapula"),
        }

        # Torque saturation — flat list → np.ndarray for physics
        if torque_lims is not None:
            result["enable_clamp"] = True
            result["torque_limits"] = torque_lims  # list[float], len 3
        else:
            result["enable_clamp"] = False

        # Joint angle limits — (mins_rad, maxs_rad, stiffness)
        if joint_lims is not None:
            mins_rad, maxs_rad, stiffness = joint_lims
            result["enable_limits"] = True
            result["limit_mins_rad"] = mins_rad
            result["limit_maxs_rad"] = maxs_rad
            result["limit_stiffness"] = stiffness
        else:
            result["enable_limits"] = False

        return result

    def _update_torque_preview(self) -> None:
        try:
            t_end = float(self.inp_tend.value)
        except ValueError:
            t_end = 2.0

        self.torque_preview.set_duration(t_end)
        self.torque_preview.set_profiles(
            [
                (
                    "Shoulder",
                    parse_coeffs_lenient(self.inp_tau_shoulder),
                    QColor(230, 120, 50),
                ),
                (
                    "Elbow",
                    parse_coeffs_lenient(self.inp_tau_elbow),
                    QColor(120, 200, 140),
                ),
                (
                    "Wrist",
                    parse_coeffs_lenient(self.inp_tau_wrist),
                    QColor(120, 180, 230),
                ),
            ]
        )
