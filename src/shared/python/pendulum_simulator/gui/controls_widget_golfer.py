# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Control panel widget for the golfer upper-body model.

Refactored to extend ControlsWidgetBase (DRY).
Only model-specific sections (8-segment params, 7 joints) remain here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .controls_utils import (
    STYLE_GROUP,
    parse_coeffs,
    parse_float,
    require_non_negative,
    require_positive,
)
from .controls_widget import LabeledInput
from .controls_widget_base import ControlsWidgetBase

try:
    from sidekick.ui.widgets.unit_aware_input import UnitAwareInput

    _HAS_UAI = True
except ImportError:
    _HAS_UAI = False


class ControlsWidgetGolfer(ControlsWidgetBase):
    """Parameter input panel for the golfer upper-body model."""

    PRESETS = {
        "Address Position": {
            "m_hub": 0.001,  # Standoff is massless (near-zero for numerics)
            "m_r_upper": 3.5,
            "m_r_fore": 2.0,
            "m_l_upper": 3.5,
            "m_l_fore": 2.0,
            "m_club": 0.5,
            "L_hub": 0.15,
            "L_r_upper": 0.35,
            "L_r_fore": 0.30,
            "L_l_upper": 0.35,
            "L_l_fore": 0.30,
            "L_club": 1.1,
            "d_rs": 0.20,
            "d_ls": 0.20,
            "grip_right": 0.05,
            "grip_left": 0.25,
            "m_clubhead": 0.2,
            "theta_hub": 0.0,
            "alpha_rs": 0.0,
            "alpha_re": 0.0,
            "alpha_rh": 0.0,
            "alpha_ls": 0.0,
            "alpha_le": 0.0,
            "alpha_lh": 0.0,
            "tau_hub": "0",
            "tau_rs": "0",
            "tau_re": "0",
            "tau_rh": "0",
            "tau_ls": "0",
            "tau_le": "0",
            "tau_lh": "0",
            "t_end": 2.0,
            "L_rscap": 0.18,
            "L_lscap": 0.18,
            "m_rscap": 7.0,  # Upper body segment (~2× arm mass)
            "m_lscap": 7.0,  # Upper body segment (~2× arm mass)
        },
        "Backswing Start": {
            "m_hub": 0.001,  # Standoff is massless
            "m_r_upper": 3.5,
            "m_r_fore": 2.0,
            "m_l_upper": 3.5,
            "m_l_fore": 2.0,
            "m_club": 0.5,
            "L_hub": 0.15,
            "L_r_upper": 0.35,
            "L_r_fore": 0.30,
            "L_l_upper": 0.35,
            "L_l_fore": 0.30,
            "L_club": 1.1,
            "d_rs": 0.20,
            "d_ls": 0.20,
            "grip_right": 0.05,
            "grip_left": 0.25,
            "m_clubhead": 0.2,
            "theta_hub": 15.0,
            "alpha_rs": -10.0,
            "alpha_re": 20.0,
            "alpha_rh": -5.0,
            "alpha_ls": 10.0,
            "alpha_le": -15.0,
            "alpha_lh": 5.0,
            "tau_hub": "-5, 2",
            "tau_rs": "-10, 5",
            "tau_re": "0",
            "tau_rh": "0",
            "tau_ls": "10, -5",
            "tau_le": "0",
            "tau_lh": "0",
            "t_end": 2.0,
            "L_rscap": 0.18,
            "L_lscap": 0.18,
            "m_rscap": 7.0,  # Upper body segment
            "m_lscap": 7.0,  # Upper body segment
        },
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setMaximumWidth(400)
        self._build_ui()
        self._apply_preset("Address Position")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        main = QVBoxLayout(container)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        main.addWidget(self._build_preset_section())
        main.addWidget(self._build_mass_section())
        main.addWidget(self._build_length_section())
        main.addWidget(self._build_geometry_section())
        main.addWidget(self._build_ic_section())
        main.addWidget(self._build_torque_section())
        _golfer_joints = [
            "Hub",
            "R Shoulder",
            "R Elbow",
            "R Wrist",
            "L Shoulder",
            "L Elbow",
            "L Wrist",
        ]
        main.addWidget(
            self._build_torque_clamp_section_ndof(
                _golfer_joints,
                [100.0, 80.0, 50.0, 30.0, 80.0, 50.0, 30.0],
            )
        )
        main.addWidget(
            self._build_joint_limits_section_ndof(
                _golfer_joints,
                [-90.0, -180.0, -150.0, -90.0, -180.0, -150.0, -90.0],
                [90.0, 180.0, 150.0, 90.0, 180.0, 150.0, 90.0],
            )
        )
        main.addWidget(self._build_sim_section())
        main.addWidget(self._build_dissipation_section())
        main.addLayout(self._build_run_reset_buttons())
        self._build_hidden_compat_widgets()
        main.addWidget(self._build_export_section())
        main.addWidget(self._build_gravity_section())
        main.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ── Model-specific section builders ──────────────────────────────

    def _build_mass_section(self) -> QGroupBox:
        from PyQt6.QtWidgets import QHBoxLayout, QLabel

        from .controls_utils import STYLE_LABEL

        box = QGroupBox("Segment Masses")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)

        mass_specs = [
            ("Stoff", "m_hub", 0.001),  # Standoff (massless)
            ("R Upr", "m_r_upper", 3.5),
            ("R Fore", "m_r_fore", 2.0),
            ("L Upr", "m_l_upper", 3.5),
            ("L Fore", "m_l_fore", 2.0),
            ("Club", "m_club", 0.5),
            ("Head", "m_clubhead", 0.2),
        ]
        if _HAS_UAI:
            for lbl_text, attr, default in mass_specs:
                w = UnitAwareInput(
                    category="mass",
                    default_value=default,
                    default_unit="kg",
                    min_value=0,
                    max_value=500,
                    decimals=3,
                    compact=True,
                )
                setattr(self, f"inp_{attr}", w)
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(3)
                lbl = QLabel(lbl_text)
                lbl.setFixedWidth(44)
                lbl.setStyleSheet(STYLE_LABEL)
                row.addWidget(lbl)
                row.addWidget(w)
                layout.addLayout(row)
        else:
            self.inp_m_hub = LabeledInput(
                "Standoff", "0.001", "Standoff mass (massless)"
            )  # type: ignore[assignment]
            self.inp_m_r_upper = LabeledInput("R Upper", "3.5", "Right upper arm")  # type: ignore[assignment]
            self.inp_m_r_fore = LabeledInput("R Fore", "2.0", "Right forearm")  # type: ignore[assignment]
            self.inp_m_l_upper = LabeledInput("L Upper", "3.5", "Left upper arm")  # type: ignore[assignment]
            self.inp_m_l_fore = LabeledInput("L Fore", "2.0", "Left forearm")  # type: ignore[assignment]
            self.inp_m_club = LabeledInput("Club", "0.5", "Club shaft mass")  # type: ignore[assignment]
            self.inp_m_clubhead = LabeledInput("Clubhead", "0.2", "Clubhead point mass")  # type: ignore[assignment]
            for w in [  # type: ignore[assignment]
                self.inp_m_hub,
                self.inp_m_r_upper,
                self.inp_m_r_fore,
                self.inp_m_l_upper,
                self.inp_m_l_fore,
                self.inp_m_club,
                self.inp_m_clubhead,
            ]:
                layout.addWidget(w)
        return box

    def _build_length_section(self) -> QGroupBox:
        from PyQt6.QtWidgets import QHBoxLayout, QLabel

        from .controls_utils import STYLE_LABEL

        box = QGroupBox("Segment Lengths")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)

        length_specs = [
            ("Stoff", "L_hub", 0.15),  # Standoff length
            ("R Upr", "L_r_upper", 0.35),
            ("R Fore", "L_r_fore", 0.30),
            ("L Upr", "L_l_upper", 0.35),
            ("L Fore", "L_l_fore", 0.30),
            ("Club", "L_club", 1.10),
        ]
        if _HAS_UAI:
            for lbl_text, attr, default in length_specs:
                w = UnitAwareInput(
                    category="length",
                    default_value=default,
                    default_unit="m",
                    min_value=0,
                    max_value=100,
                    decimals=4,
                    compact=True,
                )
                setattr(self, f"inp_{attr}", w)
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(3)
                lbl = QLabel(lbl_text)
                lbl.setFixedWidth(44)
                lbl.setStyleSheet(STYLE_LABEL)
                row.addWidget(lbl)
                row.addWidget(w)
                layout.addLayout(row)
        else:
            self.inp_L_hub = LabeledInput(
                "Standoff", "0.15", "Standoff length (COM offset)"
            )  # type: ignore[assignment]
            self.inp_L_r_upper = LabeledInput("R Upper", "0.35", "Right upper arm")  # type: ignore[assignment]
            self.inp_L_r_fore = LabeledInput("R Fore", "0.30", "Right forearm")  # type: ignore[assignment]
            self.inp_L_l_upper = LabeledInput("L Upper", "0.35", "Left upper arm")  # type: ignore[assignment]
            self.inp_L_l_fore = LabeledInput("L Fore", "0.30", "Left forearm")  # type: ignore[assignment]
            self.inp_L_club = LabeledInput("Club", "1.1", "Club total length")  # type: ignore[assignment]
            for w in [  # type: ignore[assignment]
                self.inp_L_hub,
                self.inp_L_r_upper,
                self.inp_L_r_fore,
                self.inp_L_l_upper,
                self.inp_L_l_fore,
                self.inp_L_club,
            ]:
                layout.addWidget(w)
        return box

    def _build_geometry_section(self) -> QGroupBox:
        box = QGroupBox("Geometry")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_d_rs = LabeledInput(
            "d_RS (m)", "0.20", "Hub bar to right shoulder offset"
        )
        self.inp_d_ls = LabeledInput(
            "d_LS (m)", "0.20", "Hub bar to left shoulder offset"
        )
        self.inp_grip_right = LabeledInput(
            "Grip R (m)", "0.05", "Right hand grip from club base"
        )
        self.inp_grip_left = LabeledInput(
            "Grip L (m)", "0.25", "Left hand grip from club base"
        )
        self.inp_L_rscap = LabeledInput(
            "R UBody (m)",
            "0.18",
            "Right upper body segment length.\n"
            "Connects hub to right shoulder via revolute joint.\n"
            "Represents the upper torso on the right side.\n"
            "Set to 0 to disable.",
        )
        self.inp_L_lscap = LabeledInput(
            "L UBody (m)",
            "0.18",
            "Left upper body segment length.\n"
            "Connects hub to left shoulder via revolute joint.\n"
            "Represents the upper torso on the left side.\n"
            "Set to 0 to disable.",
        )
        self.inp_m_rscap = LabeledInput(
            "R UBody m",
            "7.0",
            "Right upper body mass (kg).\nShould be ~2× arm mass to represent torso.",
        )
        self.inp_m_lscap = LabeledInput(
            "L UBody m",
            "7.0",
            "Left upper body mass (kg).\nShould be ~2× arm mass to represent torso.",
        )
        for w in [
            self.inp_d_rs,
            self.inp_d_ls,
            self.inp_grip_right,
            self.inp_grip_left,
            self.inp_L_rscap,
            self.inp_L_lscap,
            self.inp_m_rscap,
            self.inp_m_lscap,
        ]:
            layout.addWidget(w)
        return box

    def _build_ic_section(self) -> QGroupBox:
        box = QGroupBox("Initial Conditions (deg)")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_th_hub = LabeledInput("Hub", "0", "Hub angle")
        self.inp_a_rs = LabeledInput("R Shoulder", "0", "Relative angle")
        self.inp_a_re = LabeledInput("R Elbow", "0", "Relative angle")
        self.inp_a_rh = LabeledInput("R Wrist", "0", "Relative angle")
        self.inp_a_ls = LabeledInput("L Shoulder", "0", "Relative angle")
        self.inp_a_le = LabeledInput("L Elbow", "0", "Relative angle")
        self.inp_a_lh = LabeledInput("L Wrist", "0", "Relative angle")
        for w in [
            self.inp_th_hub,
            self.inp_a_rs,
            self.inp_a_re,
            self.inp_a_rh,
            self.inp_a_ls,
            self.inp_a_le,
            self.inp_a_lh,
        ]:
            layout.addWidget(w)
        return box

    def _build_torque_section(self) -> QGroupBox:
        box = QGroupBox("Torque Polynomials (c0, c1, ...)")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_tau_hub = LabeledInput("Hub", "0", "Hub torque")
        self.inp_tau_rs = LabeledInput("R Shoulder", "0", "RS torque")
        self.inp_tau_re = LabeledInput("R Elbow", "0", "RE torque")
        self.inp_tau_rh = LabeledInput("R Wrist", "0", "RH torque")
        self.inp_tau_ls = LabeledInput("L Shoulder", "0", "LS torque")
        self.inp_tau_le = LabeledInput("L Elbow", "0", "LE torque")
        self.inp_tau_lh = LabeledInput("L Wrist", "0", "LH torque")
        for w in [
            self.inp_tau_hub,
            self.inp_tau_rs,
            self.inp_tau_re,
            self.inp_tau_rh,
            self.inp_tau_ls,
            self.inp_tau_le,
            self.inp_tau_lh,
        ]:
            layout.addWidget(w)
        return box

    def _build_sim_section(self) -> QGroupBox:
        box = QGroupBox("Simulation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        self.inp_tend = LabeledInput("Duration (s)", "2.0", "Total time")
        layout.addWidget(self.inp_tend)
        return box

    def _build_dissipation_section(self) -> QGroupBox:
        box = QGroupBox("Dissipation")
        box.setStyleSheet(STYLE_GROUP)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(3)
        self.inp_b_hub = LabeledInput("b hub", "0.0", "Viscous (N·m·s)")
        self.inp_b_rs = LabeledInput("b RS", "0.0", "Viscous (N·m·s)")
        self.inp_b_re = LabeledInput("b RE", "0.0", "Viscous (N·m·s)")
        self.inp_b_rh = LabeledInput("b RH", "0.0", "Viscous (N·m·s)")
        self.inp_b_ls = LabeledInput("b LS", "0.0", "Viscous (N·m·s)")
        self.inp_b_le = LabeledInput("b LE", "0.0", "Viscous (N·m·s)")
        self.inp_b_lh = LabeledInput("b LH", "0.0", "Viscous (N·m·s)")
        for w in [
            self.inp_b_hub,
            self.inp_b_rs,
            self.inp_b_re,
            self.inp_b_rh,
            self.inp_b_ls,
            self.inp_b_le,
            self.inp_b_lh,
        ]:
            layout.addWidget(w)
        return box

    # ── Abstract interface implementation ────────────────────────────

    def _get_joint_names(self) -> list[str]:
        return [
            "Hub",
            "R Shoulder",
            "R Elbow",
            "R Wrist",
            "L Shoulder",
            "L Elbow",
            "L Wrist",
        ]

    def _get_torque_inputs(self) -> dict[str, LabeledInput]:
        return {
            "Hub": self.inp_tau_hub,
            "R Shoulder": self.inp_tau_rs,
            "R Elbow": self.inp_tau_re,
            "R Wrist": self.inp_tau_rh,
            "L Shoulder": self.inp_tau_ls,
            "L Elbow": self.inp_tau_le,
            "L Wrist": self.inp_tau_lh,
        }

    def _apply_preset(self, name: str) -> None:
        if not (name is not None):
            raise ValueError("name must be provided")
        if name not in self.PRESETS:
            return
        p = self.PRESETS[name]

        def _sv(widget: Any, val: Any) -> None:
            """Set value on UnitAwareInput (float) or LabeledInput (str)."""
            try:
                widget.set_value(float(str(val)), is_si=True)
            except TypeError:
                widget.set_value(str(val))

        _sv(self.inp_m_hub, p["m_hub"])
        _sv(self.inp_m_r_upper, p["m_r_upper"])
        _sv(self.inp_m_r_fore, p["m_r_fore"])
        _sv(self.inp_m_l_upper, p["m_l_upper"])
        _sv(self.inp_m_l_fore, p["m_l_fore"])
        _sv(self.inp_m_club, p["m_club"])
        _sv(self.inp_m_clubhead, p["m_clubhead"])
        _sv(self.inp_L_hub, p["L_hub"])
        _sv(self.inp_L_r_upper, p["L_r_upper"])
        _sv(self.inp_L_r_fore, p["L_r_fore"])
        _sv(self.inp_L_l_upper, p["L_l_upper"])
        _sv(self.inp_L_l_fore, p["L_l_fore"])
        _sv(self.inp_L_club, p["L_club"])
        _sv(self.inp_d_rs, p["d_rs"])
        _sv(self.inp_d_ls, p["d_ls"])
        _sv(self.inp_grip_right, p["grip_right"])
        _sv(self.inp_grip_left, p["grip_left"])
        _sv(self.inp_th_hub, p["theta_hub"])
        _sv(self.inp_a_rs, p["alpha_rs"])
        _sv(self.inp_a_re, p["alpha_re"])
        _sv(self.inp_a_rh, p["alpha_rh"])
        _sv(self.inp_a_ls, p["alpha_ls"])
        _sv(self.inp_a_le, p["alpha_le"])
        _sv(self.inp_a_lh, p["alpha_lh"])
        _sv(self.inp_tau_hub, p["tau_hub"])
        _sv(self.inp_tau_rs, p["tau_rs"])
        _sv(self.inp_tau_re, p["tau_re"])
        _sv(self.inp_tau_rh, p["tau_rh"])
        _sv(self.inp_tau_ls, p["tau_ls"])
        _sv(self.inp_tau_le, p["tau_le"])
        _sv(self.inp_tau_lh, p["tau_lh"])
        _sv(self.inp_tend, p["t_end"])
        _sv(self.inp_L_rscap, p.get("L_rscap", 0.12))
        _sv(self.inp_L_lscap, p.get("L_lscap", 0.12))
        _sv(self.inp_m_rscap, p.get("m_rscap", 0.5))
        _sv(self.inp_m_lscap, p.get("m_lscap", 0.5))

    def get_params(self) -> dict:
        """Parse all inputs into a simulation parameter dict.

        Raises ValueError on invalid input.
        """
        params = {
            "m_hub": self._uai_or_parse(self.inp_m_hub, "m_hub"),
            "m_r_upper": self._uai_or_parse(self.inp_m_r_upper, "m_r_upper"),
            "m_r_fore": self._uai_or_parse(self.inp_m_r_fore, "m_r_fore"),
            "m_l_upper": self._uai_or_parse(self.inp_m_l_upper, "m_l_upper"),
            "m_l_fore": self._uai_or_parse(self.inp_m_l_fore, "m_l_fore"),
            "m_club": self._uai_or_parse(self.inp_m_club, "m_club"),
            "m_clubhead": self._uai_or_parse(self.inp_m_clubhead, "m_clubhead"),
            "L_hub": self._uai_or_parse(self.inp_L_hub, "L_hub"),
            "L_r_upper": self._uai_or_parse(self.inp_L_r_upper, "L_r_upper"),
            "L_r_fore": self._uai_or_parse(self.inp_L_r_fore, "L_r_fore"),
            "L_l_upper": self._uai_or_parse(self.inp_L_l_upper, "L_l_upper"),
            "L_l_fore": self._uai_or_parse(self.inp_L_l_fore, "L_l_fore"),
            "L_club": self._uai_or_parse(self.inp_L_club, "L_club"),
            "d_rs": parse_float(self.inp_d_rs, "d_rs"),
            "d_ls": parse_float(self.inp_d_ls, "d_ls"),
            "grip_right": parse_float(self.inp_grip_right, "grip_right"),
            "grip_left": parse_float(self.inp_grip_left, "grip_left"),
            "theta_hub_rad": np.radians(parse_float(self.inp_th_hub, "theta_hub")),
            "alpha_rs_rad": np.radians(parse_float(self.inp_a_rs, "alpha_rs")),
            "alpha_re_rad": np.radians(parse_float(self.inp_a_re, "alpha_re")),
            "alpha_rh_rad": np.radians(parse_float(self.inp_a_rh, "alpha_rh")),
            "alpha_ls_rad": np.radians(parse_float(self.inp_a_ls, "alpha_ls")),
            "alpha_le_rad": np.radians(parse_float(self.inp_a_le, "alpha_le")),
            "alpha_lh_rad": np.radians(parse_float(self.inp_a_lh, "alpha_lh")),
            "hub_coeffs": parse_coeffs(self.inp_tau_hub, "Hub torque"),
            "rs_coeffs": parse_coeffs(self.inp_tau_rs, "RS torque"),
            "re_coeffs": parse_coeffs(self.inp_tau_re, "RE torque"),
            "rh_coeffs": parse_coeffs(self.inp_tau_rh, "RH torque"),
            "ls_coeffs": parse_coeffs(self.inp_tau_ls, "LS torque"),
            "le_coeffs": parse_coeffs(self.inp_tau_le, "LE torque"),
            "lh_coeffs": parse_coeffs(self.inp_tau_lh, "LH torque"),
            "t_end": parse_float(self.inp_tend, "Duration"),
            "gravity_on": True,  # Gravity always on (#1209)
            "b_hub": parse_float(self.inp_b_hub, "b_hub"),
            "b_rs": parse_float(self.inp_b_rs, "b_rs"),
            "b_re": parse_float(self.inp_b_re, "b_re"),
            "b_rh": parse_float(self.inp_b_rh, "b_rh"),
            "b_ls": parse_float(self.inp_b_ls, "b_ls"),
            "b_le": parse_float(self.inp_b_le, "b_le"),
            "b_lh": parse_float(self.inp_b_lh, "b_lh"),
            "L_rscap": parse_float(self.inp_L_rscap, "L_rscap"),
            "L_lscap": parse_float(self.inp_L_lscap, "L_lscap"),
            "m_rscap": parse_float(self.inp_m_rscap, "m_rscap"),
            "m_lscap": parse_float(self.inp_m_lscap, "m_lscap"),
        }
        for name in (
            "m_hub",
            "m_r_upper",
            "m_r_fore",
            "m_l_upper",
            "m_l_fore",
            "m_club",
            "L_hub",
            "L_r_upper",
            "L_r_fore",
            "L_l_upper",
            "L_l_fore",
            "L_club",
        ):
            require_positive(params[name], name)
        for name in (
            "m_clubhead",
            "d_rs",
            "d_ls",
            "grip_right",
            "grip_left",
            "b_hub",
            "b_rs",
            "b_re",
            "b_rh",
            "b_ls",
            "b_le",
            "b_lh",
        ):
            require_non_negative(params[name], name)
        if params["grip_right"] > params["L_club"]:
            raise ValueError("grip_right must be ≤ L_club")
        if params["grip_left"] > params["L_club"]:
            raise ValueError("grip_left must be ≤ L_club")

        # Torque saturation — flat list → np.ndarray for physics
        torque_lims = self._parse_torque_limits()
        if torque_lims is not None:
            params["enable_clamp"] = True
            params["torque_limits"] = torque_lims  # list[float], len 7
        else:
            params["enable_clamp"] = False

        # Joint angle limits — (mins_rad, maxs_rad, stiffness)
        joint_lims = self._parse_joint_limits()
        if joint_lims is not None:
            mins_rad, maxs_rad, stiffness = joint_lims
            params["enable_limits"] = True
            params["limit_mins_rad"] = mins_rad
            params["limit_maxs_rad"] = maxs_rad
            params["limit_stiffness"] = stiffness
        else:
            params["enable_limits"] = False

        return params
