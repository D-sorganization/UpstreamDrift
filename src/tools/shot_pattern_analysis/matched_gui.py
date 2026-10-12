"""Optional controls for the paired matched-delivery study."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Mapping

from PyQt6 import QtCore, QtGui, QtWidgets

from src.tools.async_action import AsyncActionBar, WorkerContext
from .core import PATTERN_NAMES
from .preview import scale_previews
from .matched_config import MatchedConfig, matched_preset
from .noise import NOISE_FIELDS, NoiseSpec


def _execute_and_export(request: Any, output_dir: Path, **kwargs: Any) -> Path:
    from .matched_export import execute_and_export

    return execute_and_export(request, output_dir, **kwargs)


from .matched_formatting import (
    format_sg_interval,
    format_sg_value,
    is_finite_real,
)

_is_finite_real = is_finite_real
_format_sg_value = format_sg_value


def _format_scenario_context(
    scoring: Mapping[str, Any], effective_club: str | None, target_x_m: float | None
) -> str:
    """Describe the saved course context without inventing missing distances."""
    driver = effective_club == "driver"
    hole_distance = scoring.get("hole_distance_m")
    fairway_width = scoring.get("fairway_half_width_m")
    green_radius = scoring.get("green_radius_m")
    target_distance = (
        scoring.get("target_x_m")
        if scoring.get("target_x_m") is not None
        else target_x_m
    )

    scenario = scoring.get("scenario")
    if not scenario:
        if driver or hole_distance is not None:
            scenario = "Historical PGA tee shot on a flat hole with a fairway corridor"
        else:
            scenario = "Historical PGA approach to a centered circular green with rough outside"

    context_details: list[str] = []
    if driver or hole_distance is not None:
        if _is_finite_real(hole_distance) and f"{hole_distance:g} m" not in scenario:
            context_details.append(f"{hole_distance:g} m hole")
        if (
            _is_finite_real(fairway_width)
            and f"{2 * fairway_width:g} m" not in scenario
        ):
            context_details.append(f"{2 * fairway_width:g} m fairway corridor")
    else:
        if (
            _is_finite_real(target_distance)
            and f"{target_distance:.1f} m" not in scenario
        ):
            context_details.append(f"{target_distance:.1f} m fairway start")
        if _is_finite_real(green_radius) and f"{green_radius:g} m" not in scenario:
            context_details.append(f"{green_radius:g} m green radius")

    detail_str = f" ({', '.join(context_details)})" if context_details else ""
    return f"Scenario: {scenario}{detail_str}."


def _format_supported_population(scoring: Mapping[str, Any]) -> str:
    """Retain the saved pairing denominator and conditioning statement."""
    status = scoring.get("status")
    paired_n = scoring.get("paired_n")
    total_n = scoring.get("total_n")
    if (
        isinstance(paired_n, int)
        and not isinstance(paired_n, bool)
        and isinstance(total_n, int)
        and not isinstance(total_n, bool)
    ):
        if status == "conditional_only" or paired_n < total_n:
            pop_line = (
                f"Common supported pairs: {paired_n}/{total_n} "
                "(conditional on common supported indices only; full population unavailable due to unsupported terminal lies)."
            )
        else:
            pop_line = f"Common supported pairs: {paired_n}/{total_n} (all samples)."
    else:
        pop_line = "Common supported pairs: unspecified population."
    return pop_line


def format_matched_course_scoring(
    scoring: Mapping[str, Any] | None,
    *,
    club_id: str | None = None,
    target_x_m: float | None = None,
) -> str:
    """Format saved course scoring results for compact display in the matched GUI.

    Preconditions:
      - scoring is None or a Mapping representing saved course_scoring
      - club_id is None or a string indicating the club identifier
      - target_x_m is None or a finite float indicating target downrange distance
    Postcondition:
      - Returns a non-empty human-readable string without fabricating empirical claims
        or converting missing/null values to zeros.
    """
    if scoring is None:
        return "Model-conditional strokes gained: Not available in saved results."

    if not isinstance(scoring, Mapping):
        return "Model-conditional strokes gained: Invalid scoring payload format."

    effective_club = scoring.get("club_id") or club_id
    status = scoring.get("status")

    if status not in ("available", "conditional_only"):
        reason = str(scoring.get("reason", "unspecified"))
        if reason == "fewer_than_two_common_supported_shots":
            paired_n = scoring.get("paired_n", 0)
            total_n = scoring.get("total_n", 0)
            return (
                "Model-conditional strokes gained is unavailable: "
                f"Insufficient common supported shots ({paired_n}/{total_n} common pairs; minimum 2 required)."
            )
        if effective_club == "custom" or "preset" in reason or "club_id" in reason:
            return (
                "Model-conditional strokes gained is unavailable for custom club: "
                "No published baseline context (only Driver, 7-Iron, and Pitching Wedge illustrative presets are supported)."
            )
        return f"Model-conditional strokes gained is unavailable for this configuration: {reason}."

    context_line = _format_scenario_context(scoring, effective_club, target_x_m)

    pop_line = _format_supported_population(scoring)

    # Pattern mean SG
    patterns = scoring.get("patterns")
    mean_parts: list[str] = []
    if isinstance(patterns, Mapping):
        for name in PATTERN_NAMES:
            p_data = patterns.get(name)
            val = (
                p_data.get("mean_strokes_gained")
                if isinstance(p_data, Mapping)
                else None
            )
            mean_parts.append(f"{name}: {_format_sg_value(val)}")
    mean_sg_line = "Mean Strokes Gained: " + (
        "; ".join(mean_parts) if mean_parts else "unavailable"
    )

    # Paired differences vs Straight with actual provided 95% Monte Carlo interval
    paired = scoring.get("paired_benefit_vs_straight")
    paired_parts: list[str] = []
    if isinstance(paired, Mapping):
        for name in ("Draw", "Fade"):
            entry = paired.get(name)
            if isinstance(entry, Mapping):
                est = entry.get("estimate")
                low = entry.get("lower_95")
                high = entry.get("upper_95")
                if (
                    _is_finite_real(est)
                    and _is_finite_real(low)
                    and _is_finite_real(high)
                ):
                    rendered = format_sg_interval(est, low, high, check_reversed=False)
                    paired_parts.append(f"{name}: {rendered} strokes")
                else:
                    paired_parts.append(f"{name}: unavailable")
            else:
                paired_parts.append(f"{name}: unavailable")
    paired_line = "Paired Difference vs. Straight (95% Monte Carlo Interval): " + (
        "; ".join(paired_parts) if paired_parts else "unavailable"
    )

    # Caveats: conditional model, approximate baseline, carry-as-stop
    caveat_line = (
        "Caveats: Model-conditional historical PGA Tour benchmark (approximate baseline fit; not player-specific). "
        "Flat hypothetical geometry; carry landing is final lie without roll, hazards, or obstruction. "
        "Intervals describe Monte Carlo sampling only."
    )

    return f"{context_line}\n{pop_line}\n{mean_sg_line}.\n{paired_line}.\n{caveat_line}"


class MatchedStudyWidget(QtWidgets.QWidget):
    """Configure and run explicitly paired delivery-noise experiments."""

    CONFIG_SCHEMA = "shot-pattern-matched-config/1"
    _DEFAULT_SD = (1.2, 0.5, 0.25, 0.25 * 0.0254, 0.25 * 0.0254, 0.5, 0.5)
    _DISPLAY = {
        "shaft_rotation_deg": ("Shaft Rotation SD", "° SD", 0.1, 0.0, 15.0),
        "path_deg": ("Path SD", "° SD", 0.1, 0.0, 10.0),
        "speed_mps": ("Club Speed SD", "mph SD", 0.01, 0.0, 20.0),
        "contact_horizontal_m": ("Horizontal Contact SD", "in SD", 0.01, 0.0, 2.0),
        "contact_vertical_m": ("Vertical Contact SD", "in SD", 0.01, 0.0, 2.0),
        "shaft_lean_deg": ("Shaft Lean SD", "° SD", 0.1, 0.0, 10.0),
        "attack_angle_deg": ("Attack Angle SD", "° SD", 0.1, 0.0, 10.0),
    }
    _SCALE_TO_SI = {
        "shaft_rotation_deg": 1.0,
        "path_deg": 1.0,
        "speed_mps": 0.44704,
        "contact_horizontal_m": 0.0254,
        "contact_vertical_m": 0.0254,
        "shaft_lean_deg": 1.0,
        "attack_angle_deg": 1.0,
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._closed = False
        self.club_config = matched_preset("driver")
        self._applying_config = False
        self._correlation_implicit_identity = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Matched Delivery Study")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        root.addWidget(title)
        intro = QtWidgets.QLabel(
            "Model-conditional paired shots with all seven delivery-noise inputs. "
            "Illustrative equipment assumptions; not measured golfer data."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setObjectName("matchedControlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(controls)
        root.addWidget(self.controls_scroll, stretch=2)
        self._build_general_controls(controls_layout)
        self._build_club_controls(controls_layout)
        self._build_noise_controls(controls_layout)
        self._build_correlation_controls(controls_layout)
        self._build_result_widgets(root)

    def _build_general_controls(self, controls_layout: QtWidgets.QVBoxLayout) -> None:
        self.output_edit = QtWidgets.QLineEdit(
            str(Path.cwd() / "matched_delivery_results")
        )
        self.shot_count = QtWidgets.QSpinBox()
        self.shot_count.setRange(2, 100_000)
        self.shot_count.setValue(10_000)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(20_261_009)
        self.curve_scale = QtWidgets.QDoubleSpinBox()
        self.curve_scale.setRange(0.1, 2.0)
        self.curve_scale.setDecimals(1)
        self.curve_scale.setValue(1.0)
        self.target_distance = QtWidgets.QDoubleSpinBox()
        self.target_distance.setRange(0.0, 1_000_000.0)
        self.target_distance.setDecimals(15)
        self.target_distance.setSingleStep(0.1)
        self.target_distance.setValue(0.0)
        self.target_distance.setSpecialValueText("Match Within Lean")
        self.match_tolerance = QtWidgets.QDoubleSpinBox()
        self.match_tolerance.setRange(0.005, 1.0)
        self.match_tolerance.setDecimals(3)
        self.match_tolerance.setValue(0.05)
        self.target_radius = QtWidgets.QDoubleSpinBox()
        self.target_radius.setRange(0.1, 100.0)
        self.target_radius.setDecimals(1)
        self.target_radius.setValue(15.0)
        self.use_absolute_face_bounds = QtWidgets.QCheckBox(
            "Use Absolute Face Bounds (otherwise each pattern's reference face ±3°)"
        )
        self.face_bound_min = QtWidgets.QDoubleSpinBox()
        self.face_bound_max = QtWidgets.QDoubleSpinBox()
        for control, value in ((self.face_bound_min, -6.0), (self.face_bound_max, 6.0)):
            control.setRange(-15.0, 15.0)
            control.setDecimals(8)
            control.setSuffix(" °")
            control.setValue(value)
            control.setEnabled(False)
            self.use_absolute_face_bounds.toggled.connect(control.setEnabled)
        self.speed_factor_min = QtWidgets.QDoubleSpinBox()
        self.speed_factor_max = QtWidgets.QDoubleSpinBox()
        for control, value in (
            (self.speed_factor_min, 0.75),
            (self.speed_factor_max, 1.25),
        ):
            control.setRange(0.0, 1_000_000.0)
            control.setDecimals(8)
            control.setValue(value)

        form = QtWidgets.QFormLayout()
        form.addRow("Output Folder", self.output_edit)
        form.addRow("Shots Per Pattern", self.shot_count)
        form.addRow("Random Seed", self.seed)
        form.addRow("Curvature Scale", self.curve_scale)
        form.addRow("Target Downrange (m; 0 = match within lean)", self.target_distance)
        form.addRow("Nominal Match Tolerance (m)", self.match_tolerance)
        form.addRow("Target Radius (m)", self.target_radius)
        form.addRow(self.use_absolute_face_bounds)
        form.addRow("Absolute Face Minimum (°)", self.face_bound_min)
        form.addRow("Absolute Face Maximum (°)", self.face_bound_max)
        form.addRow("Speed Minimum (× reference club speed)", self.speed_factor_min)
        form.addRow("Speed Maximum (× reference club speed)", self.speed_factor_max)
        controls_layout.addLayout(form)

    def _build_club_controls(self, controls_layout: QtWidgets.QVBoxLayout) -> None:
        club_box = QtWidgets.QGroupBox("Illustrative Club Assumptions")
        club_form = QtWidgets.QFormLayout(club_box)
        self.club_choice = QtWidgets.QComboBox()
        for club_id, label in (
            ("driver", "Driver"),
            ("seven_iron", "7-Iron"),
            ("pitching_wedge", "Pitching Wedge"),
            ("custom", "Custom"),
        ):
            self.club_choice.addItem(label, club_id)
        self.club_choice.currentIndexChanged.connect(self._preset_selected)
        self.club_speed_mph = QtWidgets.QDoubleSpinBox()
        self.club_speed_mph.setRange(10.0, 200.0)
        self.club_speed_mph.setDecimals(1)
        self.loft = QtWidgets.QDoubleSpinBox()
        self.loft.setRange(1.0, 79.0)
        self.loft.setDecimals(1)
        self.shaft_lean = QtWidgets.QDoubleSpinBox()
        self.shaft_lean.setRange(-30.0, 30.0)
        self.shaft_lean.setDecimals(1)
        self.attack_angle = QtWidgets.QDoubleSpinBox()
        self.attack_angle.setRange(-29.0, 29.0)
        self.attack_angle.setDecimals(1)
        self.club_mass = QtWidgets.QDoubleSpinBox()
        self.club_mass.setRange(0.05, 1.0)
        self.club_mass.setDecimals(3)
        self.contact_law = QtWidgets.QComboBox()
        self.contact_law.addItem(
            "Incremental Coulomb (Energetic Restitution)",
            "incremental_coulomb_energetic",
        )
        self.contact_law.addItem(
            "Maximum-Dissipation Impulse (Newton Restitution)",
            "maximum_dissipation_newton",
        )
        self.contact_law_help = QtWidgets.QLabel(
            "Contact-law alternatives are numerical model assumptions; neither has been validated against measured club-ball impacts."
        )
        self.contact_law_help.setWordWrap(True)
        for widget in (
            self.club_speed_mph,
            self.loft,
            self.shaft_lean,
            self.attack_angle,
            self.club_mass,
        ):
            widget.valueChanged.connect(self._mark_custom)
        self.contact_law.currentIndexChanged.connect(self._mark_custom)
        self.static_loft_label = QtWidgets.QLabel()
        self.assumption_label = QtWidgets.QLabel()
        self.assumption_label.setWordWrap(True)
        club_form.addRow("Club", self.club_choice)
        club_form.addRow("Club Speed (mph)", self.club_speed_mph)
        club_form.addRow("Nominal Static Loft (°)", self.loft)
        club_form.addRow("Nominal Shaft Lean (°)", self.shaft_lean)
        club_form.addRow("Nominal Attack Angle (°)", self.attack_angle)
        club_form.addRow("Head Mass (kg)", self.club_mass)
        club_form.addRow("Contact Law", self.contact_law)
        club_form.addRow(self.contact_law_help)
        club_form.addRow(self.static_loft_label)
        club_form.addRow(self.assumption_label)
        controls_layout.addWidget(club_box)
        self._set_club_values(self.club_config)

    def _build_noise_controls(self, controls_layout: QtWidgets.QVBoxLayout) -> None:
        noise_box = QtWidgets.QGroupBox("Delivery Variation (Gaussian SD Assumptions)")
        noise_form = QtWidgets.QFormLayout(noise_box)
        self.noise_inputs: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for field in NOISE_FIELDS:
            label, suffix, step, low, high = self._DISPLAY[field]
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(low, high)
            control.setDecimals(
                2
                if field in ("speed_mps", "contact_horizontal_m", "contact_vertical_m")
                else 1
            )
            control.setSingleStep(step)
            control.setSuffix(f" {suffix}")
            control.setValue(
                self._DEFAULT_SD[NOISE_FIELDS.index(field)] / self._SCALE_TO_SI[field]
            )
            self.noise_inputs[field] = control
            noise_form.addRow(label, control)
        controls_layout.addWidget(noise_box)

    def _build_correlation_controls(
        self, controls_layout: QtWidgets.QVBoxLayout
    ) -> None:
        corr_box = QtWidgets.QGroupBox("Editable Input Correlation Matrix")
        corr_layout = QtWidgets.QVBoxLayout(corr_box)
        self.correlation_table = QtWidgets.QTableWidget(7, 7)
        correlation_labels = (
            "Shaft Twist",
            "Path",
            "Speed",
            "Toe/Heel",
            "High/Low",
            "Shaft Lean",
            "Attack",
        )
        self.correlation_table.setHorizontalHeaderLabels(correlation_labels)
        self.correlation_table.setVerticalHeaderLabels(correlation_labels)
        header_view = self.correlation_table.horizontalHeader()
        assert header_view is not None
        header_view.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for index, field in enumerate(NOISE_FIELDS):
            tooltip = f"{self._DISPLAY[field][0]} • Export Field: {field}. Correlations are dimensionless."
            horizontal_item = self.correlation_table.horizontalHeaderItem(index)
            vertical_item = self.correlation_table.verticalHeaderItem(index)
            assert horizontal_item is not None and vertical_item is not None
            horizontal_item.setToolTip(tooltip)
            vertical_item.setToolTip(tooltip)
        self._setting_correlation = False
        for i in range(7):
            for j in range(7):
                item = QtWidgets.QTableWidgetItem("1.000" if i == j else "0.000")
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if i == j:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.correlation_table.setItem(i, j, item)
        self.correlation_table.itemChanged.connect(self._correlation_edited)
        corr_layout.addWidget(self.correlation_table)
        self.face_path_rho = QtWidgets.QDoubleSpinBox()
        self.face_path_rho.setRange(-0.99, 0.99)
        self.face_path_rho.setDecimals(2)
        self.approx_rho_button = QtWidgets.QPushButton(
            "Set Approximate Face/Path Correlation"
        )
        self.approx_rho_button.clicked.connect(
            self.apply_approximate_face_path_correlation
        )
        rho_row = QtWidgets.QHBoxLayout()
        rho_row.addWidget(QtWidgets.QLabel("Approximate Face/Path ρ"))
        rho_row.addWidget(self.face_path_rho)
        rho_row.addWidget(self.approx_rho_button)
        corr_layout.addLayout(rho_row)
        self.correlation_help = QtWidgets.QLabel(
            "Face is approximately nominal minus shaft rotation; setting this correlation negates the requested face/path ρ."
        )
        self.correlation_help.setWordWrap(True)
        corr_layout.addWidget(self.correlation_help)
        controls_layout.addWidget(corr_box)

    def _build_result_widgets(self, root: QtWidgets.QVBoxLayout) -> None:
        actions = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run Matched Study")
        self.save_config_button = QtWidgets.QPushButton("Export Config JSON…")
        self.load_config_button = QtWidgets.QPushButton("Import Config JSON…")
        self.run_button.clicked.connect(self.run_analysis)
        self.save_config_button.clicked.connect(self._save_dialog)
        self.load_config_button.clicked.connect(self._load_dialog)
        actions.addWidget(self.run_button)
        actions.addWidget(self.save_config_button)
        actions.addWidget(self.load_config_button)
        root.addLayout(actions)
        self.action_bar = AsyncActionBar()
        self.action_bar.set_trigger_buttons(self.run_button)
        self.action_bar.cancelled.connect(self._cancelled)
        self.action_bar.failed.connect(self._failed)
        root.addWidget(self.action_bar)
        self.status_label = QtWidgets.QLabel(
            "Ready. Invalid shots remain in the exported population."
        )
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self.result_label = QtWidgets.QLabel(
            "Run a study to see supported/unsupported counts and exports."
        )
        self.result_label.setWordWrap(True)
        root.addWidget(self.result_label)
        self.scoring_label = QtWidgets.QLabel(
            "Model-conditional strokes gained appears after running a study."
        )
        self.scoring_label.setWordWrap(True)
        root.addWidget(self.scoring_label)
        self.previews = QtWidgets.QTabWidget()
        self.preview = QtWidgets.QLabel(
            "A report preview will appear after a successful run."
        )
        self.overhead_preview = QtWidgets.QLabel(
            "Raw overhead trajectories will appear after a successful run."
        )
        for preview in (self.preview, self.overhead_preview):
            preview.setMinimumHeight(260)
            preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.previews.addTab(self.preview, "Endpoint Density")
        self.previews.addTab(self.overhead_preview, "Raw Flight Trajectories")
        root.addWidget(self.previews, stretch=1)

    def _set_club_values(self, config: MatchedConfig) -> None:
        was_applying_config = self._applying_config
        self._applying_config = True
        self.club_config = config
        self.club_speed_mph.setValue(config.club_speed_mps / 0.44704)
        self.loft.setValue(config.loft_deg)
        self.shaft_lean.setValue(config.shaft_lean_deg)
        self.attack_angle.setValue(config.attack_angle_deg)
        self.club_mass.setValue(config.clubhead_mass_kg)
        self.contact_law.setCurrentIndex(self.contact_law.findData(config.contact_law))
        self.static_loft_label.setText(
            "Nominal Static Loft: The same face baseline is used for every pattern; delivery lean can change dynamic loft."
        )
        self._applying_config = was_applying_config
        self._update_assumption()

    def _preset_selected(self, _index: int) -> None:
        club_id = str(self.club_choice.currentData())
        if club_id != "custom":
            self._set_club_values(matched_preset(club_id))
            return
        self._mark_custom()

    def _mark_custom(self, _value: float | None = None) -> None:
        if self._applying_config:
            return
        selected_id = str(self.club_choice.currentData() or self.club_config.club_id)
        self.club_config = replace(
            self.club_config,
            club_id=selected_id,
            club_speed_mps=self.club_speed_mph.value() * 0.44704,
            loft_deg=self.loft.value(),
            shaft_lean_deg=self.shaft_lean.value(),
            attack_angle_deg=self.attack_angle.value(),
            clubhead_mass_kg=self.club_mass.value(),
            contact_law=str(self.contact_law.currentData()),
        )
        self._update_assumption()

    def _update_assumption(self) -> None:
        club_id = self.club_config.club_id
        context = "Custom" if club_id == "custom" else club_id.replace("_", " ").title()
        preset = (
            matched_preset(club_id)
            if club_id in {"driver", "seven_iron", "pitching_wedge"}
            else None
        )
        modified = preset is not None and any(
            getattr(self.club_config, key) != getattr(preset, key)
            for key in (
                "club_speed_mps",
                "loft_deg",
                "shaft_lean_deg",
                "attack_angle_deg",
                "clubhead_mass_kg",
            )
        )
        if modified:
            context += " (Modified)"
        self.assumption_label.setText(
            f"{context} equipment settings are illustrative model assumptions, not measured player averages. "
            "Loft remains static and common across patterns; advanced equipment geometry is retained in the JSON configuration."
        )

    def _correlation_edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._setting_correlation or item.row() == item.column():
            return
        self._correlation_implicit_identity = False
        try:
            value = float(item.text())
            if not -1 <= value <= 1:
                raise ValueError
        except ValueError:
            value = 0.0
            item.setText("0.000")
        self._setting_correlation = True
        mirror = self._correlation_cell(item.column(), item.row())
        mirror.setText(f"{value:.3f}")
        self._setting_correlation = False

    def _correlation_cell(self, row: int, column: int) -> QtWidgets.QTableWidgetItem:
        item = self.correlation_table.item(row, column)
        if item is None:
            raise RuntimeError("correlation matrix cell is missing")
        return item

    def set_correlation(self, field_a: str, field_b: str, value: float) -> None:
        if field_a not in NOISE_FIELDS or field_b not in NOISE_FIELDS:
            raise ValueError("unknown noise field")
        if field_a == field_b or not -1 <= value <= 1:
            raise ValueError("off-diagonal correlations must be within [-1,1]")
        self._correlation_implicit_identity = False
        i, j = NOISE_FIELDS.index(field_a), NOISE_FIELDS.index(field_b)
        self._setting_correlation = True
        self._correlation_cell(i, j).setText(f"{value:.3f}")
        self._correlation_cell(j, i).setText(f"{value:.3f}")
        self._setting_correlation = False

    def correlation_value(self, field_a: str, field_b: str) -> float:
        return float(
            self._correlation_cell(
                NOISE_FIELDS.index(field_a), NOISE_FIELDS.index(field_b)
            ).text()
        )

    def correlation_matrix(self) -> Any:
        import numpy as np

        matrix: list[list[float]] = []
        for i in range(7):
            row = [float(self._correlation_cell(i, j).text()) for j in range(7)]
            matrix.append(row)
        return np.asarray(matrix)

    def apply_approximate_face_path_correlation(self) -> None:
        self.set_correlation(
            "shaft_rotation_deg", "path_deg", -self.face_path_rho.value()
        )

    def build_experiment(self) -> Any:
        from .matched_runner import ExperimentConfig

        deviations = tuple(
            self.noise_inputs[field].value() * self._SCALE_TO_SI[field]
            for field in NOISE_FIELDS
        )
        corr = tuple(tuple(float(v) for v in row) for row in self.correlation_matrix())
        noise = NoiseSpec(
            deviations, None if self._correlation_implicit_identity else corr
        )
        target = self.target_distance.value() or None
        return ExperimentConfig(
            club=self.club_config,
            noise=noise,
            n_shots=self.shot_count.value(),
            seed=self.seed.value(),
            curve_scale=self.curve_scale.value(),
            target_x_m=target,
            match_tolerance_m=self.match_tolerance.value(),
            target_radius_m=self.target_radius.value(),
            match_face_bounds_deg=(
                self.face_bound_min.value(),
                self.face_bound_max.value(),
            )
            if self.use_absolute_face_bounds.isChecked()
            else None,
            match_speed_factors=(
                self.speed_factor_min.value(),
                self.speed_factor_max.value(),
            ),
        )

    @staticmethod
    def _request_json(request: Any) -> dict[str, Any]:
        return {"schema": MatchedStudyWidget.CONFIG_SCHEMA, **asdict(request)}

    def export_config(self, path: Path, request: Any | None = None) -> None:
        request = request or self.build_experiment()
        Path(path).write_text(
            json.dumps(self._request_json(request), indent=2) + "\n", encoding="utf-8"
        )

    def load_config(self, path: Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != self.CONFIG_SCHEMA:
            raise ValueError("unsupported matched-study config schema")
        values = {key: value for key, value in payload.items() if key != "schema"}
        club_values = dict(values["club"])
        club_values["inertia_diagonal_kg_m2"] = tuple(
            club_values["inertia_diagonal_kg_m2"]
        )
        club = MatchedConfig(**club_values)
        noise_values = values["noise"]
        noise = NoiseSpec(
            tuple(noise_values["standard_deviations"]),
            tuple(tuple(row) for row in noise_values["correlation"])
            if noise_values.get("correlation") is not None
            else None,
        )
        from .matched_runner import ExperimentConfig

        request = ExperimentConfig(
            club=club,
            noise=noise,
            **{k: v for k, v in values.items() if k not in {"club", "noise"}},
        )
        self._apply_request(request)

    def _apply_request(self, request: Any) -> None:
        self._validate_import_display(request)
        self._applying_config = True
        self.club_choice.setCurrentIndex(
            self.club_choice.findData(request.club.club_id)
        )
        self.club_config = request.club
        self.club_speed_mph.setValue(request.club.club_speed_mps / 0.44704)
        self.loft.setValue(request.club.loft_deg)
        self.shaft_lean.setValue(request.club.shaft_lean_deg)
        self.attack_angle.setValue(request.club.attack_angle_deg)
        self.club_mass.setValue(request.club.clubhead_mass_kg)
        self.contact_law.setCurrentIndex(
            self.contact_law.findData(request.club.contact_law)
        )
        self._applying_config = False
        self.shot_count.setValue(request.n_shots)
        self.seed.setValue(request.seed)
        self.curve_scale.setValue(request.curve_scale)
        self.target_distance.setValue(request.target_x_m or 0.0)
        if self.target_distance.value() != (request.target_x_m or 0.0):
            raise ValueError("target distance cannot display exactly in GUI")
        self.match_tolerance.setValue(request.match_tolerance_m)
        self.target_radius.setValue(request.target_radius_m)
        bounds = request.match_face_bounds_deg
        self.use_absolute_face_bounds.setChecked(bounds is not None)
        bound_controls: list[tuple[QtWidgets.QDoubleSpinBox, float]] = []
        if bounds is not None:
            bound_controls.extend(
                ((self.face_bound_min, bounds[0]), (self.face_bound_max, bounds[1]))
            )
        bound_controls.extend(
            (
                (self.speed_factor_min, request.match_speed_factors[0]),
                (self.speed_factor_max, request.match_speed_factors[1]),
            )
        )
        for control, value in bound_controls:
            control.setValue(value)
            if control.value() != value:
                raise ValueError("nominal search bound cannot display exactly in GUI")
        for field, value in zip(
            NOISE_FIELDS, request.noise.standard_deviations, strict=True
        ):
            self.noise_inputs[field].setValue(value / self._SCALE_TO_SI[field])
        corr = request.noise.correlation_matrix()
        self._setting_correlation = True
        for i in range(7):
            for j in range(7):
                self._correlation_cell(i, j).setText(f"{corr[i, j]:.3f}")
        self._setting_correlation = False
        self._correlation_implicit_identity = request.noise.correlation is None
        self._update_assumption()

    def _validate_import_display(self, request: Any) -> None:
        """Reject controls that would silently change a loaded experiment."""
        if (
            not self.shot_count.minimum()
            <= request.n_shots
            <= self.shot_count.maximum()
        ):
            raise ValueError("shot count cannot display exactly in GUI")
        if not self.seed.minimum() <= request.seed <= self.seed.maximum():
            raise ValueError("random seed cannot display exactly in GUI")

        def require_exact(control: QtWidgets.QDoubleSpinBox, value: float) -> None:
            probe = QtWidgets.QDoubleSpinBox()
            probe.setRange(control.minimum(), control.maximum())
            probe.setDecimals(control.decimals())
            probe.setValue(value)
            if probe.value() != value:
                raise ValueError("study control cannot display exactly in GUI")

        for control, value in (
            (self.curve_scale, request.curve_scale),
            (self.target_distance, request.target_x_m or 0.0),
            (self.match_tolerance, request.match_tolerance_m),
            (self.target_radius, request.target_radius_m),
            (self.speed_factor_min, request.match_speed_factors[0]),
            (self.speed_factor_max, request.match_speed_factors[1]),
        ):
            require_exact(control, value)
        if request.match_face_bounds_deg is not None:
            require_exact(self.face_bound_min, request.match_face_bounds_deg[0])
            require_exact(self.face_bound_max, request.match_face_bounds_deg[1])
        for field, value in zip(
            NOISE_FIELDS, request.noise.standard_deviations, strict=True
        ):
            require_exact(self.noise_inputs[field], value / self._SCALE_TO_SI[field])
        if request.noise.correlation is not None:
            matrix = request.noise.correlation_matrix()
            if any(
                float(f"{matrix[i, j]:.3f}") != float(matrix[i, j])
                for i in range(7)
                for j in range(7)
            ):
                raise ValueError("correlation cannot display exactly in GUI")

    def run_analysis(self) -> None:
        self.scoring_label.setText(
            "Model-conditional strokes gained: no result for the requested study yet."
        )
        output_text = self.output_edit.text().strip()
        if not output_text:
            self.status_label.setText("Choose an output folder first.")
            return
        try:
            request = self.build_experiment()
            output_dir = Path(output_text).expanduser()
            if output_dir.exists() and not output_dir.is_dir():
                raise ValueError("output path exists and is not a folder")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Invalid study settings: {exc}")
            return

        def work(ctx: WorkerContext) -> dict[str, Any]:
            try:
                destination = _execute_and_export(
                    request,
                    output_dir,
                    progress=lambda done, total: ctx.report(
                        min(1.0, done / total),
                        f"Simulating paired deliveries: {done}/{total} shots",
                    ),
                    cancelled=lambda: ctx.is_cancelled,
                )
            except Exception as exc:
                from .core import AnalysisCancelled

                if isinstance(exc, AnalysisCancelled):
                    ctx.raise_if_cancelled()
                raise
            ctx.raise_if_cancelled()
            return {"output_dir": destination}

        self.action_bar.start(
            "Matched delivery analysis", work, on_finished=self._completed
        )

    def _completed(self, bundle: dict[str, Any]) -> None:
        if self._closed:
            return
        output_dir = Path(bundle["output_dir"])
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") == "unavailable":
            self.result_label.setText(
                "No matched density was generated because the nominal setup is unavailable. "
                f"Reason: {summary.get('reason', 'unspecified')}. Results saved in {output_dir}"
            )
            self.status_label.setText(
                "Study completed with unavailable nominal geometry; no endpoint figures were fabricated."
            )
            self.scoring_label.setText(
                "Model-conditional strokes gained is unavailable because the nominal delivery could not be solved."
            )
            return
        config = summary["config"]
        club = config["club"]
        target_protocol = (
            "matched within this lean setting"
            if config["target_x_m"] is None
            else "fixed user-selected target"
        )
        header = (
            f"Target: {summary['target_x_m']:.1f} m ({target_protocol}). "
            "Solved nominal deliveries (face/path; club speed; delivered loft; residual):"
        )
        parts = []
        from .club_geometry import club_delivery

        for name, stats in summary.get("summary", {}).items():
            nominal = summary["nominals"][name]
            delivered_loft = club_delivery(
                club["loft_deg"],
                club["lie_deg"],
                club["shaft_lean_deg"],
                nominal["face_deg"],
                0.0,
            ).loft_deg
            empirical_rho = stats.get("realized_face_path_correlation")
            rho_text = (
                "unavailable" if empirical_rho is None else f"{empirical_rho:+.3f}"
            )
            parts.append(
                f"{name}: {nominal['face_deg']:+.2f}°/{nominal['path_deg']:+.2f}°; "
                f"{nominal['speed_mps'] / 0.44704:.1f} mph; {delivered_loft:.2f}° loft; "
                f"{nominal['residual_m']:.3f} m residual; "
                f"{stats.get('supported_n', 0)}/{stats.get('total_n', 0)} supported; "
                f"empirical face/path ρ={rho_text}; "
                f"unsupported={stats.get('unsupported_reasons', {})}"
            )
        self.result_label.setText(
            header + "\n" + "\n".join(parts) + f"\nResults saved in {output_dir}"
        )
        scoring = summary.get("course_scoring")
        scoring_text = format_matched_course_scoring(
            scoring,
            club_id=club.get("club_id"),
            target_x_m=summary.get("target_x_m"),
        )
        self.scoring_label.setText(scoring_text)
        self.status_label.setText(
            "Matched study complete. Density is conditional on supported airborne shots where any attempts were unsupported."
        )
        for filename, preview_label in (
            ("density.png", self.preview),
            ("overhead.png", self.overhead_preview),
        ):
            image_path = output_dir / filename
            if not image_path.is_file():
                continue
            from PyQt6.QtGui import QPixmap

            preview = QPixmap(str(image_path))
            if not preview.isNull():
                preview_label.setProperty("sourcePixmap", preview)
                preview_label.setToolTip(str(image_path))
        self._scale_previews()

    def _failed(self, message: str) -> None:
        if not self._closed:
            self.status_label.setText(f"Matched study failed: {message}")
            self.scoring_label.setText(
                "Model-conditional strokes gained is unavailable: the latest study failed."
            )

    def _cancelled(self) -> None:
        if not self._closed:
            self.status_label.setText(
                "Matched study cancelled; staged outputs were discarded."
            )
            self.scoring_label.setText(
                "Model-conditional strokes gained is unavailable: the latest study was cancelled."
            )

    def _save_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Study Config", "matched-study.json", "JSON (*.json)"
        )
        if path:
            try:
                self.export_config(Path(path))
                self.status_label.setText(f"Configuration saved to {path}")
            except (OSError, ValueError) as exc:
                self.status_label.setText(f"Could not save configuration: {exc}")

    def _load_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Study Config", "", "JSON (*.json)"
        )
        if path:
            try:
                self.load_config(Path(path))
                self.status_label.setText(f"Configuration loaded from {path}")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.status_label.setText(f"Could not load configuration: {exc}")

    @property
    def is_running(self) -> bool:
        """Return whether the asynchronous study worker is running."""
        return self.action_bar.is_running

    def cleanup(self) -> bool:
        self._closed = True
        return self.action_bar.shutdown()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._scale_previews()

    def _scale_previews(self) -> None:
        scale_previews((self.preview, self.overhead_preview))

    def delete_when_idle(self) -> None:
        self.action_bar.finished.connect(self.deleteLater)
        self.action_bar.failed.connect(self.deleteLater)
        self.action_bar.cancelled.connect(self.deleteLater)


__all__ = ["MatchedStudyWidget"]
