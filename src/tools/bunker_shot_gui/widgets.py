"""Qt input and display widgets for the BunkerShot3D workbench (issue #8618).

Every widget here is a *view*: it reads and writes the value objects in
:mod:`src.tools.bunker_shot_gui.design` and paints what
:mod:`src.tools.bunker_shot_gui.model` computed. None of them does physics,
and none of them draws anything it was not handed -- there is no procedural
preview anywhere in this package.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bunkershot3d.geometry import get_preset
from bunkershot3d.solvers import EnvelopeStatus

from .design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
    grind_preset_names,
    playing_condition_names,
)
from .report import status_colour, status_headline

__all__ = [
    "ConditionPanel",
    "DesignPanel",
    "GridMapWidget",
    "VerdictBanner",
]

_EMPTY_CELL = QColor(238, 234, 226)
_WINDOW_EDGE = QColor(20, 90, 40)
_MIN_MAP_HEIGHT_PX = 150
_CELL_GAP_PX = 1


class VerdictBanner(QLabel):
    """The validity verdict, stated before any number the tool reports.

    ADR-0032 makes this the load-bearing feature of the F0 tier: a bunker
    shot sits roughly 60x outside 3D-RFT's stated Froude limit, so a force
    without its verdict is not a result. The banner is therefore never empty
    and never quiet -- a refusal paints red and says no number is reported.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the banner in its "nothing has been run yet" state."""
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.show_idle()

    def show_idle(self) -> None:
        """Show the pre-run state, which claims nothing."""
        self._paint("#4a4a4a", "No shot has been run yet.")

    def show_busy(self, message: str) -> None:
        """Show a neutral in-progress state.

        Args:
            message: What the model is doing.
        """
        self._paint("#4a4a4a", message)

    def show_error(self, message: str) -> None:
        """Show an input error, which is not a solver verdict.

        Args:
            message: Why the inputs cannot be evaluated.
        """
        self._paint("#7a3b00", f"INPUT ERROR - {message}")

    def show_status(self, status: EnvelopeStatus) -> None:
        """Show a solver verdict.

        Args:
            status: The verdict status carried by the result.
        """
        self._paint(status_colour(status), status_headline(status))

    def _paint(self, colour: str, text: str) -> None:
        """Apply a colour and a message."""
        self.setStyleSheet(
            f"background-color: {colour}; color: white; font-weight: bold; "
            "padding: 10px; border-radius: 4px;"
        )
        self.setText(text)


class GridMapWidget(QWidget):
    """Paints a 2-D array of numbers it was given, and nothing else.

    Used for both spatial maps the workbench reports: the bounce-utilisation
    map over the sole, and the playability window over the delivery sweep.
    Cells holding no data are painted as empty rather than as zero.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build an empty map.

        Args:
            title: Heading painted above the grid.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._title = str(title)
        self._caption = ""
        self._values: np.ndarray = np.zeros((0, 0), dtype=float)
        self._mask: np.ndarray | None = None
        self.setMinimumHeight(_MIN_MAP_HEIGHT_PX)

    @property
    def title(self) -> str:
        """The heading painted above the grid."""
        return self._title

    @property
    def caption(self) -> str:
        """The line painted under the grid."""
        return self._caption

    @property
    def values(self) -> np.ndarray:
        """A copy of the grid currently displayed."""
        return self._values.copy()

    def clear(self) -> None:
        """Drop the grid, so nothing stale is painted after a refusal."""
        self._values = np.zeros((0, 0), dtype=float)
        self._mask = None
        self._caption = ""
        self.update()

    def set_grid(
        self,
        values: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        caption: str = "",
    ) -> None:
        """Display a grid.

        Args:
            values: ``(n, m)`` array; NaN marks a cell with no data.
            mask: Optional ``(n, m)`` boolean overlay, outlined in green.
            caption: Line painted under the grid.

        Raises:
            ValueError: If the array is not two-dimensional, or the mask does
                not match it.
        """
        grid = np.asarray(values, dtype=float)
        if grid.ndim != 2:
            raise ValueError(f"a grid map needs a 2-D array, got {grid.ndim}D")
        if mask is not None and np.shape(mask) != grid.shape:
            raise ValueError(
                f"mask shape {np.shape(mask)} does not match grid shape {grid.shape}"
            )
        self._values = grid
        self._mask = None if mask is None else np.asarray(mask, dtype=bool)
        self._caption = str(caption)
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 - Qt API
        """Paint the heading, the cells and the caption."""
        painter = QPainter(self)
        try:
            metrics = painter.fontMetrics()
            line_px = metrics.height()
            painter.drawText(4, line_px, self._title)
            if self._values.size == 0:
                painter.drawText(4, 2 * line_px + 4, "no data")
                return
            self._paint_cells(painter, top_px=line_px + 6, line_px=line_px)
            if self._caption:
                painter.drawText(4, self.height() - 4, self._caption)
        finally:
            painter.end()

    def _paint_cells(self, painter: QPainter, *, top_px: int, line_px: int) -> None:
        """Paint the grid body between the heading and the caption."""
        rows, columns = self._values.shape
        bottom_px = self.height() - (line_px + 6 if self._caption else 4)
        height_px = max(bottom_px - top_px, 1)
        width_px = max(self.width() - 8, 1)
        cell_w = width_px / columns
        cell_h = height_px / rows
        finite = np.isfinite(self._values)
        peak = float(np.nanmax(self._values)) if finite.any() else 0.0
        floor = float(np.nanmin(self._values)) if finite.any() else 0.0
        span = peak - floor
        for row in range(rows):
            for column in range(columns):
                x = int(4 + column * cell_w)
                y = int(top_px + row * cell_h)
                w = max(int(cell_w) - _CELL_GAP_PX, 1)
                h = max(int(cell_h) - _CELL_GAP_PX, 1)
                value = float(self._values[row, column])
                colour = (
                    _EMPTY_CELL
                    if not np.isfinite(value)
                    else _ramp((value - floor) / span if span > 0.0 else 1.0)
                )
                painter.fillRect(x, y, w, h, colour)
                if self._mask is not None and bool(self._mask[row, column]):
                    painter.setPen(_WINDOW_EDGE)
                    painter.drawRect(x, y, w, h)
                    painter.setPen(Qt.GlobalColor.black)


def _ramp(fraction: float) -> QColor:
    """Map ``fraction`` in ``[0, 1]`` onto a pale-sand to burnt-orange ramp.

    Args:
        fraction: Position on the ramp; clamped into range.

    Returns:
        The cell colour.
    """
    position = min(max(float(fraction), 0.0), 1.0)
    return QColor(
        int(247 - 88 * position), int(233 - 148 * position), int(200 - 168 * position)
    )


class DesignPanel(QGroupBox):
    """The W2 wedge parameter set for one candidate sole.

    The controls are the actual design vector -- grind preset, loft, marketed
    bounce, sole width, sole entry height, leading-edge radius, camber area
    and heel/toe relief -- and each one is wired to the geometry the solver
    integrates over.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        title: str,
        default_name: str,
        preset: str,
        parent: QWidget | None = None,
    ) -> None:
        """Build the panel and load a preset into it.

        Args:
            title: Group-box heading.
            default_name: Initial design name.
            preset: Grind preset to load.
            parent: Parent widget.
        """
        super().__init__(title, parent)
        form = QFormLayout(self)
        self._name = QLineEdit(default_name)
        form.addRow("Name:", self._name)

        self._preset = QComboBox()
        self._preset.addItems(grind_preset_names())
        self._preset.setCurrentText(preset)
        form.addRow("Grind preset:", self._preset)

        self._loft = _spin(" deg", 40.0, 66.0, 0.5, 2)
        self._bounce = _spin(" deg", 0.0, 20.0, 0.5, 2)
        self._sole_width = _spin(" mm", 8.0, 26.0, 0.5, 2)
        self._entry_height = _spin(" mm", 1.5, 7.5, 0.1, 2)
        self._leading_radius = _spin(" mm", 2.0, 12.0, 0.25, 2)
        self._camber_area = _spin(" mm^2", 20.0, 70.0, 1.0, 1)
        self._heel_relief = _spin("", 0.0, 0.6, 0.01, 3)
        self._toe_relief = _spin("", 0.0, 0.6, 0.01, 3)
        for label, widget in (
            ("Loft:", self._loft),
            ("Bounce (marketed):", self._bounce),
            ("Sole width (d1):", self._sole_width),
            ("Sole entry height (d3):", self._entry_height),
            ("Leading-edge radius:", self._leading_radius),
            ("Camber area:", self._camber_area),
            ("Heel relief:", self._heel_relief),
            ("Toe relief:", self._toe_relief),
        ):
            form.addRow(label, widget)
            widget.valueChanged.connect(self.changed)
        self._preset.currentTextChanged.connect(self.load_preset)
        self._name.textChanged.connect(self.changed)
        self.load_preset(preset)

    def load_preset(self, name: str) -> None:
        """Load a named grind's numbers into the controls.

        Args:
            name: The preset name.
        """
        geometry = get_preset(name).geometry
        for widget, value in (
            (self._loft, geometry.loft_deg),
            (self._bounce, geometry.marketed_bounce.angle_deg),
            (self._sole_width, geometry.sole_width_m * 1e3),
            (self._entry_height, geometry.entry_height_m * 1e3),
            (self._leading_radius, geometry.leading_edge_radius_m * 1e3),
            (self._camber_area, geometry.sole_camber_area_m2 * 1e6),
            (self._heel_relief, geometry.heel_relief_fraction),
            (self._toe_relief, geometry.toe_relief_fraction),
        ):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        self.changed.emit()

    def design(self) -> WedgeDesign:
        """Read the controls into a design.

        Returns:
            The candidate sole the panel describes.

        Raises:
            WorkbenchInputError: If the name is blank.
        """
        return WedgeDesign(
            name=self._name.text().strip(),
            grind_preset=self._preset.currentText(),
            loft_deg=self._loft.value(),
            marketed_bounce_deg=self._bounce.value(),
            sole_width_mm=self._sole_width.value(),
            entry_height_mm=self._entry_height.value(),
            leading_edge_radius_mm=self._leading_radius.value(),
            camber_area_mm2=self._camber_area.value(),
            heel_relief_fraction=self._heel_relief.value(),
            toe_relief_fraction=self._toe_relief.value(),
        )


class ConditionPanel(QWidget):
    """The W3 sand condition, the swing condition and the study settings."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel with the documented defaults."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        defaults = SwingSetup()
        study = SolverSetup()

        sand_box = QGroupBox("Sand (W3)")
        sand_form = QFormLayout(sand_box)
        self._condition = QComboBox()
        self._condition.addItems(playing_condition_names())
        sand_form.addRow("Playing condition:", self._condition)
        self._firmness = _spin(" kg/cm^2", 1.2, 3.2, 0.1, 2, 2.4)
        sand_form.addRow("Penetrometer firmness:", self._firmness)
        layout.addWidget(sand_box)

        swing_box = QGroupBox("Swing")
        swing_form = QFormLayout(swing_box)
        self._speed = _spin(" m/s", 10.0, 40.0, 0.5, 1, defaults.clubhead_speed_mps)
        self._attack = _spin(" deg", -20.0, -0.5, 0.5, 1, defaults.attack_angle_deg)
        self._face_open = _spin(" deg", 0.0, 40.0, 1.0, 1, defaults.face_open_deg)
        self._shaft_lean = _spin(" deg", -10.0, 25.0, 0.5, 1, defaults.shaft_lean_deg)
        self._entry = _spin(
            " mm", 10.0, 200.0, 5.0, 1, defaults.entry_distance_behind_ball_m * 1e3
        )
        self._ball_depth = _spin(
            " mm", -10.0, 20.0, 1.0, 1, defaults.ball_depth_m * 1e3
        )
        for label, widget in (
            ("Clubhead speed:", self._speed),
            ("Attack angle:", self._attack),
            ("Face open:", self._face_open),
            ("Shaft lean:", self._shaft_lean),
            ("Entry behind ball:", self._entry),
            ("Ball depth in sand:", self._ball_depth),
        ):
            swing_form.addRow(label, widget)
        self._dynamic = QCheckBox("DRFT inertial term active")
        self._dynamic.setChecked(True)
        self._dynamic.setToolTip(
            "Unchecking this gives quasi-static RFT. Above Fr ~ 1 the envelope "
            "refuses to report a force at all, which is what a quasi-static "
            "solver deserves at bunker-shot speeds."
        )
        swing_form.addRow(self._dynamic)
        layout.addWidget(swing_box)

        study_box = QGroupBox("Study")
        study_form = QFormLayout(study_box)
        self._target_carry = _spin(" m", 2.0, 40.0, 0.5, 1, study.target_carry_m)
        study_form.addRow("Target carry:", self._target_carry)
        self._tolerance = _spin("", 0.02, 0.5, 0.01, 2, study.carry_tolerance_fraction)
        study_form.addRow("Carry tolerance:", self._tolerance)
        self._grid = QSpinBox()
        self._grid.setRange(2, 9)
        self._grid.setValue(study.playability_points)
        study_form.addRow("Playability grid (n x n):", self._grid)
        self._stations = QSpinBox()
        self._stations.setRange(5, 25)
        self._stations.setValue(study.n_stations)
        study_form.addRow("Mesh stations:", self._stations)
        self._profile_points = QSpinBox()
        self._profile_points.setRange(12, 48)
        self._profile_points.setValue(study.n_profile_points)
        study_form.addRow("Sole samples:", self._profile_points)
        layout.addWidget(study_box)

        for widget in (
            self._firmness,
            self._speed,
            self._attack,
            self._face_open,
            self._shaft_lean,
            self._entry,
            self._ball_depth,
            self._target_carry,
            self._tolerance,
        ):
            widget.valueChanged.connect(self.changed)
        self._condition.currentTextChanged.connect(self.changed)
        self._dynamic.toggled.connect(self.changed)

    def sand_condition(self) -> SandCondition:
        """Read the sand controls.

        Returns:
            The playing condition and its firmness override.
        """
        return SandCondition(
            preset=self._condition.currentText(),
            firmness_kg_per_cm2=self._firmness.value(),
        )

    def swing_setup(self) -> SwingSetup:
        """Read the swing controls.

        Returns:
            The delivery.

        Raises:
            WorkbenchInputError: If the delivery is not usable.
        """
        return SwingSetup(
            clubhead_speed_mps=self._speed.value(),
            attack_angle_deg=self._attack.value(),
            face_open_deg=self._face_open.value(),
            shaft_lean_deg=self._shaft_lean.value(),
            entry_distance_behind_ball_m=self._entry.value() * 1e-3,
            ball_depth_m=self._ball_depth.value() * 1e-3,
            dynamic_terms_active=self._dynamic.isChecked(),
        )

    def solver_setup(self) -> SolverSetup:
        """Read the discretisation and study controls.

        Returns:
            The settings.

        Raises:
            WorkbenchInputError: If a setting is out of range.
        """
        return SolverSetup(
            n_profile_points=self._profile_points.value(),
            n_stations=self._stations.value(),
            target_carry_m=self._target_carry.value(),
            carry_tolerance_fraction=self._tolerance.value(),
            playability_points=self._grid.value(),
        )


def _spin(
    suffix: str,
    minimum: float,
    maximum: float,
    step: float,
    decimals: int,
    value: float | None = None,
) -> QDoubleSpinBox:
    """Build a configured spin box.

    Args:
        suffix: Unit shown after the number.
        minimum: Lower bound.
        maximum: Upper bound.
        step: Single-step increment.
        decimals: Displayed precision.
        value: Initial value; defaults to the minimum.

    Returns:
        The spin box.
    """
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setValue(minimum if value is None else float(value))
    return box
