"""Qt input and display widgets for the BunkerShot3D workbench (issue #8618).

Every widget here is a *view*: it reads and writes the value objects in
:mod:`src.tools.bunker_shot_gui.design` and paints what
:mod:`src.tools.bunker_shot_gui.model` computed. None of them does physics,
and none of them draws anything it was not handed -- there is no procedural
preview anywhere in this package.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bunkershot3d.geometry import get_preset
from bunkershot3d.solvers import EnvelopeStatus
from src.shared.python.ui.qt import MplCanvas

from .design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
    grind_preset_names,
    playing_condition_names,
)
from .field import ContactPatch, LoadComponent, LoadScale, SoleLoadField
from .render import ShotFrameArtists, field_scales, viewport_fallback
from .report import status_colour, status_headline

__all__ = [
    "ConditionPanel",
    "DesignPanel",
    "GridMapWidget",
    "SoleLoadFieldWidget",
    "VerdictBanner",
]

_EMPTY_CELL = QColor(238, 234, 226)
_WINDOW_EDGE = QColor(20, 90, 40)
_MIN_MAP_HEIGHT_PX = 150
_CELL_GAP_PX = 1
_FRAME_INTERVAL_MS = 120
"""Transport period. Measured: mutating the artists costs 0.75 ms, and the
canvas render they sit in costs about 98 ms at the shipped discretization, so
the interval is set above the renderer's own cost rather than below it. A
faster timer would not produce a faster animation, only dropped ticks."""
_MIN_FIELD_HEIGHT_PX = 380


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
        self._limits: tuple[float, float] | None = None
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

    @property
    def limits(self) -> tuple[float, float] | None:
        """The pinned colour limits, or ``None`` when the grid self-scales."""
        return self._limits

    def clear(self) -> None:
        """Drop the grid, so nothing stale is painted after a refusal."""
        self._values = np.zeros((0, 0), dtype=float)
        self._mask = None
        self._limits = None
        self._caption = ""
        self.update()

    def set_grid(
        self,
        values: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        caption: str = "",
        limits: tuple[float, float] | None = None,
    ) -> None:
        """Display a grid.

        Args:
            values: ``(n, m)`` array; NaN marks a cell with no data.
            mask: Optional ``(n, m)`` boolean overlay, outlined in green.
            caption: Line painted under the grid.
            limits: Optional ``(low, high)`` pinning the colour ramp. Without
                it each grid is stretched to its own extremes, which is right
                for a single map and wrong for two that are meant to be
                compared: two grinds each normalised to their own peak look
                identical whatever the difference between them.

        Raises:
            ValueError: If the array is not two-dimensional, the mask does not
                match it, or the limits are not increasing.
        """
        grid = np.asarray(values, dtype=float)
        if grid.ndim != 2:
            raise ValueError(f"a grid map needs a 2-D array, got {grid.ndim}D")
        if mask is not None and np.shape(mask) != grid.shape:
            raise ValueError(
                f"mask shape {np.shape(mask)} does not match grid shape {grid.shape}"
            )
        if limits is not None and not limits[0] < limits[1]:
            raise ValueError(
                f"colour limits must increase, got {limits[0]} to {limits[1]}"
            )
        self._values = grid
        self._mask = None if mask is None else np.asarray(mask, dtype=bool)
        self._limits = None if limits is None else (float(limits[0]), float(limits[1]))
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
        if self._limits is not None:
            floor, peak = self._limits
        else:
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


class SoleLoadFieldWidget(QWidget):
    """The per-element sole load, animated across the shot (#8705, #8707).

    The workbench's other spatial view, :class:`GridMapWidget`, shows the
    strike summed over time and binned onto 12x12 cells. This one shows the
    field the solver actually produced: one value per surface element, one
    frame per sample, with the depth-linear and inertial terms side by side
    and the contact patch tracked against the leading edge.

    It does no physics and no drawing of its own. The arithmetic is
    :mod:`~src.tools.bunker_shot_gui.field`'s and the drawing is
    :mod:`~src.tools.bunker_shot_gui.render`'s, which is why the same figure
    can be produced in a headless test. This class owns only the transport:
    a frame, a slider, a timer and a play button.

    Two behaviours are deliberate:

    * **The colour scale is injected, not inferred.** A comparison hands both
      views one set of scales, so the two panels are directly readable against
      each other. A view left to its own devices scales to its own shot, which
      is correct for a single design and wrong for two.
    * **Clearing stops playback.** A refused query must not leave a sole
      animating under a red banner.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build an empty view.

        Args:
            title: Heading shown above the canvas.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._title = str(title)
        self._field: SoleLoadField | None = None
        self._patch: ContactPatch | None = None
        self._scales: dict[LoadComponent, LoadScale] | None = None
        self._artists: ShotFrameArtists | None = None
        self._frame = 0
        self._fallback = viewport_fallback()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._heading = QLabel(self._title)
        layout.addWidget(self._heading)
        self._canvas = MplCanvas(width=9.0, height=5.5, dpi=96)
        self._canvas.setMinimumHeight(_MIN_FIELD_HEIGHT_PX)
        layout.addWidget(self._canvas)

        transport = QHBoxLayout()
        self._play_button = QPushButton("Play")
        self._play_button.clicked.connect(self.toggle_play)
        transport.addWidget(self._play_button)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider)
        transport.addWidget(self._slider, stretch=1)
        self._readout = QLabel("no shot")
        transport.addWidget(self._readout)
        layout.addLayout(transport)

        # The full degradation reason names three optional packages and their
        # install hints, which is four lines of text beside a figure. The
        # label states the renderer actually in use; the reason is a hover
        # away rather than a paragraph the designer reads once and never again.
        self._note = QLabel(
            f"Renderer: {self._fallback.renderer}"
            + (" (no 3-D viewport installed)" if self._fallback.degraded else "")
        )
        self._note.setToolTip(self._fallback.describe())
        layout.addWidget(self._note)

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self.advance)

    # ------------------------------------------------------------ accessors

    @property
    def title(self) -> str:
        """The heading shown above the canvas."""
        return self._title

    @property
    def has_shot(self) -> bool:
        """Whether a field is loaded."""
        return self._field is not None

    @property
    def n_frames(self) -> int:
        """Number of samples in the loaded shot; zero when empty."""
        return 0 if self._field is None else self._field.n_frames

    @property
    def frame_index(self) -> int:
        """The sample currently displayed."""
        return self._frame

    @property
    def is_playing(self) -> bool:
        """Whether the transport is running."""
        return self._timer.isActive()

    @property
    def scales(self) -> dict[LoadComponent, LoadScale] | None:
        """The fixed colour scales in force, or ``None`` when empty."""
        return self._scales

    @property
    def renderer_note(self) -> str:
        """What the ADR-0027 viewport layer left this view drawing with."""
        return self._fallback.describe()

    # --------------------------------------------------------------- content

    def set_shot(
        self,
        field: SoleLoadField,
        patch: ContactPatch | None = None,
        *,
        scales: dict[LoadComponent, LoadScale] | None = None,
    ) -> None:
        """Load one shot and open on the moment the sole carried most.

        Args:
            field: The per-element load field.
            patch: The contact-patch series, when there is one.
            scales: Fixed colour scales shared with any other view this one is
                compared against. Defaults to this field's own.

        Raises:
            ValueError: If the patch does not describe the same shot as the
                field; the drawing layer refuses the pair.
        """
        self.pause()
        self._field = field
        self._patch = patch
        self._scales = field_scales((field,)) if scales is None else scales
        # The axes are built once here, not once per frame: rebuilding the
        # colour bars and re-running the layout pass costs about 250 ms at the
        # shipped discretization, which is slower than the transport interval.
        self._artists = ShotFrameArtists(self._canvas.fig, field, patch, self._scales)
        self._frame = int(field.resultant_force_N(LoadComponent.TOTAL).argmax())
        self._slider.blockSignals(True)
        self._slider.setRange(0, field.n_frames - 1)
        self._slider.setValue(self._frame)
        self._slider.blockSignals(False)
        self._redraw()

    def clear(self) -> None:
        """Drop the shot and stop playing, so nothing stale keeps moving."""
        self.pause()
        self._field = None
        self._patch = None
        self._scales = None
        self._artists = None
        self._frame = 0
        self._slider.blockSignals(True)
        self._slider.setRange(0, 0)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._readout.setText("no shot")
        self._canvas.fig.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------- transport

    def set_frame(self, frame: int) -> None:
        """Show one sample.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If there is no shot, or the index is outside it. A
                wrapped or clamped index would leave the readout describing a
                different moment from the one drawn.
        """
        if self._field is None:
            raise ValueError("there is no shot loaded, so no frame can be shown")
        if not 0 <= int(frame) < self._field.n_frames:
            raise ValueError(
                f"frame {frame} is outside the shot, which has "
                f"{self._field.n_frames} samples"
            )
        self._frame = int(frame)
        self._slider.blockSignals(True)
        self._slider.setValue(self._frame)
        self._slider.blockSignals(False)
        self._redraw()

    def advance(self) -> None:
        """Step one sample forward, wrapping at the end of the shot."""
        if self._field is None:
            return
        self.set_frame((self._frame + 1) % self._field.n_frames)

    def play(self) -> None:
        """Start the transport, unless there is nothing to play."""
        if self._field is None:
            return
        self._timer.start()
        self._play_button.setText("Pause")

    def pause(self) -> None:
        """Stop the transport."""
        self._timer.stop()
        self._play_button.setText("Play")

    def toggle_play(self) -> None:
        """Play if paused, pause if playing."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    # ------------------------------------------------------------- rendering

    def _on_slider(self, value: int) -> None:
        """Follow the slider."""
        if self._field is not None:
            self.set_frame(int(value))

    def _redraw(self) -> None:
        """Repaint the canvas at the current frame."""
        if self._field is None or self._artists is None:
            return
        self._artists.update(self._frame)
        self._canvas.draw_idle()
        self._readout.setText(
            f"sample {self._frame + 1} of {self._field.n_frames}  "
            f"({self._field.time_s[self._frame] * 1e3:.2f} ms)"
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
