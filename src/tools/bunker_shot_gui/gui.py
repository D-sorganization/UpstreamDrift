"""BunkerShot3D designer workbench (issue #8618, W11 of epic #8607).

The single user-facing surface of the BunkerShot3D club-design tool. It runs
the **real F0 solver** -- dynamic 3D Resistive Force Theory, the default tier
of ADR-0032 -- over a parametric wedge sole in a USGA-referenced sand bed, and
answers the question the tool exists for:

    given two wedge sole geometries, which one performs better, in what
    conditions, and how confident are we?

Everything numeric lives in :mod:`src.tools.bunker_shot_gui.model` and
:mod:`src.tools.bunker_shot_gui.report`, both of which import no GUI toolkit.
This module is the Qt shell around them.

Two behaviours are deliberate and must not be softened:

* **The verdict is shown before the numbers, always.** A greenside bunker shot
  sits roughly 60x outside 3D-RFT's stated Froude limit, so every result is
  labelled with how far outside the calibrated envelope it was produced.
* **A refusal shows no number.** When the solver declines a query the banner
  turns red, the maps are cleared and the report states why. There is no code
  path in this package that paints a force beside a ``REFUSED`` verdict.

Running a design costs a second or two; the cross-tier check costs minutes.
Both used to block the interface behind a wait cursor alone. They now run
off the GUI thread through :mod:`src.tools.async_action` (issue #8880): the
triggering button (and its siblings) is disabled for the duration of the
run, which is what keeps a result from arriving out of order with its
inputs -- the concern a wait cursor alone did not address. ``run_design_a``,
``run_comparison`` and ``run_cross_tier`` stay the synchronous, testable
core; ``run_design_a_async`` etc. are what the buttons now trigger.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bunkershot3d.fields.store import load_field
from bunkershot3d.solvers import EnvelopeStatus

from src.launchers.help_menu import build_help_menu
from src.shared.python.ui import HoverCopyTextBrowser  # type: ignore[attr-defined]
from src.tools.async_action import AsyncActionBar, WorkerContext

from .crosstier import CrossTierComparison
from .crosstier_run import cross_tier_check
from .design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
    WorkbenchInputError,
)
from .export import (
    ExportedFigure,
    evaluation_stamp,
    export_figure,
    load_workbench_evaluation,
    save_workbench_evaluation,
)
from .field import LoadComponent, LoadScale, SoleLoadField
from .model import DesignEvaluation, WorkbenchComparison, WorkbenchModel
from .render import field_scales, sole_load_still
from .render3d import SceneScale, scene_scale
from .report import comparison_report, evaluation_report
from .shot3d import ShotScene
from .slices import CursorMap
from .viewport_widgets import (
    CrossTierWidget,
    SandSliceWidget,
    ShotViewportWidget,
    TracePanelWidget,
)
from .widgets import (
    ConditionPanel,
    DesignPanel,
    GridMapWidget,
    SoleLoadFieldWidget,
    VerdictBanner,
)

__all__ = [
    "BunkerShotWidget",
    "BunkerShotWindow",
    "get_dockable_ui",
]

logger = logging.getLogger(__name__)

_CROSS_TIER_BUSY_BANNER = (
    "Running the F1 continuum beside F0 on design A. This is minutes, "
    "not milliseconds: F1 has no shot history yet, so every probe is "
    "a separate march to one recorded pose."
)

_IDLE_TEXT = (
    "BunkerShot3D designer workbench (Preview)\n"
    "=========================================\n\n"
    "Set the sole parameters, the playing condition and the delivery, then\n"
    "run design A or compare A against B.\n\n"
    "Every result carries a validity verdict. At greenside delivery speeds\n"
    "3D-RFT is roughly 60x outside its stated Froude limit and about 20x\n"
    "beyond any published validation, so the verdict is part of the answer,\n"
    "not a disclaimer attached to it. When the solver refuses a query no\n"
    "force, depth or carry is reported at all.\n"
)

_SAND_FIELD_HINT = (
    "No sand field loaded. F1 fields are computed offline -- a march at "
    "ADR-0033's bulk resolution takes tens of seconds, so running one here "
    "would freeze the workbench and running a coarser one would put a "
    "picture on screen at a resolution nobody chose. Capture one with "
    "capture_f1_field, store it with save_field, and open it here. The "
    "field carries its own tier and validity status, checked against the "
    "file's digest on load."
)

_STALE_FIELD_HINT = (
    "Sand field cleared: it was cut from a different run. A cut beside a "
    "freshly evaluated design would read as that design's sand, which it "
    "is not. Load the field belonging to this run."
)

ModelFactory = Callable[[SolverSetup], WorkbenchModel]

_ResultT = TypeVar("_ResultT")


def _fields(
    evaluations: tuple[DesignEvaluation, ...],
) -> tuple[SoleLoadField, ...]:
    """Return the load fields of the evaluations that produced one."""
    return tuple(
        evaluation.shot.sole_field
        for evaluation in evaluations
        if evaluation.shot.sole_field is not None
    )


def _shared_scales(
    evaluations: tuple[DesignEvaluation, ...],
) -> dict[LoadComponent, LoadScale] | None:
    """Return the one set of colour scales every supplied design is drawn on.

    Args:
        evaluations: The designs about to be painted together.

    Returns:
        The merged scales, or ``None`` when no design produced a field.
    """
    fields = _fields(evaluations)
    return field_scales(fields) if fields else None


def _scenes(
    evaluations: tuple[DesignEvaluation, ...],
) -> tuple[ShotScene, ...]:
    """Return every 3-D scene among the supplied designs.

    Args:
        evaluations: The designs about to be drawn together.

    Returns:
        The scenes, skipping any design whose shot did not produce one.
    """
    return tuple(
        scene
        for evaluation in evaluations
        if (scene := evaluation.shot.scene) is not None
    )


def _shared_scene_scale(
    evaluations: tuple[DesignEvaluation, ...],
) -> SceneScale | None:
    """Return the one world box every supplied scene is drawn in.

    Args:
        evaluations: The designs about to be drawn together.

    Returns:
        The covering box, or ``None`` when no design produced a scene. The
        same argument as the colour scales: two divots each framed to their
        own extent look like the same divot.
    """
    scenes = _scenes(evaluations)
    return scene_scale(scenes) if scenes else None


def _density_limits(
    evaluations: tuple[DesignEvaluation, ...],
) -> tuple[float, float] | None:
    """Return the one impulse-density ramp every supplied map is drawn on.

    Args:
        evaluations: The designs about to be painted together.

    Returns:
        ``(0, peak)`` over all of them, or ``None`` when no map exists or the
        maps are empty -- in which case the widget falls back to its own
        stretch, which for a single empty map is harmless.
    """
    maps = [
        evaluation.shot.sole_load
        for evaluation in evaluations
        if evaluation.shot.sole_load is not None
    ]
    peaks = [sole_load.peak_density_pa_s for sole_load in maps]
    top = max(peaks, default=0.0)
    return (0.0, top) if top > 0.0 else None


class BunkerShotWidget(QWidget):
    """The workbench: two candidate soles, one condition, one verdict."""

    evaluation_changed = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        """Build the workbench.

        Args:
            parent: Parent widget.
            model_factory: Builds the headless model from the study settings.
                Injected so a test can substitute a cheap stand-in; the
                production path is :class:`~.model.WorkbenchModel` itself.
        """
        super().__init__(parent)
        self._model_factory: ModelFactory = model_factory or WorkbenchModel
        self._last_evaluation: DesignEvaluation | None = None
        self._last_comparison: WorkbenchComparison | None = None
        self._loaded_record: dict[str, Any] | None = None
        self._build_ui()
        self.update_control_states()

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        """Assemble the controls on the left and the results on the right."""
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_results())
        splitter.setSizes([420, 780])
        layout.addWidget(splitter)

    def _build_controls(self) -> QWidget:
        """Build the scrollable input column."""
        inner = QWidget()
        column = QVBoxLayout(inner)

        title = QLabel("BunkerShot3D Designer Workbench")
        font = title.font()
        font.setPointSize(13)
        font.setBold(True)
        title.setFont(font)
        column.addWidget(title)
        column.addWidget(
            QLabel("F0 tier: dynamic 3D Resistive Force Theory (ADR-0032)")
        )

        self._design_a = DesignPanel("Design A (W2 sole parameters)", "A", "sm9_58_m")
        self._design_b = DesignPanel("Design B (W2 sole parameters)", "B", "sm9_54_f")
        column.addWidget(self._design_a)
        column.addWidget(self._design_b)

        self._conditions = ConditionPanel()
        column.addWidget(self._conditions)

        self._run_button = QPushButton("Run design A")
        self._run_button.clicked.connect(self.run_design_a_async)
        column.addWidget(self._run_button)

        self._compare_button = QPushButton("Compare A vs B")
        self._compare_button.clicked.connect(self.run_comparison_async)
        column.addWidget(self._compare_button)

        # Separate button, and separate for a reason rather than for tidiness.
        # The cross-tier check marches the F1 continuum once per probe -- F1
        # has no shot history yet (#8733) -- so it costs minutes where a
        # design costs milliseconds. Running it on every evaluation would
        # make the workbench unusable, and running it silently would make
        # the wait inexplicable.
        self._cross_tier_button = QPushButton("Cross-tier check A (F0 vs F1)")
        self._cross_tier_button.setToolTip(
            "Puts the F1 plane-strain MPM continuum beside F0 on design A. "
            "Minutes, not milliseconds: F1 has no shot history yet, so each "
            "probe is a separate march to one recorded pose. Consistency "
            "between two uncalibrated models is not validation."
        )
        self._cross_tier_button.clicked.connect(self.run_cross_tier_async)
        column.addWidget(self._cross_tier_button)

        self._export_button = QPushButton("Export views…")
        self._export_button.clicked.connect(self.choose_export_views)
        column.addWidget(self._export_button)

        self._save_button = QPushButton("Save evaluation…")
        self._save_button.clicked.connect(self.choose_save_evaluation)
        column.addWidget(self._save_button)

        self._load_button = QPushButton("Load evaluation…")
        self._load_button.clicked.connect(self.choose_load_evaluation)
        column.addWidget(self._load_button)

        self.action_bar = AsyncActionBar()
        self.action_bar.set_trigger_buttons(
            self._run_button, self._compare_button, self._cross_tier_button
        )
        column.addWidget(self.action_bar)
        column.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _build_results(self) -> QWidget:
        """Build the verdict banner, the two map columns and the report."""
        panel = QWidget()
        column = QVBoxLayout(panel)

        self._banner = VerdictBanner()
        column.addWidget(self._banner)

        maps_page = QWidget()
        maps = QGridLayout(maps_page)
        self._bounce_map_a = GridMapWidget("A: bounce utilisation")
        self._bounce_map_b = GridMapWidget("B: bounce utilisation")
        self._window_map_a = GridMapWidget("A: playability window")
        self._window_map_b = GridMapWidget("B: playability window")
        maps.addWidget(self._bounce_map_a, 0, 0)
        maps.addWidget(self._bounce_map_b, 0, 1)
        maps.addWidget(self._window_map_a, 1, 0)
        maps.addWidget(self._window_map_b, 1, 1)

        fields_page = QWidget()
        fields = QHBoxLayout(fields_page)
        self._field_a = SoleLoadFieldWidget("A: sole load and contact patch")
        self._field_b = SoleLoadFieldWidget("B: sole load and contact patch")
        fields.addWidget(self._field_a)
        fields.addWidget(self._field_b)

        shot_page = QWidget()
        shot = QHBoxLayout(shot_page)
        self._scene_a = ShotViewportWidget("A: the head through the sand")
        self._traces_a = TracePanelWidget("A: traces on the shared cursor")
        shot.addWidget(self._scene_a, stretch=3)
        shot.addWidget(self._traces_a, stretch=2)
        # One transport drives all three views of design A (#8706, #8708).
        # The field view already owns a slider, a timer and a play button;
        # a second one beside the scene would let the views drift apart,
        # which is the one thing linking them exists to prevent.
        self._field_a.link(self._scene_a)
        self._field_a.link(self._traces_a)

        cross_tier_page = QWidget()
        cross_tier = QHBoxLayout(cross_tier_page)
        self._cross_tier = CrossTierWidget("A: F0 against F1, on the shared cursor")
        cross_tier.addWidget(self._cross_tier)
        # The fourth follower of the one transport (#8705, #8706, #8708).
        self._field_a.link(self._cross_tier)

        slice_page = QWidget()
        cut = QVBoxLayout(slice_page)
        controls = QHBoxLayout()
        self._load_field_button = QPushButton("Load sand field (F1)...")
        self._load_field_button.clicked.connect(self.choose_sand_field)
        controls.addWidget(self._load_field_button)
        self._field_note = QLabel(_SAND_FIELD_HINT)
        self._field_note.setWordWrap(True)
        controls.addWidget(self._field_note, stretch=1)
        cut.addLayout(controls)
        self._slice_a = SandSliceWidget("A: sand cut through the impact zone")
        cut.addWidget(self._slice_a, stretch=1)
        # Another view on the same transport (#8711). Its own record is a
        # strided F1 march rather than the F0 shot, so it maps the cursor
        # rather than taking a slider of its own.
        self._field_a.link(self._slice_a)

        self._views = QTabWidget()
        self._views.addTab(maps_page, "Summed maps")
        self._views.addTab(fields_page, "Sole load field (per element, per sample)")
        self._views.addTab(shot_page, "Shot in 3-D and traces")
        self._views.addTab(cross_tier_page, "Cross-tier check (F0 vs F1)")
        self._views.addTab(slice_page, "Sand cut (F1 field)")
        column.addWidget(self._views, stretch=3)

        self._results = HoverCopyTextBrowser()
        self._results.setReadOnly(True)
        self._results.setFont(QFont("Consolas", 9))
        self._results.setLineWrapMode(HoverCopyTextBrowser.LineWrapMode.NoWrap)
        self._results.setPlainText(_IDLE_TEXT)
        column.addWidget(self._results, stretch=3)
        return panel

    # -------------------------------------------------------------- accessors

    @property
    def report_text(self) -> str:
        """The report currently displayed."""
        return self._results.toPlainText()

    @property
    def verdict_text(self) -> str:
        """The verdict banner's current text."""
        return self._banner.text()

    # ------------------------------------------------------------------ runs

    def _design_a_inputs(
        self,
    ) -> tuple[SolverSetup, SandCondition, SwingSetup, WedgeDesign] | None:
        """Read the shared controls and design A, reporting a bad combination.

        Returns:
            The four inputs every design-A action needs, or ``None`` when
            one of them could not be read -- in which case the error has
            already been shown and the caller should simply return.
        """
        inputs = self._read_inputs()
        if inputs is None:
            return None
        settings, sand, swing = inputs
        try:
            return (settings, sand, swing, self._design_a.design())
        except WorkbenchInputError as error:
            self._show_input_error(error)
            return None

    def _read_comparison_designs(
        self,
    ) -> tuple[SolverSetup, SandCondition, SwingSetup, WedgeDesign, WedgeDesign] | None:
        """Read shared controls plus designs A and B for a comparison run."""
        inputs = self._read_inputs()
        if inputs is None:
            return None
        settings, sand, swing = inputs
        try:
            left = self._design_a.design()
            right = self._design_b.design()
        except WorkbenchInputError as error:
            self._show_input_error(error)
            return None
        if left.name == right.name:
            self._show_input_error(
                WorkbenchInputError(
                    "designs A and B need different names so the ranking can "
                    f"report them; both are {left.name!r}"
                )
            )
            return None
        return settings, sand, swing, left, right

    def run_design_a(self) -> None:
        """Evaluate design A and display the result."""
        read = self._design_a_inputs()
        if read is None:
            return
        settings, sand, swing, design = read
        self._banner.show_busy("Running the F0 solver over design A...")
        result = self._guarded(
            lambda: self._model_factory(settings).evaluate(design, sand, swing)
        )
        if result is None:
            return
        self.show_evaluation(result)

    def run_comparison(self) -> None:
        """Evaluate both designs and rank them, with uncertainty attached."""
        read = self._read_comparison_designs()
        if read is None:
            return
        settings, sand, swing, left, right = read
        self._banner.show_busy("Running the F0 solver over both designs...")
        result = self._guarded(
            lambda: self._model_factory(settings).compare(left, right, sand, swing)
        )
        if result is None:
            return
        self.show_comparison(result)

    def run_cross_tier(self) -> None:
        """Put F1 beside F0 on design A, on demand.

        Deliberately its own action rather than part of
        :meth:`run_design_a`: the check marches the continuum once per
        probe and costs minutes. The banner says so before the wait rather
        than after it.
        """
        read = self._design_a_inputs()
        if read is None:
            return
        settings, sand, swing, design = read
        self._banner.show_busy(_CROSS_TIER_BUSY_BANNER)
        result = self._guarded(
            lambda: cross_tier_check(self._model_factory(settings), design, sand, swing)
        )
        if result is None:
            return
        self.show_cross_tier(result)

    # ---- async entry points (#8880) --------------------------------------
    #
    # The buttons call these; the synchronous run_* methods above stay the
    # testable core. Each async wrapper reads the (cheap) inputs on the GUI
    # thread, hands the solver call to a worker, and renders the result back
    # on the GUI thread in on_finished. AsyncActionBar disables all three
    # trigger buttons for the duration of a run, so a design's answer can
    # never arrive interleaved with a different one still in flight.

    def run_design_a_async(self) -> None:
        """Evaluate design A off the GUI thread, with progress and cancel."""
        read = self._design_a_inputs()
        if read is None:
            return
        settings, sand, swing, design = read
        self._banner.show_busy("Running the F0 solver over design A...")

        def _work(ctx: WorkerContext) -> DesignEvaluation:
            return self._model_factory(settings).evaluate(design, sand, swing)

        self.action_bar.start(
            "Design A",
            _work,
            on_finished=self.show_evaluation,
            on_failed=self._report_solver_failure,
        )

    def run_comparison_async(self) -> None:
        """Evaluate both designs off the GUI thread, with progress and cancel."""
        read = self._read_comparison_designs()
        if read is None:
            return
        settings, sand, swing, left, right = read
        self._banner.show_busy("Running the F0 solver over both designs...")

        def _work(ctx: WorkerContext) -> WorkbenchComparison:
            return self._model_factory(settings).compare(left, right, sand, swing)

        self.action_bar.start(
            "Compare A vs B",
            _work,
            on_finished=self.show_comparison,
            on_failed=self._report_solver_failure,
        )

    def run_cross_tier_async(self) -> None:
        """Put F1 beside F0 on design A off the GUI thread.

        The clearest win: this check costs minutes on the GUI thread today.
        """
        read = self._design_a_inputs()
        if read is None:
            return
        settings, sand, swing, design = read
        self._banner.show_busy(_CROSS_TIER_BUSY_BANNER)

        def _work(ctx: WorkerContext) -> CrossTierComparison:
            return cross_tier_check(self._model_factory(settings), design, sand, swing)

        self.action_bar.start(
            "Cross-tier check",
            _work,
            on_finished=self.show_cross_tier,
            on_failed=self._report_solver_failure,
        )

    def _report_solver_failure(self, message: str) -> None:
        """Report a worker-thread failure. GUI-thread only (on_failed)."""
        logger.error("bunker workbench solver failed: %s", message)
        self._banner.show_error(message)
        self._results.setPlainText(f"The solver failed unexpectedly.\n\n{message}")

    def show_cross_tier(self, comparison: CrossTierComparison) -> None:
        """Display a cross-tier comparison and open its tab.

        The banner keeps the shot's own verdict. Nothing on this page
        improves it: agreement between two uncalibrated models is not
        validation, and the licence statement inside the figure and the
        status line under it both say so.

        Args:
            comparison: The comparison to show.
        """
        self._banner.show_status(comparison.worst_status)
        self._cross_tier.set_comparison(comparison)
        self._views.setCurrentIndex(self._views.count() - 1)
        self._results.setPlainText(comparison.summary())

    def _read_inputs(
        self,
    ) -> tuple[SolverSetup, SandCondition, SwingSetup] | None:
        """Read the shared condition controls, reporting a bad combination."""
        try:
            return (
                self._conditions.solver_setup(),
                self._conditions.sand_condition(),
                self._conditions.swing_setup(),
            )
        except WorkbenchInputError as error:
            self._show_input_error(error)
            return None

    def _guarded(self, run: Callable[[], _ResultT]) -> _ResultT | None:
        """Run the model under a wait cursor, reporting input errors."""
        application = QApplication.instance()
        if application is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return run()
        except WorkbenchInputError as error:
            self._show_input_error(error)
            return None
        finally:
            if application is not None:
                QApplication.restoreOverrideCursor()

    # -------------------------------------------------------------- rendering

    def show_evaluation(self, evaluation: DesignEvaluation) -> None:
        """Display one evaluated design.

        Args:
            evaluation: The evaluated design.
        """
        self._last_evaluation = evaluation
        self._last_comparison = None
        self._loaded_record = None
        self._banner.show_status(evaluation.shot.status)
        self._results.setPlainText(evaluation_report(evaluation))
        self._paint_maps(
            evaluation,
            self._bounce_map_a,
            self._window_map_a,
            limits=_density_limits((evaluation,)),
        )
        # Before the field view is repainted, not after: painting it emits
        # frame_changed, and a sand cut still linked to the previous shot's
        # record would be handed an index from a record it does not have.
        self._clear_sand_field()
        self._paint_field(evaluation, self._field_a, _shared_scales((evaluation,)))
        self._paint_shot(evaluation, _shared_scene_scale((evaluation,)))
        self._bounce_map_b.clear()
        self._window_map_b.clear()
        self._field_b.clear()
        # A cross-tier check belongs to the shot it was run on. Leaving the
        # old one up beside a new design would put two different shots on
        # one cursor, which is the one thing the shared transport exists to
        # make impossible.
        self._cross_tier.clear()
        self.update_control_states()
        self.evaluation_changed.emit()

    def show_comparison(self, comparison: WorkbenchComparison) -> None:
        """Display an A/B comparison.

        The banner shows the worse of the two verdicts, because a comparison
        is only as trustworthy as its least trustworthy half.

        Args:
            comparison: The two evaluations and their ranking.
        """
        self._last_comparison = comparison
        self._last_evaluation = comparison.left
        self._loaded_record = None
        self._banner.show_status(comparison.worst_status)
        self._results.setPlainText(
            "\n".join(
                (
                    comparison_report(comparison),
                    evaluation_report(comparison.left),
                    evaluation_report(comparison.right),
                )
            )
        )
        # Both halves of a comparison are painted on one ramp and one set of
        # colour scales. Left to themselves each panel stretches to its own
        # extremes, which makes two grinds look identical however far apart
        # they are -- the failure this pair of arguments exists to prevent.
        pair = (comparison.left, comparison.right)
        limits = _density_limits(pair)
        scales = _shared_scales(pair)
        self._paint_maps(
            comparison.left, self._bounce_map_a, self._window_map_a, limits=limits
        )
        self._paint_maps(
            comparison.right, self._bounce_map_b, self._window_map_b, limits=limits
        )
        self._clear_sand_field()
        self._paint_field(comparison.left, self._field_a, scales)
        self._paint_field(comparison.right, self._field_b, scales)
        self._paint_shot(comparison.left, _shared_scene_scale(pair))
        self._cross_tier.clear()
        self.update_control_states()
        self.evaluation_changed.emit()

    def _clear_sand_field(self) -> None:
        """Drop a loaded sand field before a new run is displayed.

        Two reasons, and both matter.

        A cut belongs to the march that produced it, not to whatever was
        evaluated most recently. Leaving it on screen beside a fresh
        design would let it read as that design's sand, which is the
        relabelling issue #8710 spent a content digest preventing --
        undone by a stale widget.

        And it must happen **before** the sole-field view is repainted.
        That view is the transport: painting it emits ``frame_changed``,
        and a cut whose :class:`~.slices.CursorMap` was built against the
        previous shot's record would be handed an index out of a record
        it no longer has. Its ``set_frame`` raises on that, correctly --
        but a Python exception escaping a Qt slot aborts the process
        rather than surfacing, so the ordering here is load-bearing and
        not cosmetic.
        """
        if not self._slice_a.has_shot:
            return
        self._slice_a.clear()
        self._field_note.setText(_STALE_FIELD_HINT)

    def choose_sand_field(self) -> None:
        """Ask for a stored sand field and load it into the cut view."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load a stored F1 sand field",
            "",
            "BunkerShot3D result (*.h5);;All files (*)",
        )
        if path:
            self.load_sand_field(path)

    def load_sand_field(self, path: str) -> None:
        """Load a stored sand field and cut it on the shared cursor.

        Separate from :meth:`choose_sand_field` so the loading can be
        driven without a file dialog -- by a test, or by a script that
        already knows the path.

        The field is **loaded, not computed**. An F1 march at ADR-0033's
        bulk resolution takes tens of seconds; running one inside a
        button handler would freeze the workbench, and running a coarser
        one to keep it responsive would put a picture on screen at a
        resolution nobody chose. So the field is produced offline by
        :func:`~bunkershot3d.fields.capture.capture_f1_field`, stored by
        :func:`~bunkershot3d.fields.store.save_field`, and opened here --
        with the tier and validity status it was written with, which
        :func:`~bunkershot3d.fields.store.load_field` verifies against
        the file's own digest before returning it.

        Args:
            path: Path to a stored result carrying a sand field.
        """
        try:
            series = load_field(path)
        except (OSError, ValueError) as error:
            # A refused field must reach the user as a message. It is not a
            # bug in the workbench: an integrity failure is the loader doing
            # its job, and a traceback in a terminal nobody is reading would
            # end with the tab silently staying empty.
            self._slice_a.clear()
            self._field_note.setText(f"Could not load {path}\n{error}")
            logger.warning("sand field refused: %s", error)
            return
        transport = self._field_a.n_frames or series.n_frames
        self._slice_a.set_shot(
            series,
            cursor=CursorMap(n_transport=transport, n_field=series.n_frames),
        )
        provenance = series.provenance
        self._field_note.setText(
            f"{path}\n{provenance.headline()}\n"
            f"kinematics: {provenance.kinematics}\n"
            f"{series.retention.describe()}"
        )

    def _paint_field(
        self,
        evaluation: DesignEvaluation,
        view: SoleLoadFieldWidget,
        scales: dict[LoadComponent, LoadScale] | None,
    ) -> None:
        """Load one design's per-element field, or clear the view.

        Args:
            evaluation: The evaluated design.
            view: The field view to fill.
            scales: The colour scales both designs share, or ``None`` when
                there is no field to draw.
        """
        field = evaluation.shot.sole_field
        if field is None or scales is None:
            view.clear()
            return
        view.set_shot(field, evaluation.shot.contact_patch, scales=scales)

    def _paint_shot(
        self,
        evaluation: DesignEvaluation,
        scale: SceneScale | None,
    ) -> None:
        """Load one design's 3-D scene and traces, or clear both views.

        Args:
            evaluation: The evaluated design.
            scale: The world box shared with any other design being drawn,
                or ``None`` when there is no scene to draw.
        """
        shot = evaluation.shot
        scene, traces = shot.scene, shot.traces
        if scene is None or scale is None:
            self._scene_a.clear()
        else:
            self._scene_a.set_shot(
                scene,
                scale=scale,
                band=None if traces is None else traces.band,
            )
        if traces is None:
            self._traces_a.clear()
        else:
            self._traces_a.set_shot(traces)

    def _paint_maps(
        self,
        evaluation: DesignEvaluation,
        bounce_map: GridMapWidget,
        window_map: GridMapWidget,
        *,
        limits: tuple[float, float] | None = None,
    ) -> None:
        """Paint one design's two maps, clearing them when there is no data."""
        sole_load = evaluation.shot.sole_load
        if sole_load is None:
            bounce_map.clear()
        else:
            utilisation = sole_load.utilisation
            bounce_map.set_grid(
                sole_load.density_pa_s,
                limits=limits,
                caption=(
                    f"{utilisation.utilisation_fraction:.0%} of the sole carried "
                    f"load; {utilisation.removable_area_m2 * 1e4:.1f} cm^2 removable"
                ),
            )
        playability = evaluation.playability
        window = playability.window
        if window is None or playability.carry_verdict is None:
            window_map.clear()
        else:
            # A carry number never appears without the statement it may be
            # read under (issue #8657): the grid and its verdict travel
            # together on the outcome, and the caption is where that reaches
            # the designer.
            status = playability.carry_verdict.status
            window_map.set_grid(
                playability.carry_m,
                mask=window.in_window,
                caption=(
                    f"carry [{status.value.replace('_', ' ').upper()}] vs attack "
                    f"angle x firmness; window {window.fraction:.0%} of the domain"
                ),
            )

    def _show_input_error(self, error: WorkbenchInputError) -> None:
        """Report an unusable input without pretending it is a verdict."""
        logger.info("bunker workbench input rejected: %s", error)
        self._banner.show_error(str(error))
        self._results.setPlainText(
            "The inputs do not describe a shot that can be solved.\n\n"
            f"{error}\n\n"
            "This is an input error, not a solver verdict: the model was never "
            "asked for a number."
        )
        for widget in (
            self._bounce_map_a,
            self._bounce_map_b,
            self._window_map_a,
            self._window_map_b,
            self._field_a,
            self._field_b,
        ):
            widget.clear()

    def cleanup(self) -> None:
        """Cancel and join any running action (#8880)."""
        self.action_bar.shutdown()

    @property
    def current_evaluation(self) -> DesignEvaluation | None:
        """The evaluation currently on screen, or None if none run."""
        return self._last_evaluation

    def disabled_reasons(self) -> dict[str, str | None]:
        """Each control mapped to None (active) or why it is disabled."""
        reasons: dict[str, str | None] = {
            "run_design_a": None,
            "compare": None,
            "cross_tier": None,
            "load_sand_field": None,
            "load_evaluation": None,
        }
        if self.action_bar.is_running:
            reasons["run_design_a"] = "solver is busy"
            reasons["compare"] = "solver is busy"
            reasons["cross_tier"] = "solver is busy"
        has_eval = self._last_evaluation is not None or self._loaded_record is not None
        no_eval = (
            None
            if has_eval
            else "no evaluation available; run design A or compare A vs B first"
        )
        reasons["export_views"] = no_eval
        reasons["save_evaluation"] = no_eval
        return reasons

    def control_enabled(self, name: str) -> bool:
        """Whether the named control accepts input."""
        reasons = self.disabled_reasons()
        return reasons.get(name) is None

    def update_control_states(self) -> None:
        """Update button enabled states and tooltips from disabled_reasons."""
        reasons = self.disabled_reasons()
        controls = (
            ("export_views", self._export_button, "Export views…"),
            ("save_evaluation", self._save_button, "Save evaluation…"),
            ("load_evaluation", self._load_button, "Load evaluation…"),
        )
        for name, button, default_tip in controls:
            reason = reasons.get(name)
            enabled = reason is None
            button.setEnabled(enabled)
            button.setToolTip(default_tip if enabled else f"Disabled: {reason}")

    def export_views(self, destination_dir: Path | str) -> list[ExportedFigure]:
        """Export stamped figures of the current evaluation to a directory.

        Args:
            destination_dir: Directory where figures will be saved.

        Returns:
            List of verified ExportedFigure records.

        Raises:
            ValueError: If no evaluation is available to export.
        """
        evaluation = self._last_evaluation
        if evaluation is None:
            raise ValueError(
                "no evaluation available to export; run design A or compare A vs B first"
            )
        out_dir = Path(destination_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = None
        if self._loaded_record is not None:
            digest = self._loaded_record.get("digest")

        exported: list[ExportedFigure] = []
        shot = evaluation.shot
        sole_field = shot.sole_field
        if sole_field is not None:
            stamp = evaluation_stamp(evaluation, view="sole_load", record_digest=digest)
            fig = sole_load_still(sole_field)
            p = out_dir / "sole_load.png"
            exported.append(export_figure(fig, p, stamp))

        traces = shot.traces
        if traces is not None:
            from .render_traces import trace_panel_still

            stamp = evaluation_stamp(evaluation, view="traces", record_digest=digest)
            fig = trace_panel_still(traces)
            p = out_dir / "traces.png"
            exported.append(export_figure(fig, p, stamp))

        scene = shot.scene
        if scene is not None:
            from .render3d import shot_scene_still

            stamp = evaluation_stamp(
                evaluation, view="shot_scene", record_digest=digest
            )
            fig = shot_scene_still(scene)
            p = out_dir / "shot_scene.png"
            exported.append(export_figure(fig, p, stamp))

        return exported

    def choose_export_views(self) -> None:
        """Ask for an output directory and export all stamped views."""
        dir_path = QFileDialog.getExistingDirectory(self, "Export views to directory")
        if dir_path:
            try:
                exported = self.export_views(dir_path)
                logger.info("exported %d views to %s", len(exported), dir_path)
            except (ValueError, OSError) as exc:
                logger.error("failed to export views: %s", exc)
                self._banner.show_error(f"Export failed: {exc}")

    def save_evaluation(self, path: Path | str) -> Path:
        """Save the current evaluation to a verifiable JSON record.

        Args:
            path: Destination path ending in .json.

        Returns:
            The saved Path.

        Raises:
            ValueError: If no evaluation is available to save.
        """
        if self._last_evaluation is not None:
            swing = self._conditions.swing_setup()
            return save_workbench_evaluation(self._last_evaluation, path, swing=swing)
        if self._loaded_record is not None:
            return save_workbench_evaluation(self._loaded_record, path)
        raise ValueError(
            "no evaluation available to save; run design A or compare A vs B first"
        )

    def choose_save_evaluation(self) -> None:
        """Prompt for a file path and save the evaluation."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save workbench evaluation",
            "",
            "JSON evaluation (*.json);;All files (*)",
        )
        if path:
            try:
                saved = self.save_evaluation(path)
                logger.info("saved evaluation to %s", saved)
            except (ValueError, OSError) as exc:
                logger.error("failed to save evaluation: %s", exc)
                self._banner.show_error(f"Save failed: {exc}")

    def load_evaluation(self, path: Path | str) -> dict[str, Any]:
        """Load a verified workbench evaluation record and update controls.

        Args:
            path: Path to the .json record.

        Returns:
            The verified record dictionary.

        Raises:
            ExportIntegrityError: If verification fails.
        """
        record = load_workbench_evaluation(path)
        self._loaded_record = record
        self._last_evaluation = None
        self._last_comparison = None

        inputs = record.get("inputs", {})
        design_data = inputs.get("design", {})
        if design_data and hasattr(self._design_a, "set_design"):
            self._design_a.set_design(design_data)

        sand_data = inputs.get("sand", {})
        swing_data = inputs.get("swing", {})
        if hasattr(self._conditions, "set_conditions"):
            self._conditions.set_conditions(sand_data, swing_data)

        verdict_info = record.get("verdict", {})
        status_str = verdict_info.get("status", "beyond_validation")
        status = EnvelopeStatus(status_str)
        self._banner.show_status(status)

        src_path = Path(path)
        digest = record.get("digest", "")
        stamp = record.get("stamp", "")
        schema = record.get("schema_version", "")
        headline = verdict_info.get("headline", "")
        summary = (
            f"Loaded evaluation: {src_path.name}\n"
            f"Schema: {schema}\n"
            f"Digest: {digest}\n"
            f"Stamp: {stamp}\n"
            f"Verdict: {headline}"
        )
        self._results.setPlainText(summary)
        self.update_control_states()
        self.evaluation_changed.emit()
        return record

    def choose_load_evaluation(self) -> None:
        """Prompt for a file path and load the verified evaluation."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load workbench evaluation",
            "",
            "JSON evaluation (*.json);;All files (*)",
        )
        if path:
            try:
                self.load_evaluation(path)
            except (ValueError, OSError) as exc:
                logger.error("failed to load evaluation: %s", exc)
                self._banner.show_error(f"Load failed: {exc}")


class BunkerShotWindow(QMainWindow):
    """Standalone window for the designer workbench."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        widget: BunkerShotWidget | None = None,
    ) -> None:
        """Build the window around a :class:`BunkerShotWidget`."""
        super().__init__(parent)
        self.setWindowTitle("BunkerShot3D Designer Workbench")
        self.setMinimumSize(1200, 800)
        self._widget = widget if widget is not None else BunkerShotWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(
            "F0 dynamic RFT. Every result carries a validity verdict; "
            "out-of-envelope queries are refused, not estimated."
        )
        menubar = self.menuBar()
        assert menubar is not None

        file_menu = menubar.addMenu("&File")
        self._export_action = file_menu.addAction("Export views…")
        self._export_action.setShortcut("Ctrl+E")
        self._export_action.triggered.connect(self._widget.choose_export_views)

        self._save_action = file_menu.addAction("Save evaluation…")
        self._save_action.setShortcut("Ctrl+S")
        self._save_action.triggered.connect(self._widget.choose_save_evaluation)

        self._load_action = file_menu.addAction("Load evaluation…")
        self._load_action.setShortcut("Ctrl+O")
        self._load_action.triggered.connect(self._widget.choose_load_evaluation)

        self._widget.evaluation_changed.connect(self.update_action_states)
        self.update_action_states()

        build_help_menu(
            menubar,
            self,
            doc_target=(
                "BunkerShot3D Credibility Statement",
                "docs/bunkershot3d/credibility.md",
            ),
        )

    @property
    def widget(self) -> BunkerShotWidget:
        """The embedded workbench widget."""
        return self._widget

    def disabled_reasons(self) -> dict[str, str | None]:
        """Delegate control disabled reasons to the embedded widget."""
        return self._widget.disabled_reasons()

    def update_action_states(self) -> None:
        """Update enabled states and tooltips for menu actions."""
        reasons = self._widget.disabled_reasons()
        exp_reason = reasons.get("export_views")
        self._export_action.setEnabled(exp_reason is None)
        if exp_reason:
            self._export_action.setToolTip(f"Disabled: {exp_reason}")
        else:
            self._export_action.setToolTip(
                "Export stamped figures of the current evaluation."
            )

        save_reason = reasons.get("save_evaluation")
        self._save_action.setEnabled(save_reason is None)
        if save_reason:
            self._save_action.setToolTip(f"Disabled: {save_reason}")
        else:
            self._save_action.setToolTip(
                "Save the current evaluation to a verifiable JSON record."
            )

        load_reason = reasons.get("load_evaluation")
        self._load_action.setEnabled(load_reason is None)
        if load_reason:
            self._load_action.setToolTip(f"Disabled: {load_reason}")
        else:
            self._load_action.setToolTip(
                "Load and verify a stored evaluation JSON record."
            )

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        """Clean the widget up on close."""
        self._widget.cleanup()
        super().closeEvent(event)


def get_dockable_ui() -> BunkerShotWindow:
    """Return the main window instance for docking in the unified launcher.

    Returns:
        A new workbench window.
    """
    return BunkerShotWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = BunkerShotWindow()
    window.show()
    sys.exit(app.exec())
