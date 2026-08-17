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

Running a design blocks the interface for a second or two: the lofted mesh is
solved by root-finding per station and the playability sweep is
``playability_points ** 2`` shots. A wait cursor is shown; the work is not
threaded, because a design tool whose answer arrives out of order with its
inputs is worse than one that pauses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ui import HoverCopyTextBrowser

from .design import SandCondition, SolverSetup, SwingSetup, WorkbenchInputError
from .model import DesignEvaluation, WorkbenchComparison, WorkbenchModel
from .report import comparison_report, evaluation_report
from .widgets import ConditionPanel, DesignPanel, GridMapWidget, VerdictBanner

__all__ = [
    "BunkerShotWidget",
    "BunkerShotWindow",
    "get_dockable_ui",
]

logger = logging.getLogger(__name__)

_IDLE_TEXT = (
    "BunkerShot3D designer workbench\n"
    "===============================\n\n"
    "Set the sole parameters, the playing condition and the delivery, then\n"
    "run design A or compare A against B.\n\n"
    "Every result carries a validity verdict. At greenside delivery speeds\n"
    "3D-RFT is roughly 60x outside its stated Froude limit and about 20x\n"
    "beyond any published validation, so the verdict is part of the answer,\n"
    "not a disclaimer attached to it. When the solver refuses a query no\n"
    "force, depth or carry is reported at all.\n"
)

ModelFactory = Callable[[SolverSetup], WorkbenchModel]

_ResultT = TypeVar("_ResultT")


class BunkerShotWidget(QWidget):
    """The workbench: two candidate soles, one condition, one verdict."""

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
        self._build_ui()

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
        self._run_button.clicked.connect(self.run_design_a)
        column.addWidget(self._run_button)

        self._compare_button = QPushButton("Compare A vs B")
        self._compare_button.clicked.connect(self.run_comparison)
        column.addWidget(self._compare_button)
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

        maps = QGridLayout()
        self._bounce_map_a = GridMapWidget("A: bounce utilisation")
        self._bounce_map_b = GridMapWidget("B: bounce utilisation")
        self._window_map_a = GridMapWidget("A: playability window")
        self._window_map_b = GridMapWidget("B: playability window")
        maps.addWidget(self._bounce_map_a, 0, 0)
        maps.addWidget(self._bounce_map_b, 0, 1)
        maps.addWidget(self._window_map_a, 1, 0)
        maps.addWidget(self._window_map_b, 1, 1)
        column.addLayout(maps, stretch=2)

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

    def run_design_a(self) -> None:
        """Evaluate design A and display the result."""
        inputs = self._read_inputs()
        if inputs is None:
            return
        settings, sand, swing = inputs
        try:
            design = self._design_a.design()
        except WorkbenchInputError as error:
            self._show_input_error(error)
            return
        self._banner.show_busy("Running the F0 solver over design A...")
        result = self._guarded(
            lambda: self._model_factory(settings).evaluate(design, sand, swing)
        )
        if result is None:
            return
        self.show_evaluation(result)

    def run_comparison(self) -> None:
        """Evaluate both designs and rank them, with uncertainty attached."""
        inputs = self._read_inputs()
        if inputs is None:
            return
        settings, sand, swing = inputs
        try:
            left = self._design_a.design()
            right = self._design_b.design()
        except WorkbenchInputError as error:
            self._show_input_error(error)
            return
        if left.name == right.name:
            self._show_input_error(
                WorkbenchInputError(
                    "designs A and B need different names so the ranking can "
                    f"report them; both are {left.name!r}"
                )
            )
            return
        self._banner.show_busy("Running the F0 solver over both designs...")
        result = self._guarded(
            lambda: self._model_factory(settings).compare(left, right, sand, swing)
        )
        if result is None:
            return
        self.show_comparison(result)

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
        self._banner.show_status(evaluation.shot.status)
        self._results.setPlainText(evaluation_report(evaluation))
        self._paint_maps(evaluation, self._bounce_map_a, self._window_map_a)
        self._bounce_map_b.clear()
        self._window_map_b.clear()

    def show_comparison(self, comparison: WorkbenchComparison) -> None:
        """Display an A/B comparison.

        The banner shows the worse of the two verdicts, because a comparison
        is only as trustworthy as its least trustworthy half.

        Args:
            comparison: The two evaluations and their ranking.
        """
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
        self._paint_maps(comparison.left, self._bounce_map_a, self._window_map_a)
        self._paint_maps(comparison.right, self._bounce_map_b, self._window_map_b)

    def _paint_maps(
        self,
        evaluation: DesignEvaluation,
        bounce_map: GridMapWidget,
        window_map: GridMapWidget,
    ) -> None:
        """Paint one design's two maps, clearing them when there is no data."""
        sole_load = evaluation.shot.sole_load
        if sole_load is None:
            bounce_map.clear()
        else:
            utilisation = sole_load.utilisation
            bounce_map.set_grid(
                sole_load.density_pa_s,
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
        ):
            widget.clear()

    def cleanup(self) -> None:
        """Release resources. The workbench holds none beyond its widgets."""


class BunkerShotWindow(QMainWindow):
    """Standalone window for the designer workbench."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the window around a :class:`BunkerShotWidget`."""
        super().__init__(parent)
        self.setWindowTitle("BunkerShot3D Designer Workbench")
        self.setMinimumSize(1200, 800)
        self._widget = BunkerShotWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(
            "F0 dynamic RFT. Every result carries a validity verdict; "
            "out-of-envelope queries are refused, not estimated."
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
