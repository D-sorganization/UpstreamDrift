"""Cross-engine exercise dashboard."""

import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from src.shared.python.biomech.exercise_registry import discover_exercise
from src.shared.python.core.process_safety import narrow_catch
from src.shared.python.lifting.pack_audit.names import LIFTS
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Expected failure modes for the lift-baseline dock's lazy build: a missing
# or malformed committed receipt is already handled inside LiftBaselinePanel
# itself (it shows a message label instead of raising), so what reaches here
# is an environment/import problem with the panel module or its Qt widgets.
# Narrower than bare ``except Exception`` per ADR-0016 / narrow_catch, while
# still keeping a panel failure from ever taking down the host dashboard
# (#11748).
_LIFT_PANEL_LOAD_ERRORS: tuple[type[Exception], ...] = (
    ImportError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
)


def _engine_load_error_widget(
    name: str, error: Exception, exercise: str = "exercise"
) -> QLabel:
    """Create an actionable fallback when an optional engine cannot start."""
    message = str(error)
    is_mujoco_dll_failure = name == "MuJoCo_Models" and "dll" in message.lower()
    exercise_display = exercise.replace("_", " ").title()
    if is_mujoco_dll_failure:
        text = (
            "MuJoCo is unavailable on this computer.\n\n"
            f"The {exercise_display} exercise remains open. Choose JaxSim_Models from the "
            "Engine selector above for a dependency-light analysis view, or "
            "repair the native MuJoCo runtime and select MuJoCo_Models to retry.\n\n"
            f"Technical detail: {message}"
        )
    else:
        text = f"Error loading {name}:\n{message}"

    label = QLabel(text)
    label.setObjectName("engine-load-error")
    label.setWordWrap(True)
    return label


class ExerciseDashboard(QMainWindow):
    """Cross-engine exercise dashboard. Toolbar selects engine; body swaps dashboards."""

    def __init__(
        self,
        exercise: str,
        parent: QWidget | None = None,
        preferred_engine: str | None = None,
    ):
        super().__init__(parent)
        self.exercise = exercise
        self.setWindowTitle(f"Biomechanics Exercise: {exercise.title()}")

        self.toolbar = QToolBar("Engine Selection")
        self.addToolBar(self.toolbar)

        self.engine_selector = QComboBox()
        self.engines = discover_exercise(exercise)
        if not self.engines:
            # Fallback for UI if engines aren't discovered correctly in tests
            self.engines = [
                "MuJoCo_Models",
                "Drake_Models",
                "Pinocchio_Models",
                "JaxSim_Models",
            ]

        # JaxSim is a dependency-light analysis backend (no sibling model repo),
        # so it is offered as an always-available engine rather than discovered
        # from disk (issue #6658). The dashboard greys out unsupported features
        # from the backend's declared capabilities.
        if "JaxSim_Models" not in self.engines:
            self.engines.append("JaxSim_Models")

        self.engine_selector.addItems(self.engines)

        self.toolbar.addWidget(QLabel(" Engine: "))
        self.toolbar.addWidget(self.engine_selector)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.container)

        self._current_widget = None

        if self.engines:
            initial_engine = (
                preferred_engine
                if preferred_engine in self.engines
                else self.engines[0]
            )
            self.engine_selector.setCurrentText(initial_engine)
            self._on_engine_changed(initial_engine)

        self.engine_selector.currentTextChanged.connect(self._on_engine_changed)

        self.lift_baseline_dock: QDockWidget | None = None
        if exercise in LIFTS:
            self.lift_baseline_dock = self._build_lift_baseline_dock()

    def _build_lift_baseline_dock(self) -> QDockWidget:
        """Dock showing the LIFT-1 cross-engine baseline for a canonical lift.

        A failure building the panel (e.g. an environment/import problem)
        must never take down the rest of the dashboard: it is logged and
        replaced with an inline error label (issue #11748).
        """
        widget: QWidget | None = None
        with narrow_catch(
            *_LIFT_PANEL_LOAD_ERRORS,
            log_message="building the lift baseline panel",
        ):
            from src.launchers.lift_baseline_panel import LiftBaselinePanel

            widget = LiftBaselinePanel(initial_lift=self.exercise)
        if widget is None:
            widget = QLabel("Cross-engine lift baseline is unavailable.")
        dock = QDockWidget("Cross-Engine Lift Baseline", self)
        dock.setWidget(widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        return dock

    def _on_engine_changed(self, name: str) -> None:
        """Swap the inner widget to the engine-specific dashboard, scoped to `self.exercise`."""
        if self._current_widget is not None:
            self.layout.removeWidget(self._current_widget)
            self._current_widget.deleteLater()
            self._current_widget = None

        try:
            if name == "MuJoCo_Models":
                from src.launchers.mujoco_dashboard import MuJoCoDashboard

                self._current_widget = MuJoCoDashboard(exercise_filter=self.exercise)
            elif name == "Drake_Models":
                from src.launchers.drake_dashboard import DrakeDashboard

                self._current_widget = DrakeDashboard(exercise_filter=self.exercise)
            elif name == "Pinocchio_Models":
                from src.launchers.pinocchio_dashboard import PinocchioDashboard

                self._current_widget = PinocchioDashboard(exercise_filter=self.exercise)
            elif name == "JaxSim_Models":
                from src.launchers.jaxsim_dashboard import JaxSimDashboard

                self._current_widget = JaxSimDashboard(exercise_filter=self.exercise)
            elif name == "OpenSim_Models":
                self._current_widget = QLabel("OpenSim dashboard not yet available.")
            else:
                self._current_widget = QLabel(f"Unknown engine: {name}")

            if self._current_widget:
                # Remove window flags since we're embedding it. This hides
                # the title bar, so it can no longer be relied on to show
                # which engine/model is active -- UnifiedDashboardWindow's
                # in-body identity strip (ModelLoadStatus, issue #8829)
                # exists precisely so that identity survives this step.
                if isinstance(self._current_widget, QMainWindow):
                    self._current_widget.setWindowFlags(
                        self._current_widget.windowFlags()
                        & ~sys.modules["PyQt6.QtCore"].Qt.WindowType.Window
                    )
                self.layout.addWidget(self._current_widget)
        except Exception as error:  # noqa: BLE001 - optional engine boundary
            logger.exception("Unable to load exercise dashboard for %s", name)
            self._current_widget = _engine_load_error_widget(
                name, error, exercise=self.exercise
            )
            self.layout.addWidget(self._current_widget)


def get_dockable_ui() -> QMainWindow:
    """Return the main window instance for docking in the unified launcher."""
    import os

    exercise = os.environ.get("BIOMECH_EXERCISE", "gait")
    return ExerciseDashboard(
        exercise,
        preferred_engine=os.environ.get("BIOMECH_ENGINE"),
    )


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exercise",
        required=False,
        help="Name of the exercise (e.g. gait, sit_to_stand)",
    )
    args = parser.parse_args()

    exercise = args.exercise or os.environ.get("BIOMECH_EXERCISE", "gait")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = ExerciseDashboard(
        exercise,
        preferred_engine=os.environ.get("BIOMECH_ENGINE"),
    )
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
