"""Cross-engine exercise dashboard."""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QToolBar,
    QLabel,
)
from src.shared.python.biomech.exercise_registry import discover_exercise


class ExerciseDashboard(QMainWindow):
    """Cross-engine exercise dashboard. Toolbar selects engine; body swaps dashboards."""

    def __init__(self, exercise: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.exercise = exercise
        self.setWindowTitle(f"Biomechanics Exercise: {exercise.title()}")

        self.toolbar = QToolBar("Engine Selection")
        self.addToolBar(self.toolbar)

        self.engine_selector = QComboBox()
        self.engines = discover_exercise(exercise)
        if not self.engines:
            # Fallback for UI if engines aren't discovered correctly in tests
            self.engines = ["MuJoCo_Models", "Drake_Models", "Pinocchio_Models"]

        self.engine_selector.addItems(self.engines)
        self.engine_selector.currentTextChanged.connect(self._on_engine_changed)

        self.toolbar.addWidget(QLabel(" Engine: "))
        self.toolbar.addWidget(self.engine_selector)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.container)

        self._current_widget = None

        if self.engines:
            self._on_engine_changed(self.engines[0])

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
            elif name == "OpenSim_Models":
                self._current_widget = QLabel("OpenSim dashboard not yet available.")
            else:
                self._current_widget = QLabel(f"Unknown engine: {name}")

            if self._current_widget:
                # Remove window flags since we're embedding it
                if isinstance(self._current_widget, QMainWindow):
                    self._current_widget.setWindowFlags(
                        self._current_widget.windowFlags()
                        & ~sys.modules["PyQt6.QtCore"].Qt.WindowType.Window
                    )
                self.layout.addWidget(self._current_widget)
        except Exception as e:
            self._current_widget = QLabel(f"Error loading {name}:\n{e}")
            self.layout.addWidget(self._current_widget)


def get_dockable_ui() -> QMainWindow:
    """Return the main window instance for docking in the unified launcher."""
    import os

    exercise = os.environ.get("BIOMECH_EXERCISE", "gait")
    return ExerciseDashboard(exercise)


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

    app = QApplication(sys.argv)
    window = ExerciseDashboard(exercise)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
