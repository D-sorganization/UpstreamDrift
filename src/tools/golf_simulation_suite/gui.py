"""Golf Simulation Suite GUI component."""

from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget


class GolfSimulationSuiteWindow(QMainWindow):
    """Standalone window for Golf Simulation Suite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Golf Simulation Suite")
        self.setMinimumSize(800, 600)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Golf Simulation Suite (GUI placeholder)"))
        self.setCentralWidget(widget)


def get_dockable_ui(parent=None) -> GolfSimulationSuiteWindow:
    """Return the main window instance for docking in the unified launcher."""
    return GolfSimulationSuiteWindow(parent)
