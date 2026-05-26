"""Training Controller GUI component."""

from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget


class TrainingControllerWindow(QMainWindow):
    """Standalone window for Training Controller."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Controller")
        self.setMinimumSize(800, 600)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Training Controller (GUI placeholder)"))
        self.setCentralWidget(widget)


def get_dockable_ui(parent=None) -> TrainingControllerWindow:
    """Return the main window instance for docking in the unified launcher."""
    return TrainingControllerWindow(parent)
