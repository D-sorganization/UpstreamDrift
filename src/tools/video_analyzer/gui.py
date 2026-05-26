"""Video Analyzer GUI component."""

from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget


class VideoAnalyzerWindow(QMainWindow):
    """Standalone window for Video Analyzer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Analyzer")
        self.setMinimumSize(800, 600)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Video Analyzer (GUI placeholder)"))
        self.setCentralWidget(widget)


def get_dockable_ui(parent=None) -> VideoAnalyzerWindow:
    """Return the main window instance for docking in the unified launcher."""
    return VideoAnalyzerWindow(parent)
