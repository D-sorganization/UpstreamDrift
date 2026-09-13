"""Video Analyzer GUI component."""

from PyQt6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from src.launchers.help_menu import build_help_menu


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
        menubar = self.menuBar()
        assert menubar is not None
        build_help_menu(
            menubar,
            self,
            doc_target=(
                "Video Analysis Tutorial",
                "docs/tutorials/content/04_video_analysis.md",
            ),
        )


def get_dockable_ui(parent=None) -> VideoAnalyzerWindow:
    """Return the main window instance for docking in the unified launcher."""
    return VideoAnalyzerWindow(parent)
