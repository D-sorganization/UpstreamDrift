import sys

from PyQt6.QtWidgets import QApplication

from src.shared.python.logging_pkg.logging_config import get_logger

from .golf_visualizer_app import ModernGolfVisualizerApp

logger = get_logger(__name__)


def main() -> None:
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Modern Golf Swing Visualizer")
    app.setApplicationVersion("2.0")
    window = ModernGolfVisualizerApp()
    window.show()
    try:
        window.gl_widget.load_data("BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat")
    except (RuntimeError, ValueError, OSError) as exc:
        logger.info("Note: Sample data not found - %s", exc)
        logger.info("Please load data using File -> Load Data")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
