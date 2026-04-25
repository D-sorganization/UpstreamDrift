# mypy: disable-error-code="no-redef"
"""Thin facade for the legacy golf swing visualizer."""

from __future__ import annotations

import sys


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
