"""Standalone launcher and CLI entry point for Shadow Tracker (ST-11, #10134)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.shadow_tracker.gui import ShadowTrackerReviewModel, ShadowTrackerWidget

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point for standalone Shadow Tracker launcher."""
    parser = argparse.ArgumentParser(
        description="Shadow Tracker review and editing workbench."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to an existing review bundle directory to open on start.",
    )
    parser.add_argument(
        "--export-canonical",
        type=Path,
        default=None,
        help="Export canonical package from the loaded bundle to the specified JSON file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without launching a graphical window.",
    )
    args = parser.parse_args(argv)

    if args.headless:
        model = ShadowTrackerReviewModel()
        if args.bundle:
            model.load_bundle(args.bundle)
            logger.info("Loaded bundle (%d frames)", model.frame_count)
            if args.export_canonical:
                model.export_canonical(args.export_canonical)
                logger.info("Exported canonical package to %s", args.export_canonical)
        return 0

    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Shadow Tracker Review Workbench")
    widget = ShadowTrackerWidget(parent=window)
    if args.bundle:
        widget.load_bundle(args.bundle)
        if args.export_canonical:
            widget.export_canonical(args.export_canonical)

    window.setCentralWidget(widget)
    window.resize(1024, 720)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
