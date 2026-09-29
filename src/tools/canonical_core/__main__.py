"""Standalone PyQt6 launcher for canonical-core shell tools."""

from __future__ import annotations

import argparse
import sys

from src.tools.canonical_core.registry import get_canonical_core_tool


def main(argv: list[str] | None = None) -> int:
    """Launch a canonical-core shell as a standalone PyQt6 window."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool-id",
        choices=("canonical_core_estimation", "canonical_core_comparison"),
        default="canonical_core_estimation",
    )
    args = parser.parse_args(argv)

    from PyQt6.QtWidgets import QApplication, QMainWindow

    from src.tools.canonical_core.pyqt_shell import CanonicalCoreShellWidget

    app = QApplication.instance() or QApplication(sys.argv)
    descriptor = get_canonical_core_tool(args.tool_id)
    window = QMainWindow()
    window.setWindowTitle(descriptor.name)
    window.setCentralWidget(CanonicalCoreShellWidget(descriptor))
    window.resize(980, 680)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
