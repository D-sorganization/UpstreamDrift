"""Standalone entry point for the Putting Green Simulator (#14448).

Launch from the repository root::

    python -m src.tools.putting_green_gui          # interactive window
    python -m src.tools.putting_green_gui --smoke  # one putt, then exit

``--smoke`` opens the window, runs one putt through the shared putting core
(ADR-0058), writes the metrics to stdout and exits ``0``. Run it with
``QT_QPA_PLATFORM=offscreen`` on a headless host.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

CORE_MODULE = "src.shared.python.putting_dynamics"
EXIT_OK = 0
EXIT_SMOKE_FAILED = 1
EXIT_CORE_MISSING = 2


def _core_import_error() -> str:
    """Return a readable reason the putting core cannot load, or ``""``."""
    try:
        importlib.import_module(CORE_MODULE)
    except ImportError as exc:
        return (
            f"Putting core '{CORE_MODULE}' is not importable ({exc}). "
            "Run from the repository root with the project installed, and "
            "initialise the Tools tree: git submodule update --init vendor/ud-tools"
        )
    return ""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.tools.putting_green_gui")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one putt, print the metrics and exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the simulator; return a process exit code.

    Postcondition: returns ``EXIT_CORE_MISSING`` without creating any Qt
    object when the putting core cannot be imported.
    """
    args = _parse_args(argv)
    reason = _core_import_error()
    if reason:
        sys.stderr.write(reason + "\n")
        return EXIT_CORE_MISSING

    from PyQt6.QtWidgets import QApplication

    from src.tools.putting_green_gui.gui import PuttingGreenWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = PuttingGreenWindow()
    window.show()
    if not args.smoke:
        return app.exec()

    report = window.run_one_putt()
    window.close()
    sys.stdout.write(report + "\n")
    return EXIT_OK if report.startswith("Putting Simulation") else EXIT_SMOKE_FAILED


if __name__ == "__main__":
    sys.exit(main())
