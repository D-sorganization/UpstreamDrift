# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Entry point: ``python -m movement_optimizer``.

Dispatches between three modes per the UpstreamDrift launcher contract
(``D-sorganization/Movement-Optimizer#456``):

* default / ``--gui``: open the PyQt6 GUI (preserves prior behavior).
* ``--headless --exercise <name> [--output <path>]``: run a single
  optimization and exit. The result is written as JSON via the existing
  :mod:`movement_optimizer.cli` pipeline.
* ``--list-exercises``: print the manifest-declared exercise IDs.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Construct the launcher-aware argument parser."""
    parser = argparse.ArgumentParser(
        prog="movement-optimizer",
        description="Movement-Optimizer launcher dispatcher (GUI / headless / list).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Qt GUI (default when no mode flag is given).",
    )
    mode.add_argument(
        "--headless",
        action="store_true",
        help="Run a single headless optimization and exit.",
    )
    mode.add_argument(
        "--list-exercises",
        dest="list_exercises",
        action="store_true",
        help="Print the manifest-declared exercise IDs and exit.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Exercise to optimize (required when --headless is given).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path for --headless results.",
    )
    parser.add_argument("--body-mass", type=float, default=75.0)
    parser.add_argument("--height", type=float, default=1.75)
    parser.add_argument("--bar-mass", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--smoothness", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_gui() -> int:
    """Open the PyQt6 GUI; returns the Qt event-loop exit code."""
    import logging

    from PyQt6.QtWidgets import QApplication

    from .gui import MainWindow
    from .gui.app_icon import movement_optimizer_icon
    from .i18n import setup_translations

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = QApplication(sys.argv)
    app.setWindowIcon(movement_optimizer_icon())
    setup_translations()
    window = MainWindow()
    window.show()
    return app.exec()


def _run_list_exercises() -> int:
    """Print the manifest exercises one per line; returns 0."""
    from .tool_pack import list_exercises

    for name in list_exercises():
        sys.stdout.write(f"{name}\n")
    return 0


def _run_headless(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Forward to :mod:`movement_optimizer.cli`; returns its exit code."""
    if args.exercise is None:
        parser.error("--headless requires --exercise <name>")
    from .cli import main as cli_main

    cli_argv: list[str] = ["--exercise", args.exercise]
    if args.output is not None:
        cli_argv += ["--output", args.output]
    cli_argv += [
        "--body-mass",
        str(args.body_mass),
        "--height",
        str(args.height),
        "--bar-mass",
        str(args.bar_mass),
        "--duration",
        str(args.duration),
        "--smoothness",
        str(args.smoothness),
    ]
    if args.verbose:
        cli_argv.append("--verbose")
    return cli_main(cli_argv)


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_exercises:
        return _run_list_exercises()
    if args.headless:
        return _run_headless(parser, args)
    return _run_gui()


if __name__ == "__main__":
    sys.exit(main())
