"""Entry point for ``python -m sidekick`` — T1 (#5979).

Subcommands:
  gui   Launch the Sidekick GUI window (default when no subcommand given).
  run   Invoke a calculator headlessly — no PyQt6 imported at parse time.

Usage examples::

    python -m sidekick                                     # GUI, chat-first
    python -m sidekick gui --profile calc-first
    python -m sidekick run --calculator wgs_reactor --inputs wgs.json
    python -m sidekick --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_PROFILES = ("chat-first", "calc-first")
_VALID_FORMATS = ("json", "csv")

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level CLI argument parser.

    Postconditions:
        - Returned parser has subparsers 'gui' and 'run'.
        - '--profile' is restricted to _VALID_PROFILES.
        - '--format' on run is restricted to _VALID_FORMATS.
    """
    parser = argparse.ArgumentParser(
        prog="python -m sidekick",
        description=(
            "Sidekick — standalone process-engineering assistant.\n\n"
            "  python -m sidekick                        # GUI, chat-first layout\n"
            "  python -m sidekick gui --profile calc-first\n"
            "  python -m sidekick run --calculator X --inputs file.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.default = "gui"

    _add_gui_subparser(subparsers)
    _add_run_subparser(subparsers)

    assert parser.parse_args([]).subcommand == "gui"  # noqa: S101 - DbC postcondition
    return parser


def _add_gui_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    gui = subparsers.add_parser(
        "gui",
        help="Launch the Sidekick GUI window (default).",
        description="Launch the Sidekick GUI window.",
    )
    gui.add_argument(
        "--profile",
        choices=_VALID_PROFILES,
        default="chat-first",
        metavar="PROFILE",
        help=(
            f"Window layout profile. "
            f"Choices: {', '.join(_VALID_PROFILES)}. "
            "Default: chat-first."
        ),
    )
    gui.add_argument(
        "--theme",
        default=None,
        metavar="NAME",
        help="Theme name to apply on startup (e.g. 'catppuccin-mocha').",
    )
    gui.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        dest="data_dir",
        help="Override the session-data directory (resolved to absolute path).",
        type=_resolve_data_dir,
    )


def _add_run_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    run = subparsers.add_parser(
        "run",
        help="Invoke a calculator headlessly (no GUI).",
        description="Run a Sidekick calculator from the command line, no window required.",
    )
    run.add_argument(
        "--calculator",
        required=True,
        metavar="ID",
        help="Calculator feature id, e.g. 'wgs_reactor'. Format: ^[a-z][a-z0-9_]*$.",
    )
    run.add_argument(
        "--inputs",
        required=True,
        metavar="PATH",
        help="Path to a JSON or YAML inputs file (auto-detected by extension).",
    )
    run.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write results to PATH instead of stdout.",
    )
    run.add_argument(
        "--format",
        choices=_VALID_FORMATS,
        default="json",
        metavar="FORMAT",
        help=(f"Output format. Choices: {', '.join(_VALID_FORMATS)}. Default: json."),
    )


def _resolve_data_dir(raw: str) -> str:
    """Resolve a path string to an absolute path.

    Raises:
        ValueError: If the path cannot be resolved to an absolute path.
    """
    resolved = Path(raw).resolve()
    if not resolved.is_absolute():
        raise ValueError(  # noqa: TRY004
            f"--data-dir could not be resolved to an absolute path: {raw!r}"
        )
    return str(resolved)


def _handle_gui(ns: argparse.Namespace) -> int:
    """Launch the Sidekick GUI. Deferred PyQt6 import keeps 'run' headless.

    Preconditions:
        ns.subcommand == 'gui'
        ns.profile in _VALID_PROFILES
    """
    assert ns.subcommand == "gui"  # noqa: S101 - DbC
    assert ns.profile in _VALID_PROFILES  # noqa: S101 - DbC

    from sidekick.launcher_factory import create_launcher_config, launch_app

    config = create_launcher_config(
        app_module="sidekick",
        window_title="Sidekick",
        min_width=900,
        min_height=600,
    )

    def _window_factory() -> object:
        import platformdirs

        from sidekick.standalone.session_store import StandaloneSessionStore
        from sidekick.standalone.window import (
            StandaloneSidekickConfig,
            StandaloneSidekickWindow,
        )

        data_dir = (
            Path(ns.data_dir)
            if ns.data_dir
            else Path(platformdirs.user_data_dir("sidekick", appauthor=False))
        )
        store = StandaloneSessionStore(data_dir)
        cfg = StandaloneSidekickConfig(
            profile=ns.profile,
            theme_name=getattr(ns, "theme", None),
            session_store=store,
        )
        return StandaloneSidekickWindow(cfg)

    return launch_app(config, _window_factory)


def _handle_run(ns: argparse.Namespace) -> int:
    """Invoke a calculator headlessly. Never imports PyQt6.

    Preconditions:
        ns.subcommand == 'run'
        ns.calculator is a non-empty string
        ns.inputs is a non-empty string
    """
    assert ns.subcommand == "run"  # noqa: S101 - DbC
    if not ns.calculator:
        raise ValueError("--calculator must be a non-empty string")
    if not ns.inputs:
        raise ValueError("--inputs must be a non-empty string")

    from sidekick.standalone.run import handle_run

    return handle_run(ns)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate handler.

    Args:
        argv: Argument vector (defaults to sys.argv[1:]).

    Returns:
        Integer exit code (0 = success, non-zero = failure).
    """
    parser = build_parser()
    ns = parser.parse_args(argv)

    subcommand = getattr(ns, "subcommand", None) or "gui"
    if subcommand == "gui":
        return _handle_gui(ns)
    if subcommand == "run":
        return _handle_run(ns)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
