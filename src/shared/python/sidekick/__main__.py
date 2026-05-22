"""CLI entry point for launching and dispatching standalone Sidekick flows."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import NoReturn

_SUBCOMMANDS = ("gui", "run")
_GUI_PROFILES = ("chat-first", "calc-first")
_OUTPUT_FORMATS = ("json", "csv")
_HELP_EPILOG = """Examples:
  python -m sidekick
  python -m sidekick gui --profile calc-first --theme solarized
  python -m sidekick run --calculator unit-converter --inputs ./inputs.json
"""


class SidekickArgumentParser(argparse.ArgumentParser):
    """Argument parser that suggests the closest valid subcommand or flag."""

    def error(self, message: str) -> NoReturn:
        suggestion = _suggest_token(message, _known_cli_tokens(self))
        if suggestion is not None:
            message = f"{message}. Did you mean '{suggestion}'?"
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def _known_cli_tokens(parser: argparse.ArgumentParser) -> set[str]:
    tokens = set(parser._option_string_actions)  # noqa: SLF001 - argparse internals
    for action in parser._actions:  # noqa: SLF001 - argparse internals
        if isinstance(action, argparse._SubParsersAction):
            tokens.update(action.choices)
            for subparser in action.choices.values():
                tokens.update(
                    subparser._option_string_actions  # noqa: SLF001 - argparse internals
                )
    return tokens


def _suggest_token(message: str, candidates: set[str]) -> str | None:
    for raw_token in message.replace(",", " ").split():
        token = raw_token.strip("'\"")
        if not token.startswith("-") and token not in candidates:
            continue
        matches = difflib.get_close_matches(token, sorted(candidates), n=1, cutoff=0.6)
        if matches:
            return matches[0]
    return None


def _normalize_argv(argv: list[str] | None) -> list[str]:
    args = list(argv or [])
    if not args:
        return ["gui"]
    head = args[0]
    if head in _SUBCOMMANDS or head in {"-h", "--help"}:
        return args
    if head.startswith("-"):
        return ["gui", *args]
    return args


def _resolved_path(value: str) -> Path:
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("path must be a non-empty string")
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid path: {value}") from exc


def _existing_file(value: str) -> Path:
    path = _resolved_path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"path is not a file: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = SidekickArgumentParser(
        prog="python -m sidekick",
        description="Standalone Sidekick launcher and headless dispatcher.",
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        parser_class=SidekickArgumentParser,
    )

    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch standalone Sidekick with deferred GUI imports.",
    )
    gui_parser.add_argument(
        "--profile",
        choices=_GUI_PROFILES,
        default="chat-first",
        help="Initial standalone layout profile (default: chat-first).",
    )
    gui_parser.add_argument(
        "--theme",
        metavar="NAME",
        help="Optional theme override for the standalone session.",
    )
    gui_parser.add_argument(
        "--data-dir",
        type=_resolved_path,
        metavar="PATH",
        help="Optional absolute or relative data directory for standalone Sidekick.",
    )
    gui_parser.set_defaults(handler=launch_gui)

    run_parser = subparsers.add_parser(
        "run",
        help="Parse headless calculator invocation arguments.",
    )
    run_parser.add_argument(
        "--calculator",
        required=True,
        metavar="ID",
        help="Registered calculator identifier to invoke.",
    )
    run_parser.add_argument(
        "--inputs",
        type=_existing_file,
        required=True,
        metavar="PATH",
        help="Input payload file for the calculator invocation.",
    )
    run_parser.add_argument(
        "--output",
        type=_resolved_path,
        metavar="PATH",
        help="Optional destination file for calculator output.",
    )
    run_parser.add_argument(
        "--format",
        choices=_OUTPUT_FORMATS,
        default="json",
        help="Output format when --output is provided (default: json).",
    )
    run_parser.set_defaults(handler=run_headless)
    return parser


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Sidekick CLI arguments with implicit ``gui`` defaulting."""
    parser = build_parser()
    return parser.parse_args(_normalize_argv(argv))


def launch_gui(args: argparse.Namespace) -> int:
    """Launch the standalone GUI once the dedicated window module exists."""
    try:
        from sidekick.standalone.window import StandaloneSidekickWindow
    except ImportError:
        sys.stderr.write(
            "sidekick gui is wired, but the standalone window lands in issue #5980.\n"
        )
        return 1

    from sidekick.launcher_factory import create_launcher_config, launch_app

    data_dir = args.data_dir or Path.cwd().resolve()
    config = create_launcher_config(
        app_module="sidekick.standalone.window",
        window_title="Sidekick",
        min_width=1280,
        min_height=800,
        profile=args.profile,
        theme_name=args.theme,
        data_dir=str(data_dir),
    )
    return launch_app(
        config,
        window_factory=lambda: StandaloneSidekickWindow(
            profile=args.profile,
            theme_name=args.theme,
            data_dir=data_dir,
        ),
    )


def run_headless(args: argparse.Namespace) -> int:
    """Reserve the ``run`` subcommand contract until issue #5982 lands."""
    from src.shared.python.core.process_safety import narrow_catch

    with narrow_catch(ValueError, FileNotFoundError, log_message="sidekick run"):
        if args.output is not None and not args.output.parent.exists():
            raise FileNotFoundError(args.output.parent)
        sys.stderr.write(
            "sidekick run parsing is ready; execution lands in issue #5982.\n"
        )
        return 1
    sys.stderr.write("sidekick run failed; check calculator arguments and paths.\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch the selected Sidekick subcommand."""
    args = parse_cli_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        build_parser().print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
