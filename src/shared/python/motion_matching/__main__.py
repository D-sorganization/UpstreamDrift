"""CLI entry point for motion_matching module.

Usage:
    python3 -m src.shared.python.motion_matching leaderboard --results-dir <dir> --output <file>
    python3 -m src.shared.python.motion_matching leaderboard --results-dir <dir>  # writes to LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.shared.python.motion_matching.leaderboard import generate_report


def _print_error(message: str) -> None:
    """Emit a leaderboard CLI diagnostic on stderr."""
    sys.stderr.write(f"motion_matching leaderboard: {message}\n")


def leaderboard_cli(args: argparse.Namespace) -> int:
    """Generate a cross-engine leaderboard from a results directory.

    Args:
        args: Parsed CLI arguments with ``results_dir`` and ``output`` fields.

    Returns:
        0 on success, 1 on error.

    Raises:
        SystemExit: with exit code 1 if the results directory is invalid.
    """
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        _print_error(f"results directory does not exist: {results_dir}")
        return 1
    if not results_dir.is_dir():
        _print_error(f"results path is not a directory: {results_dir}")
        return 1

    output_path = Path(args.output).resolve()
    try:
        generate_report(results_dir, output_path)
        return 0
    except Exception as exc:  # noqa: BLE001
        _print_error(f"failed to generate report from {results_dir}: {exc}")
        return 1


def main() -> int:
    """Parse CLI arguments and dispatch to subcommand."""
    parser = argparse.ArgumentParser(
        description="Motion matching utilities",
        prog="python3 -m src.shared.python.motion_matching",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Leaderboard subcommand
    leaderboard_parser = subparsers.add_parser(
        "leaderboard", help="Generate cross-engine leaderboard"
    )
    leaderboard_parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing <trial>/<engine>.json result files",
    )
    leaderboard_parser.add_argument(
        "--output",
        type=str,
        default="LEADERBOARD.md",
        help="Output file path (default: LEADERBOARD.md)",
    )

    args = parser.parse_args()
    if args.command == "leaderboard":
        return leaderboard_cli(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
