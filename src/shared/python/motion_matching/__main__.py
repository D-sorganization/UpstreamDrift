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
        print(
            f"Error: results directory does not exist: {results_dir}",
            file=sys.stderr,
        )
        return 1
    if not results_dir.is_dir():
        print(
            f"Error: results path is not a directory: {results_dir}",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output).resolve()
    try:
        written = generate_report(results_dir, output_path)
        print(f"Leaderboard written to: {written}")
        return 0
    except Exception as exc:
        print(f"Error generating leaderboard: {exc}", file=sys.stderr)
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
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
