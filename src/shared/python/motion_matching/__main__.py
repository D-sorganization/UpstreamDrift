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


def ledger_cli(args: argparse.Namespace) -> int:
    """Scan execution receipts and emit/write matched swing ledger."""
    from src.shared.python.motion_matching.leaderboard import (
        sync_leaderboard_from_ledger,
    )
    from src.shared.python.motion_matching.ledger import (
        default_ledger_path,
        scan,
    )

    try:
        ledger = scan()
        if args.write:
            out_path = (
                Path(args.output).resolve() if args.output else default_ledger_path()
            )
            ledger.write_json(out_path)
            # Synchronize cross_engine_leaderboard.json unconditionally from ledger rows
            sync_leaderboard_from_ledger(ledger.rows)
            sys.stdout.write(
                f"Wrote ledger ({ledger.total_receipts} receipts) to {out_path}\n"
            )
        else:
            sys.stdout.write(ledger.to_json())
        return 0
    except Exception as exc:  # noqa: BLE001
        _print_error(f"failed to generate ledger: {exc}")
        return 1


def export_report_cli(args: argparse.Namespace) -> int:
    """Generate fit-quality report from an execution receipt."""
    from src.shared.python.motion_matching.export import export_report

    try:
        export_report(
            receipt=args.receipt,
            path=args.out,
            candidate=args.candidate,
        )
        sys.stdout.write(f"Wrote fit-quality report to {args.out}\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        _print_error(f"failed to export report: {exc}")
        return 1


def export_video_cli(args: argparse.Namespace) -> int:
    """Generate 3D marker overlay video animation for a candidate."""
    from src.shared.python.motion_matching.export import export_video

    try:
        export_video(
            candidate=args.candidate,
            engine=args.engine,
            path=args.out,
            fps=args.fps,
            stride=args.stride,
        )
        sys.stdout.write(f"Wrote candidate video to {args.out}\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        _print_error(f"failed to export video: {exc}")
        return 1


def _register_export_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register export-report and export-video subcommands."""
    report_parser = subparsers.add_parser(
        "export-report", help="Export fit-quality report from receipt"
    )
    report_parser.add_argument(
        "--receipt",
        type=str,
        required=True,
        help="Path to execution receipt JSON",
    )
    report_parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output report path (.md or .pdf)",
    )
    report_parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Optional path to candidate .npz package",
    )

    video_parser = subparsers.add_parser(
        "export-video", help="Export marker overlay video for candidate"
    )
    video_parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate .npz package",
    )
    video_parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Physics engine name",
    )
    video_parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output video path (.gif or .mp4)",
    )
    video_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    video_parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Frame subsampling stride (default: 5)",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Motion matching utilities",
        prog="python3 -m src.shared.python.motion_matching",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

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

    ledger_parser = subparsers.add_parser(
        "ledger", help="Scan receipts and generate matched swing ledger"
    )
    ledger_parser.add_argument(
        "--write",
        action="store_true",
        help="Write ledger to reports/matched_swing_ledger.json",
    )
    ledger_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path for ledger JSON",
    )

    _register_export_subparsers(subparsers)
    return parser


def main() -> int:
    """Parse CLI arguments and dispatch to subcommand."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "leaderboard":
        return leaderboard_cli(args)
    if args.command == "ledger":
        return ledger_cli(args)
    if args.command == "export-report":
        return export_report_cli(args)
    if args.command == "export-video":
        return export_video_cli(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
