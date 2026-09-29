"""
CLI entrypoint for FreeMoCap integration.

Usage:
    python -m upstream_drift.motion_capture.freemocap <session_dir> [options]

Or directly:
    python -m src.motion_capture.freemocap_ingest <session_dir> [options]
"""

import argparse
import logging
import sys
from pathlib import Path

from .launcher import FreeMoCapLauncher, LaunchConfig
from .output_adapter import FreeMoCapOutputAdapter


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="FreeMoCap motion capture integration for UpstreamDrift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FreeMoCap capture on a session directory
  python -m src.motion_capture.freemocap_ingest /path/to/session

  # Run with specific video directory
  python -m src.motion_capture.freemocap_ingest /path/to/session --video-dir /path/to/videos

  # Parse output after capture
  python -m src.motion_capture.freemocap_ingest --parse /path/to/freemocap_output

  # Full pipeline: capture and parse
  python -m src.motion_capture.freemocap_ingest /path/to/session --video-dir /path/to/videos --parse-output
        """,
    )

    # Positional argument
    parser.add_argument(
        "session_dir",
        type=Path,
        nargs="?",
        help="Session directory for capture data",
    )

    # Mode selection
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse mode: parse existing FreeMoCap output instead of running capture",
    )

    # Capture options
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing video files to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--freemocap-env",
        type=Path,
        help="Path to FreeMoCap Python environment",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run FreeMoCap with GUI (not headless)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600)",
    )

    # Parse options
    parser.add_argument(
        "--parse-output",
        action="store_true",
        help="After capture, parse the output files",
    )
    parser.add_argument(
        "--export-npy",
        type=Path,
        help="Export parsed data to numpy file",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Export parsed data to CSV file",
    )

    # General options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Validate arguments
    if not args.parse and not args.session_dir:
        parser.error("session_dir is required for capture mode")

    if args.parse and not args.session_dir:
        parser.error("session_dir (output directory) is required for parse mode")

    # Parse mode only
    if args.parse:
        logger.info(f"Parsing FreeMoCap output from: {args.session_dir}")
        adapter = FreeMoCapOutputAdapter(args.session_dir)
        session = adapter.parse()

        if session.frames:
            pass

        if session.calibration:
            pass

        if args.export_npy:
            adapter.export_to_numpy(args.export_npy)

        if args.export_csv:
            adapter.export_to_csv(args.export_csv)

        return 0

    # Capture mode
    logger.info(f"Session directory: {args.session_dir}")

    config = LaunchConfig(
        session_dir=args.session_dir,
        freemocap_env=args.freemocap_env,
        timeout_seconds=args.timeout,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        headless=not args.gui,
    )

    if args.dry_run:
        return 0

    launcher = FreeMoCapLauncher()
    result = launcher.launch(config)

    if result.success:
        if args.parse_output and result.output_dir:
            adapter = FreeMoCapOutputAdapter(result.output_dir)
            session = adapter.parse()

            if args.export_npy:
                adapter.export_to_numpy(args.export_npy)

            if args.export_csv:
                adapter.export_to_csv(args.export_csv)

        return 0
    if result.log_file:
        pass
    return 1


if __name__ == "__main__":
    sys.exit(main())
