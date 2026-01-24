"""Centralized CLI utilities for the Golf Modeling Suite.

This module consolidates common command-line argument parsing patterns
across the codebase, addressing DRY violations identified in
Pragmatic Programmer reviews.

Usage:
    from src.shared.python.cli_utils import (
        create_base_parser,
        add_logging_args,
        add_output_args,
        add_config_args,
        setup_from_args,
    )

    # Create parser with common arguments
    parser = create_base_parser("My script description")
    add_logging_args(parser)
    add_output_args(parser)

    args = parser.parse_args()
    setup_from_args(args)
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from src.shared.python.logging_config import LogLevel, get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

# Type alias for main functions
MainFunction = Callable[[argparse.Namespace], int | None]


def create_base_parser(
    description: str,
    *,
    prog: str | None = None,
    epilog: str | None = None,
    add_help: bool = True,
    formatter_class: type[
        argparse.HelpFormatter
    ] = argparse.RawDescriptionHelpFormatter,
) -> argparse.ArgumentParser:
    """Create a base ArgumentParser with standard configuration.

    Args:
        description: Program description.
        prog: Program name (default: script name).
        epilog: Text to display after argument help.
        add_help: Add -h/--help argument.
        formatter_class: Help formatter class.

    Returns:
        Configured ArgumentParser.

    Example:
        >>> parser = create_base_parser("Process golf data")
        >>> parser.add_argument("input_file", type=Path)
        >>> args = parser.parse_args()
    """
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        add_help=add_help,
        formatter_class=formatter_class,
    )


def add_logging_args(
    parser: argparse.ArgumentParser,
    *,
    default_level: str = "INFO",
    include_file: bool = True,
) -> argparse.ArgumentParser:
    """Add logging-related arguments to parser.

    Adds:
        -v, --verbose: Enable verbose output (DEBUG level)
        -q, --quiet: Quiet mode (WARNING level)
        --log-level: Explicit log level
        --log-file: Log file path (if include_file=True)

    Args:
        parser: ArgumentParser to add arguments to.
        default_level: Default log level.
        include_file: Include --log-file argument.

    Returns:
        The parser (for chaining).

    Example:
        >>> parser = create_base_parser("My script")
        >>> add_logging_args(parser)
    """
    group = parser.add_argument_group("logging options")

    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )

    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode (WARNING level only)",
    )

    group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=default_level,
        help=f"Set log level (default: {default_level})",
    )

    if include_file:
        group.add_argument(
            "--log-file",
            type=Path,
            metavar="PATH",
            help="Write logs to file",
        )

    return parser


def add_output_args(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path | str | None = None,
    include_format: bool = False,
) -> argparse.ArgumentParser:
    """Add output-related arguments to parser.

    Adds:
        -o, --output: Output path (file or directory)
        --overwrite: Overwrite existing files
        --format: Output format (if include_format=True)

    Args:
        parser: ArgumentParser to add arguments to.
        default_output: Default output path.
        include_format: Include --format argument.

    Returns:
        The parser (for chaining).

    Example:
        >>> parser = create_base_parser("Export data")
        >>> add_output_args(parser, include_format=True)
    """
    group = parser.add_argument_group("output options")

    group.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        metavar="PATH",
        help="Output path (file or directory)",
    )

    group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    if include_format:
        group.add_argument(
            "--format",
            choices=["json", "yaml", "csv", "parquet"],
            default="json",
            help="Output format (default: json)",
        )

    return parser


def add_config_args(
    parser: argparse.ArgumentParser,
    *,
    default_config: Path | str | None = None,
) -> argparse.ArgumentParser:
    """Add configuration-related arguments to parser.

    Adds:
        -c, --config: Configuration file path
        --no-config: Skip loading configuration file

    Args:
        parser: ArgumentParser to add arguments to.
        default_config: Default config file path.

    Returns:
        The parser (for chaining).

    Example:
        >>> parser = create_base_parser("Run simulation")
        >>> add_config_args(parser, default_config="config.yaml")
    """
    group = parser.add_argument_group("configuration options")

    group.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config,
        metavar="PATH",
        help="Configuration file path",
    )

    group.add_argument(
        "--no-config",
        action="store_true",
        help="Skip loading configuration file",
    )

    return parser


def add_simulation_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add simulation-related arguments to parser.

    Adds:
        --time-step: Simulation time step
        --duration: Simulation duration
        --engine: Physics engine to use

    Args:
        parser: ArgumentParser to add arguments to.

    Returns:
        The parser (for chaining).

    Example:
        >>> parser = create_base_parser("Run physics simulation")
        >>> add_simulation_args(parser)
    """
    group = parser.add_argument_group("simulation options")

    group.add_argument(
        "--time-step",
        type=float,
        default=0.001,
        metavar="SECONDS",
        help="Simulation time step in seconds (default: 0.001)",
    )

    group.add_argument(
        "--duration",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Simulation duration in seconds (default: 10.0)",
    )

    group.add_argument(
        "--engine",
        choices=["mujoco", "pinocchio", "drake", "opensim"],
        help="Physics engine to use",
    )

    return parser


def add_dry_run_arg(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add --dry-run argument to parser.

    Args:
        parser: ArgumentParser to add argument to.

    Returns:
        The parser (for chaining).
    """
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    return parser


def add_force_arg(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add --force argument to parser.

    Args:
        parser: ArgumentParser to add argument to.

    Returns:
        The parser (for chaining).
    """
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force operation without confirmation",
    )
    return parser


def add_parallel_args(
    parser: argparse.ArgumentParser,
    *,
    default_workers: int | None = None,
) -> argparse.ArgumentParser:
    """Add parallel processing arguments to parser.

    Adds:
        -j, --jobs: Number of parallel workers
        --sequential: Disable parallel processing

    Args:
        parser: ArgumentParser to add arguments to.
        default_workers: Default number of workers (None = CPU count).

    Returns:
        The parser (for chaining).
    """
    group = parser.add_argument_group("parallelization options")

    group.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_workers,
        metavar="N",
        help="Number of parallel workers (default: CPU count)",
    )

    group.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel processing",
    )

    return parser


def setup_from_args(
    args: argparse.Namespace,
    *,
    logger_name: str | None = None,
) -> None:
    """Setup logging based on parsed arguments.

    Args:
        args: Parsed arguments from ArgumentParser.
        logger_name: Name for the logger (default: root).

    Example:
        >>> args = parser.parse_args()
        >>> setup_from_args(args)
    """
    # Determine log level from args
    level = LogLevel.INFO

    if hasattr(args, "verbose") and args.verbose:
        level = LogLevel.DEBUG
    elif hasattr(args, "quiet") and args.quiet:
        level = LogLevel.WARNING
    elif hasattr(args, "log_level"):
        level = LogLevel[args.log_level]

    # Get log file path if specified
    log_file = getattr(args, "log_file", None)

    # Setup logging
    setup_logging(
        level=level,
        filename=log_file,
    )


def get_effective_log_level(args: argparse.Namespace) -> str:
    """Get the effective log level from parsed arguments.

    Args:
        args: Parsed arguments.

    Returns:
        Log level string.
    """
    if hasattr(args, "verbose") and args.verbose:
        return "DEBUG"
    if hasattr(args, "quiet") and args.quiet:
        return "WARNING"
    if hasattr(args, "log_level"):
        return str(args.log_level)
    return "INFO"


def resolve_output_path(
    args: argparse.Namespace,
    *,
    default_name: str = "output",
    extension: str = "",
) -> Path:
    """Resolve output path from arguments.

    Args:
        args: Parsed arguments (should have 'output' attribute).
        default_name: Default filename if output is a directory.
        extension: File extension to add if needed.

    Returns:
        Resolved output path.

    Example:
        >>> args.output = Path("./results")
        >>> resolve_output_path(args, default_name="report", extension=".json")
        PosixPath('./results/report.json')
    """
    output = getattr(args, "output", None)

    if output is None:
        output = Path.cwd()

    output = Path(output)

    # If output is a directory, append default filename
    if output.is_dir() or (not output.suffix and not output.exists()):
        if extension and not default_name.endswith(extension):
            default_name = f"{default_name}{extension}"
        output = output / default_name

    # Add extension if needed
    if extension and not output.suffix:
        output = output.with_suffix(extension)

    return output


def validate_input_files(
    paths: Sequence[Path | str],
    *,
    must_exist: bool = True,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Validate input file paths.

    Args:
        paths: Paths to validate.
        must_exist: Require files to exist.
        extensions: Allowed file extensions.

    Returns:
        List of validated Path objects.

    Raises:
        argparse.ArgumentTypeError: If validation fails.
    """
    result = []

    for p in paths:
        path = Path(p)

        if must_exist and not path.exists():
            raise argparse.ArgumentTypeError(f"File does not exist: {path}")

        if extensions and path.suffix not in extensions:
            raise argparse.ArgumentTypeError(
                f"Invalid file type: {path.suffix}. "
                f"Allowed: {', '.join(extensions)}"
            )

        result.append(path)

    return result


def path_type(
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> type:
    """Create a path type for argparse.

    Args:
        must_exist: Path must exist.
        must_be_file: Path must be a file.
        must_be_dir: Path must be a directory.

    Returns:
        A type function for argparse.

    Example:
        >>> parser.add_argument("input", type=path_type(must_exist=True, must_be_file=True))
    """

    def _path_type(value: str) -> Path:
        path = Path(value)

        if must_exist and not path.exists():
            raise argparse.ArgumentTypeError(f"Path does not exist: {path}")

        if must_be_file and path.exists() and not path.is_file():
            raise argparse.ArgumentTypeError(f"Path is not a file: {path}")

        if must_be_dir and path.exists() and not path.is_dir():
            raise argparse.ArgumentTypeError(f"Path is not a directory: {path}")

        return path

    return _path_type  # type: ignore[return-value]


def positive_int(value: str) -> int:
    """Argparse type for positive integers.

    Example:
        >>> parser.add_argument("--count", type=positive_int)
    """
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"Must be positive: {value}")
        return ivalue
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}") from e


def non_negative_int(value: str) -> int:
    """Argparse type for non-negative integers.

    Example:
        >>> parser.add_argument("--index", type=non_negative_int)
    """
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"Must be non-negative: {value}")
        return ivalue
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}") from e


def positive_float(value: str) -> float:
    """Argparse type for positive floats.

    Example:
        >>> parser.add_argument("--rate", type=positive_float)
    """
    try:
        fvalue = float(value)
        if fvalue <= 0:
            raise argparse.ArgumentTypeError(f"Must be positive: {value}")
        return fvalue
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float: {value}") from e


def run_main(
    main_func: MainFunction,
    parser: argparse.ArgumentParser,
    *,
    setup_logging_from_args: bool = True,
) -> int:
    """Run a main function with standard error handling.

    Args:
        main_func: Main function to run (takes args, returns exit code or None).
        parser: ArgumentParser to use.
        setup_logging_from_args: Setup logging from parsed args.

    Returns:
        Exit code (0 = success, non-zero = error).

    Example:
        >>> def main(args):
        ...     print(f"Processing {args.input}")
        ...     return 0
        >>> parser = create_base_parser("Process files")
        >>> parser.add_argument("input")
        >>> import sys
        >>> sys.exit(run_main(main, parser))
    """
    logger = get_logger(__name__)

    try:
        args = parser.parse_args()

        if setup_logging_from_args:
            setup_from_args(args)

        result = main_func(args)
        return result if result is not None else 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
