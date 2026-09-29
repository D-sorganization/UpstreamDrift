"""
FreeMoCap Launcher - Subprocess launcher for FreeMoCap sidecar pipeline.

This module handles spawning FreeMoCap in an isolated Python environment,
managing the subprocess lifecycle, and capturing logs/output.
"""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LaunchConfig:
    """Configuration for launching FreeMoCap."""

    session_dir: Path
    freemocap_env: Path | None = None
    timeout_seconds: int = 3600
    video_dir: Path | None = None
    output_dir: Path | None = None
    headless: bool = True
    extra_args: list[str] | None = None


@dataclass
class LaunchResult:
    """Result of a FreeMoCap launch attempt."""

    success: bool
    return_code: int
    output_dir: Path | None
    log_file: Path | None
    error_message: str | None = None


class FreeMoCapLauncher:
    """
    Launcher for FreeMoCap sidecar pipeline.

    This class manages the subprocess execution of FreeMoCap in an isolated
    Python environment, keeping the AGPL-licensed code separate from the
    main UpstreamDrift process.

    Usage:
        launcher = FreeMoCapLauncher()
        config = LaunchConfig(session_dir=Path("/path/to/session"))
        result = launcher.launch(config)
    """

    DEFAULT_FREENOCAP_ENV_NAME = "freemocap-env"
    DEFAULT_OUTPUT_SUBDIR = "freemocap_output"

    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the launcher.

        Args:
            log_level: Logging level for launcher output.
        """
        self.log_level = log_level

    def _find_freemocap_python(self, env_path: Path | None) -> str | None:
        """
        Find the Python interpreter in the FreeMoCap environment.

        Args:
            env_path: Path to the conda/venv environment, or None to search.

        Returns:
            Path to python executable or None if not found.
        """
        if env_path is not None:
            # Try as venv
            venv_python = env_path / "bin" / "python"
            if venv_python.exists():
                return str(venv_python)
            # Try as conda env
            conda_python = env_path / "bin" / "python"
            if conda_python.exists():
                return str(conda_python)

        # Search for conda env
        conda_env = (
            Path.home() / "miniconda3" / "envs" / self.DEFAULT_FREENOCAP_ENV_NAME
        )
        if conda_env.exists():
            python_path = conda_env / "bin" / "python"
            if python_path.exists():
                return str(python_path)

        # Search for venv in common locations
        common_paths = [
            Path.home() / ".venvs" / self.DEFAULT_FREENOCAP_ENV_NAME,
            Path.home() / self.DEFAULT_FREENOCAP_ENV_NAME,
            Path("/opt/freemocap-env"),
        ]
        for p in common_paths:
            if p.exists():
                python_path = p / "bin" / "python"
                if python_path.exists():
                    return str(python_path)

        return None

    def _build_command(self, config: LaunchConfig, python_exe: str) -> list[str]:
        """
        Build the command to launch FreeMoCap.

        Args:
            config: Launch configuration.
            python_exe: Path to Python interpreter.

        Returns:
            Command list for subprocess.
        """
        # Base command - run freemocap module
        cmd = [python_exe, "-m", "freemocap"]

        if config.headless:
            cmd.append("--headless")

        # Session/output directories
        cmd.extend(["--session_id", config.session_dir.name])
        cmd.extend(["--session_output_path", str(config.session_dir)])

        # Video directory if specified
        if config.video_dir is not None:
            cmd.extend(["--video_path", str(config.video_dir)])

        # Output directory
        output_dir = config.output_dir or (
            config.session_dir / self.DEFAULT_OUTPUT_SUBDIR
        )
        cmd.extend(["--output_path", str(output_dir)])

        # Extra arguments
        if config.extra_args:
            cmd.extend(config.extra_args)

        return cmd

    def _setup_logging(self, session_dir: Path) -> Path:
        """
        Setup log file for the capture session.

        Args:
            session_dir: Session directory.

        Returns:
            Path to log file.
        """
        log_dir = session_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"freemocap_{timestamp}.log"
        return log_file

    def launch(self, config: LaunchConfig) -> LaunchResult:
        """
        Launch FreeMoCap in a subprocess.

        Args:
            config: Launch configuration.

        Returns:
            LaunchResult indicating success/failure and output location.
        """
        session_dir = Path(config.session_dir).expanduser().resolve()
        if not session_dir.exists():
            return LaunchResult(
                success=False,
                return_code=-1,
                output_dir=None,
                log_file=None,
                error_message=f"Session directory does not exist: {session_dir}",
            )

        # Find FreeMoCap Python environment
        freemocap_env = config.freemocap_env
        python_exe = self._find_freemocap_python(freemocap_env)
        if python_exe is None:
            return LaunchResult(
                success=False,
                return_code=-1,
                output_dir=None,
                log_file=None,
                error_message=(
                    f"FreeMoCap environment not found. "
                    f"Please create '{self.DEFAULT_FREENOCAP_ENV_NAME}' "
                    f"using the setup script."
                ),
            )

        logger.info(f"Found FreeMoCap Python: {python_exe}")

        # Setup logging
        log_file = self._setup_logging(session_dir)
        logger.info(f"Logging to: {log_file}")

        # Build command
        cmd = self._build_command(config, python_exe)
        logger.info(f"Running command: {' '.join(cmd)}")

        # Launch subprocess
        try:
            with open(log_file, "w") as log_fh:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(session_dir),
                )

                logger.info(f"FreeMoCap started with PID {process.pid}")

                # Wait for completion with timeout
                try:
                    return_code = process.wait(timeout=config.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    return LaunchResult(
                        success=False,
                        return_code=-1,
                        output_dir=None,
                        log_file=log_file,
                        error_message=f"FreeMoCap timed out after {config.timeout_seconds}s",
                    )

                if return_code == 0:
                    output_dir = config.output_dir or (
                        session_dir / self.DEFAULT_OUTPUT_SUBDIR
                    )
                    logger.info("FreeMoCap completed successfully")
                    logger.info(f"Output directory: {output_dir}")

                    return LaunchResult(
                        success=True,
                        return_code=return_code,
                        output_dir=output_dir,
                        log_file=log_file,
                    )
                return LaunchResult(
                    success=False,
                    return_code=return_code,
                    output_dir=None,
                    log_file=log_file,
                    error_message=f"FreeMoCap exited with code {return_code}",
                )

        except Exception as e:
            logger.exception("Error launching FreeMoCap")
            return LaunchResult(
                success=False,
                return_code=-1,
                output_dir=None,
                log_file=log_file if "log_file" in locals() else None,
                error_message=str(e),
            )


def main():
    """CLI entrypoint for running FreeMoCap launcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch FreeMoCap motion capture pipeline"
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Session directory for capture data",
    )
    parser.add_argument(
        "--freemocap-env",
        type=Path,
        help="Path to FreeMoCap Python environment",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with GUI (not headless)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = LaunchConfig(
        session_dir=args.session_dir,
        freemocap_env=args.freemocap_env,
        timeout_seconds=args.timeout,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        headless=not args.gui,
    )

    launcher = FreeMoCapLauncher()
    result = launcher.launch(config)

    if result.success:
        sys.exit(0)
    else:
        if result.log_file:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
