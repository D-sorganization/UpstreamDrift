"""Build script for Windows MSI installer.

This script automates the creation of a professional Windows MSI installer
with modular physics engine selection and proper dependency management.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from installer.windows.packaging_profiles import (
    build_profile_environment,
    get_packaging_profile,
    iter_packaging_profile_ids,
)

# Project paths
_this_file = Path(__file__)
_installer_dir = _this_file.parent
PROJECT_ROOT = _installer_dir.parent.parent
INSTALLER_DIR = _installer_dir
BUILD_DIR = INSTALLER_DIR / "build"
DIST_DIR = INSTALLER_DIR / "dist"

logger = logging.getLogger(__name__)


class SetupCommandError(RuntimeError):
    """Raised when an installer setup.py command exits unsuccessfully."""

    def __init__(
        self,
        command: Sequence[str],
        returncode: int,
        stdout: str | None,
        stderr: str | None,
    ) -> None:
        self.command = tuple(str(part) for part in command)
        self.returncode = returncode
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        command_text = " ".join(self.command)
        stdout = self.stdout.rstrip() or "<empty>"
        stderr = self.stderr.rstrip() or "<empty>"
        return (
            "setup.py command failed: "
            f"{command_text} (return code {self.returncode})\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )


def _run_setup_command(
    setup_args: Sequence[str],
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run setup.py and preserve captured output in failures."""
    command = [sys.executable, "setup.py", *setup_args]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise SetupCommandError(
            command, result.returncode, result.stdout, result.stderr
        )
    return result


def check_prerequisites() -> bool:
    """Check that all required tools are available."""

    # Check Python version

    # Check cx_Freeze
    try:
        import cx_Freeze  # type: ignore[import-not-found]

        logger.info("cx_Freeze %s", cx_Freeze.version)
    except ImportError:
        return False

    # Check if we're in a virtual environment (recommended)
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        pass

    return True


def clean_build_dirs() -> None:
    """Clean previous build artifacts."""

    for dir_path in [BUILD_DIR, DIST_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)


def install_dependencies() -> bool:
    """Install required dependencies for building."""

    build_requirements = ["cx_Freeze>=6.15.0", "wheel", "setuptools>=61.0"]

    for requirement in build_requirements:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", requirement],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            return False

    return True


def detect_physics_engines() -> list[str]:
    """Detect which physics engines are available."""

    engines = {
        "mujoco": "mujoco",
        "drake": "pydrake",
        "pinocchio": "pinocchio",
        "myosuite": "myosuite",
        "opensim": "opensim",
    }

    available = []
    for engine_name, module_name in engines.items():
        try:
            __import__(module_name)
            available.append(engine_name)
        except ImportError:
            pass

    return available


def build_executable(
    profile_name: str,
    provider_roots: tuple[str | os.PathLike[str], ...] = (),
) -> bool:
    """Build the executable using cx_Freeze."""
    profile = get_packaging_profile(profile_name)

    # Change to installer directory
    original_cwd = os.getcwd()
    os.chdir(INSTALLER_DIR)

    try:
        _run_setup_command(
            ["build"],
            env=build_profile_environment(profile, provider_roots),
        )
        return True

    finally:
        os.chdir(original_cwd)


def build_msi(
    profile_name: str,
    provider_roots: tuple[str | os.PathLike[str], ...] = (),
) -> bool:
    """Build the MSI installer."""
    profile = get_packaging_profile(profile_name)

    # Change to installer directory
    original_cwd = os.getcwd()
    os.chdir(INSTALLER_DIR)

    try:
        _run_setup_command(
            ["bdist_msi"],
            env=build_profile_environment(profile, provider_roots),
        )

        # Find the generated MSI file
        msi_files = list(DIST_DIR.glob("*.msi"))
        if msi_files:
            msi_files[0]

        return True

    finally:
        os.chdir(original_cwd)


def create_installer_info(
    profile_name: str,
    provider_roots: tuple[str | os.PathLike[str], ...] = (),
) -> None:
    """Create installer information file."""
    profile = get_packaging_profile(profile_name)
    available_engines = detect_physics_engines()
    major, minor, micro = sys.version_info[:3]

    info = {
        "version": "1.0.0",
        "build_date": "2026-01-12",
        "packaging_profile": profile.profile_id,
        "profile_display_name": profile.display_name,
        "description": profile.description,
        "discovery_mode": profile.discovery_mode,
        "physics_engines": available_engines,
        "supported_provider_ids": list(profile.supported_provider_ids),
        "provider_roots": [str(Path(root)) for root in provider_roots],
        "python_version": f"{major}.{minor}.{micro}",
        "platform": "Windows x64",
    }

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    info_file = DIST_DIR / "installer_info.json"

    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _log_generated_outputs(output_files: list[Path]) -> None:
    """Log generated installer artifacts with their sizes."""
    for file_path in output_files:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info("Generated %s (%.2f MB)", file_path.name, size_mb)


def main() -> None:
    """Main build process."""
    parser = argparse.ArgumentParser(
        description="Build Golf Modeling Suite Windows installer"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean build directories first"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument(
        "--exe-only", action="store_true", help="Build executable only (no MSI)"
    )
    parser.add_argument(
        "--profile",
        choices=iter_packaging_profile_ids(),
        default="hybrid",
        help="Packaging profile to build",
    )
    parser.add_argument(
        "--provider-root",
        action="append",
        default=[],
        help="Optional external provider repository root for hybrid/full builds",
    )

    args = parser.parse_args()
    provider_roots = tuple(args.provider_root)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Clean build directories
    if args.clean:
        clean_build_dirs()

    # Install dependencies
    if not args.skip_deps and not install_dependencies():
        sys.exit(1)

    # Detect available engines
    available_engines = detect_physics_engines()
    if not available_engines:
        sys.exit(1)

    try:
        # Build executable
        if not build_executable(args.profile, provider_roots):
            sys.exit(1)

        # Build MSI (unless exe-only)
        if not args.exe_only:
            if not build_msi(args.profile, provider_roots):
                sys.exit(1)

            # Create installer info
            create_installer_info(args.profile, provider_roots)
    except SetupCommandError as exc:
        sys.stderr.write(f"{exc}\n")
        sys.exit(1)

    # List output files
    output_files = list(DIST_DIR.glob("*"))
    if output_files:
        _log_generated_outputs(output_files)


if __name__ == "__main__":
    main()
