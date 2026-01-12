"""Build script for Windows MSI installer.

This script automates the creation of a professional Windows MSI installer
with modular physics engine selection and proper dependency management.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
INSTALLER_DIR = Path(__file__).parent
BUILD_DIR = INSTALLER_DIR / "build"
DIST_DIR = INSTALLER_DIR / "dist"


def check_prerequisites():
    """Check that all required tools are available."""

    # Check Python version

    # Check cx_Freeze
    try:
        import cx_Freeze

        print(f"✓ cx_Freeze {cx_Freeze.version}")  # noqa: T201
    except ImportError:
        return False

    # Check if we're in a virtual environment (recommended)
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        pass
    else:
        pass

    return True


def clean_build_dirs():
    """Clean previous build artifacts."""

    for dir_path in [BUILD_DIR, DIST_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)


def install_dependencies():
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


def detect_physics_engines():
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


def build_executable():
    """Build the executable using cx_Freeze."""

    # Change to installer directory
    original_cwd = os.getcwd()
    os.chdir(INSTALLER_DIR)

    try:
        # Run setup.py build
        result = subprocess.run(
            [sys.executable, "setup.py", "build"], capture_output=True, text=True
        )

        if result.returncode != 0:
            return False

        return True

    finally:
        os.chdir(original_cwd)


def build_msi():
    """Build the MSI installer."""

    # Change to installer directory
    original_cwd = os.getcwd()
    os.chdir(INSTALLER_DIR)

    try:
        # Run setup.py bdist_msi
        result = subprocess.run(
            [sys.executable, "setup.py", "bdist_msi"], capture_output=True, text=True
        )

        if result.returncode != 0:
            return False

        # Find the generated MSI file
        msi_files = list(DIST_DIR.glob("*.msi"))
        if msi_files:
            msi_files[0]

        return True

    finally:
        os.chdir(original_cwd)


def create_installer_info():
    """Create installer information file."""
    available_engines = detect_physics_engines()

    info = {
        "version": "1.0.0",
        "build_date": "2026-01-12",
        "physics_engines": available_engines,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": "Windows x64",
    }

    info_file = DIST_DIR / "installer_info.json"
    import json

    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)


def main():
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

    args = parser.parse_args()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Clean build directories
    if args.clean:
        clean_build_dirs()

    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            sys.exit(1)

    # Detect available engines
    available_engines = detect_physics_engines()
    if not available_engines:
        sys.exit(1)

    # Build executable
    if not build_executable():
        sys.exit(1)

    # Build MSI (unless exe-only)
    if not args.exe_only:
        if not build_msi():
            sys.exit(1)

        # Create installer info
        create_installer_info()

    # List output files
    output_files = list(DIST_DIR.glob("*"))
    if output_files:
        for file_path in output_files:
            file_path.stat().st_size / (1024 * 1024)


if __name__ == "__main__":
    main()
