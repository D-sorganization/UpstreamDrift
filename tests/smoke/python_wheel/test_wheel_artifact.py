"""Smoke tests for the built Python wheel artifact."""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
DIST_DIR = REPO_ROOT / "dist"
pytestmark = pytest.mark.smoke


def _latest_wheel() -> Path:
    wheels = sorted(DIST_DIR.glob("upstream_drift-*.whl"))
    if not wheels:
        raise AssertionError("Build the wheel first: python -m build --wheel")
    return wheels[-1]


def _venv_python(tmp_path: Path) -> Path:
    """Create a clean venv and return its Python executable."""
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    return venv_dir / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )


def _console_script(python_bin: Path) -> Path:
    """Return the platform-specific upstream-drift console script path."""
    if sys.platform == "win32":
        return python_bin.parent / "upstream-drift.exe"
    return python_bin.parent / "upstream-drift"


def _install_wheel(python_bin: Path) -> None:
    """Install the latest built wheel into the target interpreter."""
    wheel_path = _latest_wheel()
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", str(wheel_path)],
        check=True,
    )


def test_wheel_installs_in_clean_venv(tmp_path: Path) -> None:
    """Install the built wheel into a clean venv instead of importing the source tree."""
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin)
    subprocess.run(
        [
            str(python_bin),
            "-c",
            "from src.api._version import __version__; print(__version__)",
        ],
        check=True,
    )


def test_console_script_help_runs_from_installed_wheel(tmp_path: Path) -> None:
    """The public console script must be generated and able to render help."""
    python_bin = _venv_python(tmp_path)
    _install_wheel(python_bin)
    console_script = _console_script(python_bin)

    assert console_script.is_file()
    subprocess.run(
        [str(console_script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
