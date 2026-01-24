"""Pytest configuration and shared fixtures.

This conftest.py centralizes path setup and common fixtures to avoid
duplicating sys.path manipulation across test files.

Refactored to follow DRY principle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Ensure repo root is in path - do this once here instead of in every test file
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

# Add common paths to sys.path
_paths_to_add = [
    str(REPO_ROOT),
    str(SRC_ROOT),
    str(SRC_ROOT / "shared" / "python"),
    str(SRC_ROOT / "engines" / "physics_engines" / "mujoco" / "python"),
    str(
        SRC_ROOT
        / "engines"
        / "physics_engines"
        / "mujoco"
        / "python"
        / "mujoco_humanoid_golf"
    ),
    str(
        SRC_ROOT
        / "engines"
        / "Simscape_Multibody_Models"
        / "3D_Golf_Model"
        / "python"
        / "src"
    ),
    str(
        SRC_ROOT
        / "engines"
        / "Simscape_Multibody_Models"
        / "2D_Golf_Model"
        / "python"
        / "src"
    ),
]

for path in _paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)


# Import shared utilities after path setup
if TYPE_CHECKING:
    pass


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root path."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def src_root() -> Path:
    """Return the src directory path."""
    return SRC_ROOT


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Return the data directory path."""
    return REPO_ROOT / "data"


@pytest.fixture(scope="session")
def models_root() -> Path:
    """Return the models directory path."""
    return REPO_ROOT / "models"


# Engine availability fixtures
@pytest.fixture(scope="session")
def mujoco_available() -> bool:
    """Check if MuJoCo is available."""
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def pinocchio_available() -> bool:
    """Check if Pinocchio is available."""
    try:
        import pinocchio  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def pyqt6_available() -> bool:
    """Check if PyQt6 is available."""
    try:
        from PyQt6 import QtWidgets  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# Skip markers based on engine availability
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "mujoco: mark test as requiring MuJoCo")
    config.addinivalue_line("markers", "pinocchio: mark test as requiring Pinocchio")
    config.addinivalue_line("markers", "gui: mark test as requiring GUI components")
