"""Test that all modules can be imported."""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_import_pinocchio_golf() -> None:
    """Test importing the main package."""
    import python.pinocchio_golf

    assert python.pinocchio_golf is not None


def test_import_gui() -> None:
    """Test importing the GUI module."""
    pytest.importorskip("pinocchio")
    from python.pinocchio_golf import gui

    assert gui is not None


def test_import_coppelia_bridge() -> None:
    """Test importing the Coppelia bridge."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("zmqRemoteApi")
    from python.pinocchio_golf import coppelia_bridge

    assert coppelia_bridge is not None


def test_import_torque_fitting() -> None:
    """Test importing the torque fitting module."""
    from python.pinocchio_golf import torque_fitting

    assert torque_fitting is not None
