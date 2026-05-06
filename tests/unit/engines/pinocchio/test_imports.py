"""Test that all modules can be imported."""

import pytest


def test_import_pinocchio_golf() -> None:
    """Test importing the main package."""
    pinocchio_golf = pytest.importorskip(
        "python.pinocchio_golf",
        reason="python.pinocchio_golf not available (install pinocchio engine)",
    )
    assert pinocchio_golf is not None


def test_import_gui() -> None:
    """Test importing the GUI module."""
    pytest.importorskip("pinocchio")
    pytest.importorskip(
        "python.pinocchio_golf",
        reason="python.pinocchio_golf not available",
    )
    from python.pinocchio_golf import gui  # noqa: PLC0415

    assert gui is not None


def test_import_coppelia_bridge() -> None:
    """Test importing the Coppelia bridge."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("zmqRemoteApi")
    from python.pinocchio_golf import coppelia_bridge  # noqa: PLC0415

    assert coppelia_bridge is not None


def test_import_poly_torque_util() -> None:
    """Test importing the poly torque util module."""
    pytest.importorskip(
        "python.pinocchio_golf",
        reason="python.pinocchio_golf not available (install pinocchio engine)",
    )
    from python.pinocchio_golf import poly_torque_util  # noqa: PLC0415

    assert poly_torque_util is not None
