"""
tests/heavy_integration/conftest.py
====================================
Shared fixtures for heavy integration tests across all repos.
These fixtures are live_simulation-safe: they set up headless
display, temp directories, and shared expensive resources.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


# ── Display Fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def headless_display() -> None:
    """
    Ensures a virtual framebuffer is available for GUI/rendering tests.
    On CI, xvfb-run wraps the entire pytest session.
    Locally, this fixture checks DISPLAY is set.
    """
    display = os.environ.get("DISPLAY", "")
    if not display:
        # Warn but don't fail — some tests don't need display
        import warnings
        warnings.warn(
            "DISPLAY not set. GUI tests may fail. "
            "Run via: xvfb-run pytest ... or wsl bash run_local_heavy_tests.sh",
            stacklevel=1,
        )


# ── Temp Directory ────────────────────────────────────────────────────────────

@pytest.fixture()
def temp_test_dir() -> Path:
    """Provides a clean temporary directory for each test."""
    with tempfile.TemporaryDirectory(prefix="heavy_test_") as tmpdir:
        yield Path(tmpdir)


# ── Physics Engine Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mujoco_model():
    """
    Session-scoped MuJoCo model fixture using the built-in humanoid model.
    Loaded once per test session to amortize startup cost.
    """
    try:
        import mujoco
        xml = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data
    except ImportError:
        pytest.skip("mujoco not installed")


@pytest.fixture(scope="session")
def pinocchio_model():
    """
    Session-scoped Pinocchio model fixture.
    Returns a simple 1-DOF pendulum model for FK testing.
    """
    try:
        import pinocchio as pin
        import numpy as np
        model = pin.Model()
        geom_model = pin.GeometryModel()
        joint_id = model.addJoint(
            0, pin.JointModelRY(), pin.SE3.Identity(), "joint1"
        )
        model.appendBodyToJoint(
            joint_id,
            pin.Inertia(1.0, np.zeros(3), np.eye(3)),
            pin.SE3.Identity(),
        )
        return model
    except ImportError:
        pytest.skip("pinocchio not installed")
