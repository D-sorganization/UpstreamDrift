"""
tests/heavy_integration/conftest.py
====================================
Shared fixtures for heavy integration tests across all repos.
These fixtures are live_simulation-safe: they set up headless
display, temp directories, and shared expensive resources.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

_HEAVY_INTEGRATION_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply live_simulation marker to every test in heavy_integration/.

    This ensures CI workflows that filter by ``-m live_simulation`` always
    include all tests in this directory, without requiring each file to
    declare the marker manually.
    """
    marker = pytest.mark.live_simulation
    for item in items:
        try:
            item_path = Path(item.fspath)
        except Exception as e:  # noqa: BLE001, F841
            continue
        if _HEAVY_INTEGRATION_DIR in item_path.parents:
            item.add_marker(marker, append=False)


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
def mujoco_model() -> Any:
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
def pinocchio_model() -> Any:
    """
    Session-scoped Pinocchio model fixture.
    Returns a simple 1-DOF pendulum model for FK testing.
    """
    try:
        import numpy as np
        import pinocchio as pin

        model = pin.Model()
        joint_id = model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "joint1")
        model.appendBodyToJoint(
            joint_id,
            pin.Inertia(1.0, np.zeros(3), np.eye(3)),
            pin.SE3.Identity(),
        )
        return model
    except ImportError:
        pytest.skip("pinocchio not installed")


pytestmark = pytest.mark.live_simulation
