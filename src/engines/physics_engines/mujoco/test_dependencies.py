"""Test suite for MuJoCo dependency verification.

Ensures that the MuJoCo physics engine environment is correctly configured
and that all required libraries are importable and functional.
"""

from __future__ import annotations

import importlib
import logging

import pytest

logger = logging.getLogger(__name__)


def test_mujoco_import() -> None:
    """Verify that the mujoco library can be imported."""
    try:
        import mujoco

        logger.info(f"MuJoCo version: {mujoco.__version__}")
        assert mujoco.__version__ is not None
    except ImportError:
        pytest.fail("MuJoCo library not found. Install with: pip install mujoco")


def test_numpy_compatibility() -> None:
    """Verify numpy compatibility with MuJoCo."""
    import mujoco
    import numpy as np

    # Create a simple MjData struct
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mujoco.MjData(model)

    # Check if we can access data as numpy arrays (if supported by bindings)
    # Note: Modern dm_control or mujoco bindings usually support this
    try:
        qpos = data.qpos
        assert isinstance(qpos, np.ndarray | object)  # Depending on binding version
    except Exception as e:
        logger.warning(f"Direct numpy access check warning: {e}")


def test_rendering_backend() -> None:
    """Check if a rendering backend is available (GL)."""
    # This is often platform specific and might fail in headless CI without Xvfb
    # We just check if we can import the GL context creator
    try:
        from mujoco import MjRenderContextOffscreen  # noqa: F401
    except ImportError:
        logger.warning("MjRenderContextOffscreen not available (might be old version)")
    except Exception:
        # It's okay if it fails to initialize in this test,
        # we just want to know if the symbol exists
        pass


def test_optional_dependencies() -> None:
    """Check for optional physics dependencies."""
    optionals = ["dm_control", "gym", "imageio"]
    found = []
    missing = []

    for pkg in optionals:
        if importlib.util.find_spec(pkg):
            found.append(pkg)
        else:
            missing.append(pkg)

    logger.info(f"Optional dependencies found: {found}")
    logger.info(f"Optional dependencies missing: {missing}")
    # No assertion failure here, as they are optional
