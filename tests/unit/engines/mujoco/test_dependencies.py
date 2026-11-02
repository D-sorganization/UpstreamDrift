"""Test suite for MuJoCo dependency verification.

Ensures that the MuJoCo physics engine environment is correctly configured
and that all required libraries are importable and functional.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def test_mujoco_import() -> None:
    """Verify that the mujoco library can be imported."""
    try:
        import mujoco

        logger.info(f"MuJoCo version: {mujoco.__version__}")
        assert mujoco.__version__ is not None
    except ImportError:
        pytest.fail("MuJoCo library not found. Install with: pip install mujoco")


_TWO_DOF_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.1"/>
      <body pos="0 0 -0.5">
        <joint name="slide" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_numpy_compatibility() -> None:
    """qpos is a finite float64 ndarray of shape (nq,) that supports read/write."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(_TWO_DOF_XML)
    data = mujoco.MjData(model)

    assert model.nq == 2
    qpos = data.qpos
    assert isinstance(qpos, np.ndarray)
    assert qpos.shape == (model.nq,)
    assert qpos.dtype == np.float64
    assert np.all(np.isfinite(qpos))

    data.qpos[:] = [0.25, -0.1]
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qpos, [0.25, -0.1])
    assert np.all(np.isfinite(data.xpos))
    # Writes go through to the underlying MjData buffer, not a copy.
    assert np.shares_memory(qpos, data.qpos)


def test_rendering_api_symbols_present() -> None:
    """Import-only probe: the rendering API symbols exist.

    Headless rendering is not a required capability of this dependency suite,
    so no GL context is created and no window is ever opened. A missing
    symbol in a provisioned MuJoCo install fails.
    """
    import mujoco

    for name in ("Renderer", "MjrContext", "MjvScene"):
        assert hasattr(mujoco, name), f"mujoco.{name} missing"


def test_optional_dependencies() -> None:
    """Check for optional physics dependencies."""
    optionals = ["dm_control", "gym", "imageio"]
    found = []
    missing = []

    for pkg in optionals:
        try:
            if importlib.util.find_spec(pkg):
                found.append(pkg)
            else:
                missing.append(pkg)
        except (ValueError, ModuleNotFoundError):
            missing.append(pkg)

    logger.info(f"Optional dependencies found: {found}")
    logger.info(f"Optional dependencies missing: {missing}")
    # No assertion failure here, as they are optional
