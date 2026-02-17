"""Shared fixtures and utilities for the Golf Modeling Suite test suite.

This module centralizes common setup logic to improve test orthogonality
and adherence to the DRY principle.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Engine module prefixes whose sys.modules entries must be isolated between
# tests.  Pinocchio's C extension (pinocchio_pywrap_default) is corrupted by
# PinocchioProbe.probe(); Drake gets replaced with MagicMock objects by tests
# that mock pydrake, causing downstream TypeError comparisons. Drake engine
# modules also get polluted when imported with different paths (src.engines.*
# vs engines.*), breaking test_drake_wrapper.py.
_PROTECTED_PREFIXES = (
    "pinocchio",
    "pydrake",
)


def _matches_protected(name: str) -> bool:
    """Return True if *name* is a protected engine module."""
    for prefix in _PROTECTED_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


@pytest.fixture(autouse=True)
def _protect_engine_modules() -> Generator[None, None, None]:
    """Prevent engine module state corruption from leaking between tests.

    Several tests instantiate ``EngineManager`` or ``GolfLauncher`` which
    trigger engine probes that import pinocchio/drake.  The probes can corrupt
    C extension module state or leave MagicMock objects in ``sys.modules``.
    Subsequent tests then fail with ``NameError`` or ``TypeError``.

    This fixture snapshots all engine-related ``sys.modules`` entries before
    each test and restores them afterward so that corruption cannot leak
    across test boundaries.
    """
    protected_keys = {k for k in sys.modules if _matches_protected(k)}
    saved = {k: sys.modules[k] for k in protected_keys}
    yield
    # Remove any engine modules added or mutated during the test
    for k in list(sys.modules):  # list() needed: mutating dict during iteration
        if _matches_protected(k):
            if k in saved:
                sys.modules[k] = saved[k]
            else:
                del sys.modules[k]
    # Restore any that were removed during the test
    for k, v in saved.items():
        if k not in sys.modules:
            sys.modules[k] = v


@pytest.fixture
def pendulum_urdf(tmp_path: Path) -> str:
    """Create a standardized simple pendulum URDF for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="pendulum">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""
    urdf_path = tmp_path / "pendulum.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


@pytest.fixture
def clean_pendulum_dynamics():
    """Fixture to provide standardized DoublePendulumDynamics setup for unit tests."""
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DoublePendulumDynamics,
        DoublePendulumParameters,
        LowerSegmentProperties,
        SegmentProperties,
    )

    def _create(m1_kg=1.0, l1_m=1.0):
        upper_segment = SegmentProperties(
            length_m=l1_m,
            mass_kg=m1_kg,
            center_of_mass_ratio=1.0,
            inertia_about_com=0.0,
        )
        # Quasi-massless link 2
        epsilon_kg = 1e-10
        lower_segment = LowerSegmentProperties(
            length_m=1.0,
            shaft_mass_kg=epsilon_kg,
            clubhead_mass_kg=epsilon_kg,
            shaft_com_ratio=0.5,
        )
        params = DoublePendulumParameters(
            upper_segment=upper_segment,
            lower_segment=lower_segment,
            plane_inclination_deg=0.0,
            damping_shoulder=0.0,
            damping_wrist=0.0,
            gravity_enabled=True,
            constrained_to_plane=True,
        )
        return DoublePendulumDynamics(parameters=params)

    return _create


# Mock classes that need to be defined before importing the engine
class MockPhysicsEngine:
    pass


@pytest.fixture
def mock_drake_dependencies():
    """Fixture to mock pydrake and interfaces safely.

    This fixture mocks pydrake modules to allow testing Drake integration
    without having Drake installed.
    """
    mock_pydrake = MagicMock()
    mock_interfaces = MagicMock()
    mock_interfaces.PhysicsEngine = MockPhysicsEngine

    with patch.dict(
        "sys.modules",
        {
            "pydrake": mock_pydrake,
            "pydrake.math": MagicMock(),
            "pydrake.multibody": MagicMock(),
            "pydrake.multibody.plant": MagicMock(),
            "pydrake.multibody.parsing": MagicMock(),
            "pydrake.systems": MagicMock(),
            "pydrake.systems.framework": MagicMock(),
            "pydrake.systems.analysis": MagicMock(),
            "pydrake.all": MagicMock(),
            "shared.python.interfaces": mock_interfaces,
        },
    ):
        yield mock_pydrake, mock_interfaces


@pytest.fixture(scope="module")
def mock_mujoco_dependencies():
    """Fixture to mock mujoco and interfaces safely.

    This fixture mocks mujoco modules to allow testing MuJoCo integration
    without having MuJoCo installed.
    """
    mock_mujoco = MagicMock()
    mock_interfaces = MagicMock()
    mock_interfaces.PhysicsEngine = MockPhysicsEngine

    # Create common MuJoCo structure mocks
    # These are needed for attribute access in many tests
    mock_model = MagicMock()
    mock_model.nv = 2
    mock_model.nu = 2
    mock_model.nq = 2
    mock_model.nbody = 2

    mock_data = MagicMock()
    mock_data.qpos = MagicMock()
    mock_data.qvel = MagicMock()
    mock_data.qacc = MagicMock()
    mock_data.ctrl = MagicMock()

    mock_mujoco.MjModel.return_value = mock_model
    mock_mujoco.MjData.return_value = mock_data

    with patch.dict(
        "sys.modules",
        {
            "mujoco": mock_mujoco,
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.interfaces": mock_interfaces,
        },
    ):
        yield mock_mujoco, mock_interfaces
