"""Heavy integration tests for Drake real model loading (fixes #1985).

When Drake IS installed in the heavy Docker image, these tests exercise actual
model loading and simulation — not mocked pydrake. The mock-heavy tests in
test_phase1_drake_integration.py belong in unit tests (tracked separately).

All tests skip gracefully when Drake is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

# Minimal self-contained URDF for testing
_MINIMAL_URDF = """\
<?xml version="1.0"?>
<robot name="minimal_pendulum">
  <link name="world"/>
  <link name="pendulum_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </visual>
    <collision>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </collision>
  </link>
  <joint name="pivot" type="revolute">
    <parent link="world"/>
    <child link="pendulum_link"/>
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def drake_modules():
    """Import required Drake modules or skip the entire module."""
    try:
        from pydrake.all import DiagramBuilder, Parser  # noqa: F401
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.systems.analysis import Simulator

        return {
            "DiagramBuilder": DiagramBuilder,
            "Parser": Parser,
            "MultibodyPlant": MultibodyPlant,
            "Simulator": Simulator,
        }
    except ImportError as exc:
        pytest.skip(f"Drake (pydrake) not installed: {exc}")


@pytest.fixture(scope="module")
def minimal_urdf_path(tmp_path_factory):
    """Write the minimal URDF to a temp file."""
    tmpdir = tmp_path_factory.mktemp("urdf")
    urdf = tmpdir / "minimal_pendulum.urdf"
    urdf.write_text(_MINIMAL_URDF)
    return urdf


pytestmark = pytest.mark.live_simulation
