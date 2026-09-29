"""Live integration test for the real Pinocchio bridge (issue #4963).

Skipped when ``pinocchio`` is not importable, OR when the local
install is the PyPI ``pinocchio`` 0.1 stub that lacks
:func:`buildModelFromUrdf`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

if not hasattr(pin, "buildModelFromUrdf"):
    pytest.skip(
        "pinocchio install is a stub (no buildModelFromUrdf); "
        "skipping live integration",
        allow_module_level=True,
    )

from src.shared.python.pose_interchange.canonical import canonical_zero_pose
from src.shared.python.pose_interchange.services.pinocchio import (
    PinocchioKinematicsService,
)

pytestmark = [
    pytest.mark.requires_pinocchio,
    pytest.mark.live_simulation,
    pytest.mark.integration,
]


_URDF = """<?xml version="1.0"?>
<robot name="tiny_test">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link_a">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint_a" type="revolute">
    <parent link="base_link"/>
    <child link="link_a"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
</robot>
"""


@pytest.fixture()
def urdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.urdf"
    path.write_text(_URDF, encoding="utf-8")
    return path


def test_pinocchio_service_loads_and_returns_transforms(urdf_path: Path) -> None:
    service = PinocchioKinematicsService()
    service.load(urdf_path)
    service.set_pose(canonical_zero_pose())
    transforms = service.get_link_transforms()

    assert len(transforms) >= 1, "expected at least one frame transform"
    for name, transform in transforms.items():
        assert isinstance(name, str)
        assert isinstance(transform, np.ndarray)
        assert transform.shape == (4, 4)
        assert transform.dtype == np.float64
        np.testing.assert_allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])


def test_pinocchio_service_step_advances_time(urdf_path: Path) -> None:
    service = PinocchioKinematicsService()
    service.load(urdf_path)
    service.set_pose(canonical_zero_pose())
    service.step(0.001)


def test_pinocchio_service_reset_after_load(urdf_path: Path) -> None:
    service = PinocchioKinematicsService()
    service.load(urdf_path)
    service.set_pose(canonical_zero_pose())
    service.reset()
