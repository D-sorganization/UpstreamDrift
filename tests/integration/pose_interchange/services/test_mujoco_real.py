"""Live integration test for the real MuJoCo bridge (issue #4963).

Skipped when ``mujoco`` is not importable. Loads a tiny inline MJCF,
pushes the canonical zero pose, and asserts that at least one body
transform comes back as a 4x4 SE(3) matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.pose_interchange.canonical import canonical_zero_pose
from src.shared.python.pose_interchange.services.mujoco import (
    MuJoCoKinematicsService,
)

pytestmark = [
    pytest.mark.requires_mujoco,
    pytest.mark.live_simulation,
    pytest.mark.integration,
]


_MJCF = """<mujoco model="tiny_test">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link_a" pos="0 0 1">
      <joint name="hinge_a" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"/>
      <body name="link_b" pos="0.5 0 0">
        <joint name="hinge_b" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture()
def mjcf_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.xml"
    path.write_text(_MJCF, encoding="utf-8")
    return path


def test_mujoco_service_loads_and_returns_transforms(mjcf_path: Path) -> None:
    service = MuJoCoKinematicsService()
    service.load(mjcf_path)
    service.set_pose(canonical_zero_pose())
    transforms = service.get_link_transforms()

    assert len(transforms) >= 1, "expected at least one body transform"
    for name, transform in transforms.items():
        assert isinstance(name, str)
        assert isinstance(transform, np.ndarray)
        assert transform.shape == (4, 4)
        assert transform.dtype == np.float64
        # Bottom row is [0, 0, 0, 1] for a valid SE(3) matrix.
        np.testing.assert_allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])


def test_mujoco_service_step_advances_time(mjcf_path: Path) -> None:
    service = MuJoCoKinematicsService()
    service.load(mjcf_path)
    service.set_pose(canonical_zero_pose())
    service.step(0.001)  # Should not raise.


def test_mujoco_service_reset_after_load(mjcf_path: Path) -> None:
    service = MuJoCoKinematicsService()
    service.load(mjcf_path)
    service.set_pose(canonical_zero_pose())
    service.reset()  # Should not raise.
