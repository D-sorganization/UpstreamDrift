"""Mock-vs-real parity test for the Drake kinematics service.

Per Fleet Testing Standards §4. The real
``DrakeKinematicsService.load`` is currently scaffolded with
``NotImplementedError`` (tracked by issue #4963); once that lands,
the real-side test below will exercise it without code changes
here. Tracking: issue #5111 (Phase 3 of fleet testing alignment,
EPIC #1140).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.pose_interchange.canonical import (
    canonical_from_reference_setup,
)
from src.shared.python.pose_interchange.services._mock import (
    MockKinematicsService,
)


_TINY_URDF = """<?xml version="1.0"?>
<robot name="parity_tiny">
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


def _canonical_pose_scenario(svc) -> dict[str, np.ndarray]:
    svc.set_pose(canonical_from_reference_setup())
    return svc.get_link_transforms()


def _assert_valid_se3_dict(transforms: dict[str, np.ndarray]) -> None:
    assert len(transforms) >= 1
    for name, transform in transforms.items():
        assert isinstance(name, str)
        assert isinstance(transform, np.ndarray)
        assert transform.shape == (4, 4)
        assert transform.dtype == np.float64
        np.testing.assert_allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.unit
def test_mock_drake_kinematics_scenario() -> None:
    mock = MockKinematicsService(engine_name="drake")
    result = _canonical_pose_scenario(mock)
    _assert_valid_se3_dict(result)


@pytest.mark.live_simulation
@pytest.mark.requires_drake
def test_real_drake_kinematics_scenario_matches_mock(tmp_path: Path) -> None:
    pytest.importorskip("pydrake")
    from src.shared.python.pose_interchange.services.drake import (
        DrakeKinematicsService,
    )

    mock = MockKinematicsService(engine_name="drake")
    mock_result = _canonical_pose_scenario(mock)

    urdf = tmp_path / "parity_tiny.urdf"
    urdf.write_text(_TINY_URDF, encoding="utf-8")
    real = DrakeKinematicsService()
    try:
        real.load(urdf)
    except NotImplementedError:
        pytest.skip(
            "DrakeKinematicsService is scaffolded; full bridge tracked by #4963"
        )
    real_result = _canonical_pose_scenario(real)

    _assert_valid_se3_dict(mock_result)
    _assert_valid_se3_dict(real_result)
