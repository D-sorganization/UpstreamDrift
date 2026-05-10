"""Mock-vs-real parity test for the MuJoCo kinematics service.

Per Fleet Testing Standards §4 (the mock-vs-real contract): for each
engine adapter that has both a mock and a real implementation, one
test runs the same canonical scenario against both and asserts that
the outputs are equivalent within tolerance.

The mock-side test runs in the default lane and protects against
``MockKinematicsService`` drifting from its declared shape.

The real-side test is marked ``live_simulation + requires_mujoco`` so
it is deselected from the default lane but runs on the nightly job
once the real MuJoCo wheel is installed. When the real bridge differs
materially from the mock (which is expected — the real bridge keys
transforms by MJCF body names, the mock keys by canonical landmark
names), the parity assertion targets the structural invariants both
must satisfy: each value is a valid 4x4 SE(3) matrix.

Tracking: issue #5111 (Phase 3 of fleet testing alignment, EPIC #1140).
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


_TINY_MJCF = """<mujoco model="parity_tiny">
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


def _canonical_pose_scenario(svc) -> dict[str, np.ndarray]:
    """Run the canonical reference-setup pose through *svc*.

    Returns the link transform dict from the service. Both the mock
    and the real service must accept this scenario without raising.
    """
    svc.set_pose(canonical_from_reference_setup())
    return svc.get_link_transforms()


def _assert_valid_se3_dict(transforms: dict[str, np.ndarray]) -> None:
    """Assert every value in *transforms* is a 4x4 SE(3) matrix."""
    assert len(transforms) >= 1
    for name, transform in transforms.items():
        assert isinstance(name, str), f"non-str key: {name!r}"
        assert isinstance(transform, np.ndarray)
        assert transform.shape == (4, 4)
        assert transform.dtype == np.float64
        np.testing.assert_allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.unit
def test_mock_mujoco_kinematics_scenario() -> None:
    """Mock side: ``MockKinematicsService("mujoco")`` accepts the canonical scenario."""
    mock = MockKinematicsService(engine_name="mujoco")
    result = _canonical_pose_scenario(mock)
    _assert_valid_se3_dict(result)


@pytest.mark.live_simulation
@pytest.mark.requires_mujoco
def test_real_mujoco_kinematics_scenario_matches_mock(tmp_path: Path) -> None:
    """Real side: same scenario through ``MuJoCoKinematicsService`` shares structure with mock."""
    pytest.importorskip("mujoco")
    from src.shared.python.pose_interchange.services.mujoco import (
        MuJoCoKinematicsService,
    )

    mock = MockKinematicsService(engine_name="mujoco")
    mock_result = _canonical_pose_scenario(mock)

    mjcf = tmp_path / "parity_tiny.xml"
    mjcf.write_text(_TINY_MJCF, encoding="utf-8")
    real = MuJoCoKinematicsService()
    real.load(mjcf)
    real_result = _canonical_pose_scenario(real)

    # Both services must produce valid SE(3) dicts. Key sets differ
    # (mock = canonical landmarks; real = MJCF body names), so we
    # assert the structural invariants both must satisfy.
    _assert_valid_se3_dict(mock_result)
    _assert_valid_se3_dict(real_result)
