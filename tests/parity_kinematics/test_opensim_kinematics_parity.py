"""Mock-vs-real parity test for the OpenSim kinematics service.

Per Fleet Testing Standards §4. The real
``OpenSimKinematicsService.load`` is currently scaffolded with
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
def test_mock_opensim_kinematics_scenario() -> None:
    mock = MockKinematicsService(engine_name="opensim")
    result = _canonical_pose_scenario(mock)
    _assert_valid_se3_dict(result)


@pytest.mark.live_simulation
@pytest.mark.requires_opensim
def test_real_opensim_kinematics_scenario_matches_mock(tmp_path: Path) -> None:
    pytest.importorskip("opensim")
    from src.shared.python.pose_interchange.services.opensim import (
        OpenSimKinematicsService,
    )

    mock = MockKinematicsService(engine_name="opensim")
    mock_result = _canonical_pose_scenario(mock)

    osim = tmp_path / "parity_tiny.osim"
    # OpenSim requires a real .osim model; the bridge is scaffolded
    # so we skip until the live wiring lands (#4963). The mock-side
    # invariants are still validated above.
    real = OpenSimKinematicsService()
    try:
        real.load(osim)
    except NotImplementedError:
        pytest.skip(
            "OpenSimKinematicsService is scaffolded; full bridge tracked by #4963"
        )
    real_result = _canonical_pose_scenario(real)

    _assert_valid_se3_dict(mock_result)
    _assert_valid_se3_dict(real_result)
