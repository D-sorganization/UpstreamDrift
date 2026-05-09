"""Protocol conformance for :class:`MujocoAdapter`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters.mujoco import MujocoAdapter
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

pytestmark = pytest.mark.unit


def test_mujoco_is_pose_convention_adapter() -> None:
    adapter = MujocoAdapter()
    assert isinstance(adapter, PoseConventionAdapter)
    assert adapter.engine_name == "mujoco"
