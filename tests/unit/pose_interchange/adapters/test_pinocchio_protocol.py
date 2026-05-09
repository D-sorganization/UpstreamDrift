"""Protocol conformance for :class:`PinocchioAdapter`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters.pinocchio import PinocchioAdapter
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

pytestmark = pytest.mark.unit


def test_pinocchio_is_pose_convention_adapter() -> None:
    adapter = PinocchioAdapter()
    assert isinstance(adapter, PoseConventionAdapter)
    assert adapter.engine_name == "pinocchio"
