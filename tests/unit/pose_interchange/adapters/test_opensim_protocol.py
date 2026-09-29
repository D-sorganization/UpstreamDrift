"""Protocol conformance for :class:`OpenSimAdapter`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters.opensim import OpenSimAdapter
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

pytestmark = pytest.mark.unit


def test_opensim_is_pose_convention_adapter() -> None:
    adapter = OpenSimAdapter()
    assert isinstance(adapter, PoseConventionAdapter)
    assert adapter.engine_name == "opensim"
