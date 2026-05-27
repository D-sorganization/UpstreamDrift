"""Protocol conformance for :class:`MyoSuiteAdapter`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters.myosuite import MyoSuiteAdapter
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

pytestmark = pytest.mark.unit


def test_myosuite_is_pose_convention_adapter() -> None:
    adapter = MyoSuiteAdapter()
    assert isinstance(adapter, PoseConventionAdapter)
    assert adapter.engine_name == "myosuite"
