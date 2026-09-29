"""Protocol conformance and layout test for :class:`DrakeAdapter`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters.drake import DrakeAdapter
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

pytestmark = pytest.mark.unit


def test_drake_is_pose_convention_adapter() -> None:
    adapter = DrakeAdapter()
    assert isinstance(adapter, PoseConventionAdapter)
    assert adapter.engine_name == "drake"
