"""Protocol conformance for :class:`SimscapeAdapter`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.adapters.simscape import SimscapeAdapter
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

pytestmark = pytest.mark.unit


def test_simscape_is_pose_convention_adapter() -> None:
    adapter = SimscapeAdapter()
    assert isinstance(adapter, PoseConventionAdapter)
    assert adapter.engine_name == "simscape"
