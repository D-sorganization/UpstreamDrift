"""Layout test for :class:`PinocchioAdapter`."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.adapters.pinocchio import PinocchioAdapter
from src.shared.python.pose_interchange.protocol import JointSlot

pytestmark = pytest.mark.unit


def test_pinocchio_layout_jointslots() -> None:
    layout = PinocchioAdapter().joint_layout()
    assert isinstance(layout, Mapping)
    canonical_set = set(REFERENCE_GOLFER_FIELDS)
    assert set(layout.keys()) <= canonical_set
    for name, slot in layout.items():
        assert isinstance(slot, JointSlot)
        assert slot.canonical_name == name
        assert slot.units == "rad"
