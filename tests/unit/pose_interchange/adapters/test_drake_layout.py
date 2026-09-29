"""Layout test for :class:`DrakeAdapter`."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.adapters.drake import DrakeAdapter
from src.shared.python.pose_interchange.protocol import JointSlot

pytestmark = pytest.mark.unit


def test_drake_layout_returns_mapping_of_jointslots() -> None:
    layout = DrakeAdapter().joint_layout()
    assert isinstance(layout, Mapping)
    canonical_set = set(REFERENCE_GOLFER_FIELDS)
    assert set(layout.keys()) <= canonical_set
    for name, slot in layout.items():
        assert isinstance(slot, JointSlot)
        assert slot.canonical_name == name
        assert slot.length >= 1
        assert slot.units in {"rad", "deg"}
        assert slot.sign in {1, -1}


def test_drake_layout_units_rad() -> None:
    layout = DrakeAdapter().joint_layout()
    for slot in layout.values():
        assert slot.units == "rad"
