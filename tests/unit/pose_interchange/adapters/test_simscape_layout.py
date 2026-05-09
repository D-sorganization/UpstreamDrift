"""Layout test for :class:`SimscapeAdapter` (identity layout vs canonical names)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.adapters.simscape import SimscapeAdapter
from src.shared.python.pose_interchange.protocol import JointSlot

pytestmark = pytest.mark.unit


def test_simscape_layout_jointslots() -> None:
    layout = SimscapeAdapter().joint_layout()
    assert isinstance(layout, Mapping)
    canonical_set = set(REFERENCE_GOLFER_FIELDS)
    assert set(layout.keys()) <= canonical_set
    for name, slot in layout.items():
        assert isinstance(slot, JointSlot)
        assert slot.canonical_name == name
        # Identity adapter: engine name == canonical name.
        assert slot.engine_name == name
        assert slot.units == "deg"


def test_simscape_layout_covers_full_canonical_set() -> None:
    """Simscape is the identity adapter — it should expose every canonical joint."""
    layout = SimscapeAdapter().joint_layout()
    assert set(layout.keys()) == set(REFERENCE_GOLFER_FIELDS)
