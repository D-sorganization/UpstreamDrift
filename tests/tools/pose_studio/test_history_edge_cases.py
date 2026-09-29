"""Edge-case tests for :class:`HistoryController`.

The happy path is covered in ``tests.unit.tools.pose_studio.test_core``.
This file pins the boundaries:

* Cursor stays consistent through interleaved push/undo/redo cycles.
* :attr:`current` always returns the snapshot at the cursor.
* The bottom snapshot is never trimmed past — pushing many times drops
  the oldest but keeps the cursor on the newest.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_zero_pose,
)
from src.tools.pose_studio.controllers import HistoryController

pytestmark = pytest.mark.unit


def _angle_pose(value: float) -> CanonicalPose:
    return CanonicalPose(
        pelvis_translation_m=np.zeros(3),
        pelvis_rotation_xyz_deg=np.zeros(3),
        joint_angles_deg={"HipStartPositionX": value},
    )


def test_undo_redo_idempotent_at_extremes() -> None:
    h = HistoryController(canonical_zero_pose())
    # At bottom: repeated undo returns None.
    assert h.undo() is None
    assert h.undo() is None
    # At top with no branch: redo returns None.
    assert h.redo() is None


def test_repeated_undo_then_redo_returns_to_top() -> None:
    h = HistoryController(canonical_zero_pose())
    poses = [_angle_pose(float(i)) for i in range(4)]
    for p in poses:
        h.push(p)
    assert h.current is poses[-1]
    # Undo all the way.
    while h.can_undo:
        h.undo()
    assert h.depth == 5
    # Redo all the way.
    while h.can_redo:
        h.redo()
    assert h.current is poses[-1]


def test_max_depth_keeps_cursor_at_top() -> None:
    h = HistoryController(canonical_zero_pose(), max_depth=2)
    p1 = _angle_pose(1.0)
    p2 = _angle_pose(2.0)
    p3 = _angle_pose(3.0)
    h.push(p1)
    h.push(p2)
    h.push(p3)
    # Stack capped at 2; cursor must still point at the newest pose.
    assert h.depth == 2
    assert h.current is p3
    # Undo returns the previous retained snapshot, not the discarded one.
    prev = h.undo()
    assert prev is p2


def test_push_clears_redo_branch_and_resets_redo_flag() -> None:
    h = HistoryController(canonical_zero_pose())
    p1 = _angle_pose(1.0)
    p2 = _angle_pose(2.0)
    h.push(p1)
    h.undo()
    assert h.can_redo
    h.push(p2)
    assert not h.can_redo
    assert h.current is p2


def test_max_depth_edge_two() -> None:
    """max_depth=2 is the minimum allowed and must work."""
    h = HistoryController(canonical_zero_pose(), max_depth=2)
    p1 = _angle_pose(1.0)
    h.push(p1)
    assert h.can_undo
    assert h.depth == 2
