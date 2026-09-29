from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.tools.pose_studio.widgets.view_3d import View3D
from src.shared.python.pose_interchange.canonical import (
    canonical_zero_pose,
)


def test_view_3d_initialization() -> None:
    view = View3D()
    assert view is not None
    assert view.highlighted_landmark() is None


def test_view_3d_update_pose_none() -> None:
    view = View3D()
    view.update_pose(None)
    # The canonical T-pose will be used and landmarks_order populated
    assert len(view._landmarks_order) > 0


def test_view_3d_update_pose_canonical() -> None:
    view = View3D()
    pose = canonical_zero_pose()
    view.update_pose(pose)
    assert len(view._landmarks_order) > 0


def test_view_3d_update_pose_invalid_type() -> None:
    view = View3D()
    with pytest.raises(TypeError):
        view.update_pose("invalid")  # type: ignore


def test_view_3d_update_from_service_transforms() -> None:
    view = View3D()

    pelvis_mat = np.eye(4)
    pelvis_mat[:3, 3] = [0.0, 0.0, 1.0]
    spine_mat = np.eye(4)
    spine_mat[:3, 3] = [0.0, 0.0, 1.5]

    transforms = {
        "pelvis": pelvis_mat,
        "spine_top": spine_mat,
    }

    view.update_from_service_transforms(transforms)
    assert view._landmarks_order == ["pelvis", "spine_top"]


def test_view_3d_on_pick() -> None:
    view = View3D()
    pose = canonical_zero_pose()
    view.update_pose(pose)

    # Mock event
    class PickEvent:
        def __init__(self, artist, ind):
            self.artist = artist
            self.ind = ind

    # Valid pick
    event1 = PickEvent(view._scatter, [0])
    view._on_pick(event1)

    assert view.highlighted_landmark() == view._landmarks_order[0]

    # Invalid pick - wrong artist
    view._highlighted_landmark = None
    event2 = PickEvent(MagicMock(), [0])
    view._on_pick(event2)
    assert view.highlighted_landmark() is None

    # Invalid pick - no ind
    event3 = PickEvent(view._scatter, None)
    view._on_pick(event3)
    assert view.highlighted_landmark() is None

    # Invalid pick - out of bounds ind
    event4 = PickEvent(view._scatter, [999999])
    view._on_pick(event4)
    assert view.highlighted_landmark() is None
