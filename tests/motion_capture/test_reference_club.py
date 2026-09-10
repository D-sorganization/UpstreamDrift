"""Explicit observed club centroids accompany any fitted skeleton without gap filling."""

from dataclasses import replace

import numpy as np
import pytest

from src.motion_capture.reference.fit_pipeline import fit_reference
from src.motion_capture.reference.fitting import MarkerProfile
from tests.motion_capture.test_reference_fit_pipeline import pendulum_input

pytestmark = pytest.mark.unit


def test_fit_retains_club_centroids_masks_and_provenance() -> None:
    draft, profile = pendulum_input()
    points = np.concatenate((draft.points, np.ones((5, 3, 3))), axis=1)
    points[:, 3] = (3, 2, 1)
    points[:, 4] = (5, 2, 1)
    points[2, 4] = np.nan
    draft = replace(
        draft, names=(*draft.names, "grip", "head_a", "head_b"), points=points
    )
    profile = MarkerProfile.model_validate(
        profile.model_dump()
        | {"club": {"grip": ("grip",), "head": ("head_a", "head_b")}}
    )
    result = fit_reference(draft, profile, "double_pendulum")
    assert result.asset.club_edges == ((2, 3),)
    assert result.asset.points_m[0][-1] == pytest.approx((4, -1, 2))
    assert result.asset.points_m[2][-1] is None
    assert result.asset.points_m[2][-2] is not None
    assert result.asset.joint_names[-2:] == ("observed_club_grip", "observed_club_head")
    assert result.manifest["profile"]["club"]["head"] == ["head_a", "head_b"]


def test_absent_club_does_not_fabricate_geometry() -> None:
    draft, profile = pendulum_input()
    profile = MarkerProfile.model_validate(
        profile.model_dump() | {"club": {"grip": ("missing",), "head": ("absent",)}}
    )
    result = fit_reference(draft, profile, "double_pendulum")
    assert not result.asset.club_edges
    assert "Club markers unavailable" in result.asset.notes
