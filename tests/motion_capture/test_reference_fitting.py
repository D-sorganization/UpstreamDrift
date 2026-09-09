"""Reference fitting contracts: explicit mapping, geometry and reproducibility."""

import numpy as np
import pytest

from src.motion_capture.reference.fitting import MarkerProfile, map_markers
from src.motion_capture.reference.importers import MotionDraft
from src.motion_capture.reference.model import ReferenceSource
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit


def draft(points: np.ndarray) -> MotionDraft:
    return MotionDraft(
        ReferenceSource(path="capture.c3d", sha256="a" * 64, format="c3d"),
        ("a", "b", "c"),
        tuple(np.arange(len(points)) / 100),
        points,
        "m",
        True,
    )


def test_mapping_converts_axes_and_requires_every_centroid_member() -> None:
    points = np.array([[[1, 2, 3], [3, 4, 5], [0, 0, 0]]] * 2, float)
    points[1, 1] = np.nan
    profile = MarkerProfile(
        name="test",
        axes=("+X", "-Z", "+Y"),
        joints={"mid_hip": ("a", "b"), "left_wrist": ("c",)},
    )
    mapped = map_markers(draft(points), profile)
    np.testing.assert_allclose(mapped[0, 0], [2, 3, 4])
    assert np.isnan(mapped[1, 0]).all()
    assert np.isnan(mapped[:, JOINT_NAMES.index("neck")]).all()
    np.testing.assert_array_equal(mapped[:, JOINT_NAMES.index("left_wrist")], 0)


@pytest.mark.parametrize("joints", [{"typo": ("a",)}, {"mid_hip": ()}])
def test_profile_rejects_invalid_mapping(joints: dict) -> None:
    with pytest.raises(ValueError):
        MarkerProfile(name="bad", axes=("+X", "+Y", "+Z"), joints=joints)


def test_mapping_rejects_absent_markers_and_unknown_units() -> None:
    profile = MarkerProfile(name="test", joints={"neck": ("absent",)})
    with pytest.raises(ValueError, match="absent"):
        map_markers(draft(np.ones((2, 3, 3))), profile)


def test_mapping_rejects_reflection() -> None:
    with pytest.raises(ValueError, match="right-handed"):
        MarkerProfile(name="bad", axes=("+X", "+Z", "+Y"), joints={"neck": ("a",)})
