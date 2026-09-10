"""Readouts use displayed world positions and preserve missing motion samples."""

import numpy as np
import pytest

from src.motion_capture.coaching import (
    ReferenceGeometry,
    ReferencePlane,
    ReferencePoint,
)
from src.motion_capture.coaching.measurements import reference_distances
from src.motion_capture.reference.model import ReferenceMotion, ReferenceSource
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    ReferenceTransform,
)

pytestmark = pytest.mark.unit


def scene():
    motion = ReferenceMotion(
        title="Known Motion",
        source=ReferenceSource(
            path="fixture.json", sha256="0" * 64, format="marker-trajectory/1.0.0"
        ),
        source_units="m",
        source_axes=("+X", "+Y", "+Z"),
        source_names=("wrist",),
        joint_names=("wrist",),
        time_s=(0, 0.1, 0.2),
        points_m=(((1, 2, 3),), (None,), ((1, 2, 3),)),
    )
    registration = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="synthetic-world",
        transform=ReferenceTransform(scale=2, translation_m=(0, 1, 0)),
    )
    geometry = ReferenceGeometry(
        scene_id="scene",
        planes=(
            ReferencePlane(
                id="plane", origin_m=(0, 0, 1), along_m=(1, 0, 1), across_m=(0, 1, 1)
            ),
        ),
        points=(ReferencePoint(id="point", position_m=(0, 0, 0)),),
    )
    return motion, registration, geometry


def test_world_distances_and_missing_samples():
    motion, registration, geometry = scene()
    times = np.array([0, 0.05, 0.1, 0.2, 0.3])
    plane = reference_distances(
        motion, registration, geometry, times, "wrist", "plane", scene_id="scene"
    )
    np.testing.assert_allclose(plane, [-5, np.nan, np.nan, -5, np.nan], equal_nan=True)
    point = reference_distances(
        motion, registration, geometry, times[:1], "wrist", "point", scene_id="scene"
    )
    assert point[0] == pytest.approx(np.sqrt(69))


@pytest.mark.parametrize("change", ["scene", "joint", "reference", "time"])
def test_invalid_readout_contracts(change):
    motion, registration, geometry = scene()
    with pytest.raises(ValueError):
        reference_distances(
            motion,
            registration,
            geometry,
            np.array([np.nan if change == "time" else 0]),
            "unknown" if change == "joint" else "wrist",
            "unknown" if change == "reference" else "plane",
            scene_id="wrong" if change == "scene" else "scene",
        )


def test_handedness_changes_signed_plane_distance_without_changing_source():
    motion, registration, geometry = scene()
    mirrored = registration.model_copy(update={"mirror_lateral": True})
    result = reference_distances(
        motion, mirrored, geometry, np.array([0]), "wrist", "plane", scene_id="scene"
    )
    assert result[0] == 3
    assert motion.points_m[0][0] == (1, 2, 3)
