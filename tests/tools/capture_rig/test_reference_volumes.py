"""Real world-space volumes, club visibility and alpha use the shared compositor."""

import numpy as np
import pytest

from src.motion_capture.reconstruct.cameras import PinholeCamera

from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.reference.registration import ReferenceRegistration
from src.tools.capture_rig.reference_export import draw_reference_overlay
from tests.motion_capture.test_reference_registration import (
    sample_motion,
    two_camera_rig,
)

pytestmark = pytest.mark.unit


def test_segment_mesh_is_centered_aligned_and_has_requested_radius() -> None:
    from src.tools.capture_rig.reference_volumes import segment_mesh

    vertices, faces = segment_mesh(
        np.array((1.0, 2.0, 3.0)), np.array((1.0, 2.0, 5.0)), 0.1
    )
    assert vertices[:, 2].min() == pytest.approx(3)
    assert vertices[:, 2].max() == pytest.approx(5)
    assert np.max(np.abs(vertices[:, 0] - 1)) == pytest.approx(0.2)
    assert faces.shape[1] == 3
    with pytest.raises(ValueError):
        segment_mesh(np.zeros(3), np.zeros(3), 0.1)


def test_ellipsoids_have_area_and_independent_transparency_in_two_views() -> None:
    asset = sample_motion()
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="test")
    frame = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    for camera in two_camera_rig():

        def render(alpha: float, camera: PinholeCamera = camera) -> np.ndarray:
            return draw_reference_overlay(
                frame,
                asset,
                0.5,
                reg,
                camera,
                ComparisonLayer(
                    draw_skeleton=False,
                    draw_joints=False,
                    draw_ellipsoids=True,
                    ellipsoid_opacity=alpha,
                ),
            )

        np.testing.assert_array_equal(render(0), frame)
        opaque, half = render(1), render(0.5)
        assert np.count_nonzero(opaque != frame) > 100
        np.testing.assert_allclose(
            half, np.rint((opaque.astype(float) + frame) / 2), atol=1
        )


def test_hidden_club_removes_its_joints_and_does_not_generate_body_volume() -> None:
    asset = sample_motion().changed(club_edges=((0, 1), (1, 2)))
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="test")
    camera = two_camera_rig()[0]
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    hidden = draw_reference_overlay(
        frame,
        asset,
        0,
        reg,
        camera,
        ComparisonLayer(draw_club=False, draw_ellipsoids=True),
    )
    np.testing.assert_array_equal(hidden, frame)
    shown = draw_reference_overlay(frame, asset, 0, reg, camera, ComparisonLayer())
    assert np.any(shown)


def test_club_can_be_shown_without_body_sticks() -> None:
    asset = sample_motion().changed(club_edges=((1, 2),))
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="test")
    camera = two_camera_rig()[0]
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    shown = draw_reference_overlay(
        frame,
        asset,
        0,
        reg,
        camera,
        ComparisonLayer(draw_skeleton=False, draw_joints=False, draw_club=True),
    )
    assert np.any(shown)
