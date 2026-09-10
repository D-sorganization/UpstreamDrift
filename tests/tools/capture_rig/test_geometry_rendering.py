"""Reference planes use calibrated projection and preserve alpha and timing."""

import numpy as np
import pytest

from src.motion_capture.coaching.geometry import ReferenceGeometry, ReferencePlane
from src.tools.capture_rig.geometry_rendering import render_geometry
from tests.motion_capture.test_reference_registration import two_camera_rig

pytestmark = pytest.mark.unit


def test_plane_alpha_clock_and_no_input_mutation() -> None:
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    plane = ReferencePlane(
        origin_m=(0, 1, 0),
        along_m=(1, 1, 0),
        across_m=(0, 2, 0),
        opacity=1,
        first_s=1,
        last_s=2,
    )
    camera = two_camera_rig()[0]

    def draw(opacity: float, time: float = 1) -> np.ndarray:
        geometry = ReferenceGeometry(
            scene_id="scene", planes=(plane.changed(opacity=opacity),)
        )
        return render_geometry(frame, geometry, camera, time)

    np.testing.assert_array_equal(draw(1, 0), frame)
    np.testing.assert_array_equal(draw(0), frame)
    opaque, half = draw(1), draw(0.5)
    assert np.count_nonzero(opaque != frame) > 1000
    np.testing.assert_allclose(
        half, np.rint((opaque.astype(float) + frame) / 2), atol=1
    )
    assert np.all(frame == 40)


def test_geometry_requires_camera_and_valid_image() -> None:
    plane = ReferencePlane(origin_m=(0, 0, 0), along_m=(1, 0, 0), across_m=(0, 1, 0))
    geometry = ReferenceGeometry(scene_id="scene", planes=(plane,))
    with pytest.raises(ValueError, match="camera"):
        render_geometry(np.zeros((5, 5, 3), dtype=np.uint8), geometry, None, 0)
    with pytest.raises(ValueError, match="image"):
        render_geometry(np.zeros((5, 5)), geometry, two_camera_rig()[0], 0)


def test_plane_behind_camera_is_not_drawn() -> None:
    plane = ReferencePlane(origin_m=(0, 1, 10), along_m=(1, 1, 10), across_m=(0, 2, 10))
    geometry = ReferenceGeometry(scene_id="scene", planes=(plane,))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        render_geometry(frame, geometry, two_camera_rig()[0], 0), frame
    )


def test_shared_compositor_keeps_geometry_when_model_hidden() -> None:
    from src.motion_capture.reference.comparison import ComparisonLayer
    from src.motion_capture.reference.registration import ReferenceRegistration
    from src.tools.capture_rig.reference_rendering import (
        ComparisonRenderContext,
        ComparisonRenderer,
    )
    from tests.motion_capture.test_reference_registration import sample_motion

    asset = sample_motion()
    plane = ReferencePlane(origin_m=(0, 1, 0), along_m=(1, 1, 0), across_m=(0, 2, 0))
    geometry = ReferenceGeometry(scene_id="scene", planes=(plane,))
    registration = ReferenceRegistration(reference_id=asset.id, calibration_id="test")
    context = ComparisonRenderContext(
        "a",
        asset,
        registration,
        ComparisonLayer(visible=False),
        geometry=geometry,
        scene_id="scene",
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    camera = two_camera_rig()[0]
    with ComparisonRenderer(asset) as renderer:
        rendered = renderer.overlay(frame, 0, context, camera)
    np.testing.assert_array_equal(rendered, render_geometry(frame, geometry, camera, 0))
    assert np.any(rendered)
    with pytest.raises(ValueError, match="scene"):
        ComparisonRenderContext(
            "a",
            asset,
            registration,
            ComparisonLayer(),
            geometry=geometry,
            scene_id="wrong",
        )
