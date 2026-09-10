"""Fitted assets render through the actual capture comparison projection path."""

import numpy as np
import pytest

from src.motion_capture.reference.fit_pipeline import fit_reference
from src.motion_capture.reference import ComparisonLayer, ReferenceRegistration
from src.tools.capture_rig.reference_export import draw_reference_overlay
from tests.motion_capture.test_reference_fit_pipeline import pendulum_input
from tests.motion_capture.test_reference_registration import two_camera_rig

pytestmark = pytest.mark.unit


def test_fitted_reference_renders_on_both_camera_views() -> None:
    draft, profile = pendulum_input()
    result = fit_reference(draft, profile, "double_pendulum")
    registration = ReferenceRegistration(
        reference_id=result.asset.id, calibration_id="synthetic-fixture"
    )
    frame = np.full((720, 1280, 3), 70, dtype=np.uint8)
    for camera in two_camera_rig():
        rendered = draw_reference_overlay(
            frame.copy(),
            result.asset,
            draft.time_s[2],
            registration,
            camera,
            ComparisonLayer(),
        )
        assert np.any(rendered != frame)
        outside = draw_reference_overlay(
            frame.copy(), result.asset, 100.0, registration, camera, ComparisonLayer()
        )
        np.testing.assert_array_equal(outside, frame)
