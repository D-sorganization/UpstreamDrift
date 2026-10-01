"""Image fitting uses native geometry and measured, source-clock residuals."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.estimation import project_pinhole
from src.shared.python.motion_matching.pipeline import plant
from src.shared.python.motion_matching.historical_fit import (
    CameraProjection,
    ImageFitConfig,
    ImageFitInputs,
    fit_image_trajectory,
    read_capture_evidence,
    initialize_camera_hypothesis,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def test_camera_hypothesis_recovers_supplied_geometry_and_optics():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0.5], [-0.5, 0.3, -0.2]],
        dtype=float,
    )
    known = CameraProjection(
        np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
        np.eye(3),
        np.array([0.2, -0.3, 5]),
    )
    inferred = initialize_camera_hypothesis(
        points, known.project(points), known.intrinsics
    )
    np.testing.assert_allclose(
        inferred.project(points), known.project(points), atol=1e-5
    )
    np.testing.assert_allclose(inferred.translation, known.translation, atol=1e-5)


def test_camera_hypothesis_rejects_degenerate_geometry():
    with pytest.raises(ValueError, match="geometry"):
        initialize_camera_hypothesis(np.zeros((6, 3)), np.zeros((6, 2)), np.eye(3))


def native_problem():
    pytest.importorskip("mujoco")
    native = plant.get_plant(
        "mujoco",
        (
            ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
        ).read_bytes(),
    )
    attachments = {
        "right_shoulder": ("RScap", (0.0, 0.19494, 0.05)),
        "right_elbow": ("RS", (0.0, -0.04, -0.30435249856404367)),
        "right_wrist": ("RF", (0.03, 0.0, -0.13526160252728314)),
    }
    camera = CameraProjection(
        np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]]),
        np.eye(3),
        np.array([0.0, 0.0, 5.0]),
    )
    seed = np.zeros(len(native.coordinate_order))
    targets = []
    for elbow in (-0.3, -0.8):
        q = seed.copy()
        q[native.coordinate_order.index("REInput")] = elbow
        targets.append(
            project_pinhole(
                native.marker_positions(q, attachments),
                camera.intrinsics,
                rotation_world_to_camera=camera.rotation,
                translation_world_to_camera=camera.translation,
            )
        )
    inputs = ImageFitInputs(
        np.array([110.0, 111.0]),
        np.array(targets),
        np.ones((2, 3)),
        seed,
        np.ones_like(seed),
        ("REInput",),
    )
    return native, attachments, camera, inputs


def test_native_projection_fit_reduces_actual_pixel_residual():
    native, attachments, camera, inputs = native_problem()
    result = fit_image_trajectory(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(closure_weight=0.0, prior_weight=0.001, max_iterations=100),
    )
    assert result.rms_pixels < 0.01
    assert result.rms_pixels < result.initial_rms_pixels / 100
    np.testing.assert_allclose(
        result.q[:, native.coordinate_order.index("REInput")], [-0.3, -0.8], atol=0.002
    )
    np.testing.assert_array_equal(result.source_times, [110.0, 111.0])
    assert result.model_sha == native.plant_sha
    assert result.qualification == "monocular_research_hypothesis"
    assert result.physical_time_qualified is False
    dense = result.evaluate_source_times(np.array([110.0, 110.5, 111.0]))
    np.testing.assert_allclose(dense[[0, 2]], result.q)
    assert dense[1, native.coordinate_order.index("REInput")] < -0.3
    assert dense[1, native.coordinate_order.index("REInput")] > -0.8
    with pytest.raises(ValueError, match="interval"):
        result.evaluate_source_times(np.array([109.0]))
    with pytest.raises(ValueError):
        result.q[0, 0] = 12


def test_zero_confidence_does_not_turn_missing_landmarks_into_targets():
    native, attachments, camera, inputs = native_problem()
    weights = inputs.confidence.copy()
    weights[:, 0] = 0
    observed = inputs.observed_pixels.copy()
    observed[:, 0] = [9999.0, -9999.0]
    hidden = ImageFitInputs(
        inputs.source_times,
        observed,
        weights,
        inputs.seed,
        inputs.coordinate_scales,
        inputs.free_coordinates,
    )
    result = fit_image_trajectory(
        native,
        attachments,
        camera,
        hidden,
        ImageFitConfig(closure_weight=0.0, prior_weight=0.001, max_iterations=100),
    )
    assert result.rms_pixels < 0.01
    assert result.observed_point_count == 4
    assert result.pixel_errors[:, 0].tolist() == [None, None]


def test_empty_image_evidence_is_rejected_before_optimization():
    with pytest.raises(ValueError, match="observed"):
        ImageFitInputs(
            np.array([0.0, 1.0]),
            np.zeros((2, 1, 2)),
            np.zeros((2, 1)),
            np.zeros(1),
            np.ones(1),
            ("joint",),
        )


def test_capture_reader_preserves_missingness_and_source_clock():
    class Review:
        capture_id = "capture-v1"

        def frame(self, index):
            return {
                "image_width": 320,
                "image_height": 240,
                "frame": {
                    "pts_ticks": 300 + index,
                    "timebase_numerator": 1,
                    "timebase_denominator": 30,
                    "frame_id": f"f-{index}",
                    "frame_sha256": f"hash-{index}",
                },
                "observation": {
                    "status": "detected" if index == 0 else "missing",
                    "landmarks": {"wrist": {"x": 0.5, "y": 0.25, "visibility": None}}
                    if index == 0
                    else {},
                },
            }

    evidence = read_capture_evidence(
        Review(), ("wrist",), (0, 3), unknown_visibility_weight=0.25
    )
    np.testing.assert_array_equal(evidence.source_times, [10.0, 10.1])
    np.testing.assert_array_equal(evidence.observed_pixels[0], [[160.0, 60.0]])
    np.testing.assert_array_equal(evidence.confidence, [[0.25], [0.0]])
    assert evidence.frame_ids == ("f-0", "f-3")
    assert evidence.frame_hashes == ("hash-0", "hash-3")
    assert evidence.capture_id == "capture-v1"
    with pytest.raises(ValueError):
        evidence.observed_pixels[0, 0, 0] = 12


@pytest.mark.parametrize("rotation", [np.ones((3, 3)), np.diag([1.0, 1.0, -1.0])])
def test_invalid_camera_rotation_is_rejected(rotation):
    with pytest.raises(ValueError, match="proper orthonormal"):
        CameraProjection(np.eye(3), rotation, np.zeros(3))
