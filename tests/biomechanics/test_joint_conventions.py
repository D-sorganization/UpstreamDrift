"""Physical invariants for calibrated joint convention conversion."""

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.joint_conventions import (
    AnatomicalFrame,
    RotationConvention,
    convert_orientations,
    joint_kinematics,
    matrix_to_orientations,
    orientations_to_matrix,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "sequence",
    ["".join(p) for p in itertools.permutations("xyz")]
    + [a + b + a for a in "xyz" for b in "xyz" if a != b],
)
@pytest.mark.parametrize("intrinsic", [True, False])
def test_all_sequences_reconstruct_physical_orientation(sequence, intrinsic):
    sequence = sequence.upper() if intrinsic else sequence
    convention = RotationConvention(sequence, degrees=True)
    original = Rotation.from_rotvec([0.3, -0.5, 0.7]).as_matrix()
    result = matrix_to_orientations(original, "euler", convention)
    np.testing.assert_allclose(
        orientations_to_matrix(result.values, "euler", convention), original, atol=1e-12
    )


def test_quaternion_sign_and_order_are_explicit():
    q = Rotation.from_rotvec([0.2, 0.4, 0.8]).as_quat()
    a = orientations_to_matrix(q, "quaternion_xyzw")
    np.testing.assert_allclose(a, orientations_to_matrix(-q, "quaternion_xyzw"))
    np.testing.assert_allclose(
        a, orientations_to_matrix(np.roll(q, 1), "quaternion_wxyz")
    )
    result = convert_orientations(q, "quaternion_xyzw", "rotation_vector")
    np.testing.assert_allclose(result.values, [0.2, 0.4, 0.8])


def test_singularity_is_reported_without_losing_orientation():
    convention = RotationConvention("XYZ", degrees=True)
    r = orientations_to_matrix([[10, 90, 20], [10, 30, 20]], "euler", convention)
    result = matrix_to_orientations(r, "euler", convention)
    np.testing.assert_array_equal(result.singular, [True, False])
    np.testing.assert_allclose(
        orientations_to_matrix(result.values, "euler", convention), r, atol=1e-12
    )


def test_joint_pose_is_invariant_under_world_frame_change():
    p = AnatomicalFrame(np.eye(3), np.array([1.0, 2.0, 3.0]), "pelvis-calibration")
    d = AnatomicalFrame(
        Rotation.from_rotvec([0.3, 0.2, 0.1]).as_matrix(),
        np.array([2.0, 2.0, 3.0]),
        "thorax-calibration",
    )
    baseline = joint_kinematics(p, d)
    world = Rotation.from_rotvec([0.7, -0.2, 0.4]).as_matrix()
    offset = np.array([8.0, 3.0, -2.0])
    transformed = joint_kinematics(
        AnatomicalFrame(
            world @ p.rotation_world, world @ p.origin_world + offset, p.calibration_id
        ),
        AnatomicalFrame(
            world @ d.rotation_world, world @ d.origin_world + offset, d.calibration_id
        ),
    )
    np.testing.assert_allclose(
        transformed.position_proximal, baseline.position_proximal, atol=1e-12
    )
    np.testing.assert_allclose(
        transformed.rotation_relative, baseline.rotation_relative, atol=1e-12
    )
    assert baseline.provenance == ("pelvis-calibration", "thorax-calibration")


@pytest.mark.parametrize(
    "bad", [np.diag([1.0, 1.0, -1.0]), np.eye(3) * 2, np.full((3, 3), np.nan)]
)
def test_invalid_rotations_are_rejected_without_silent_projection(bad):
    with pytest.raises(ValueError):
        orientations_to_matrix(bad, "matrix")


def test_contracts_reject_ambiguous_conventions_and_missing_calibration():
    with pytest.raises(ValueError):
        RotationConvention("XyZ")
    with pytest.raises(ValueError):
        orientations_to_matrix([0, 0, 0, 0], "quaternion_wxyz")
    with pytest.raises(ValueError):
        AnatomicalFrame(np.eye(3), np.zeros(3), "")


def test_calibration_converts_engine_body_pose_to_anatomical_pose():
    engine_r = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    calibration_r = Rotation.from_rotvec([0.4, 0, 0]).as_matrix()
    frame = AnatomicalFrame.from_body_pose(
        engine_r,
        np.ones(3),
        calibration_r,
        np.array([0.0, 0.2, 0.0]),
        "digitized-ac-joints",
    )
    np.testing.assert_allclose(frame.rotation_world, engine_r @ calibration_r)
    np.testing.assert_allclose(
        frame.origin_world, np.ones(3) + engine_r @ [0.0, 0.2, 0.0]
    )


def test_named_profiles_preserve_isb_negative_elevation_branch():
    from src.shared.python.biomechanics.joint_conventions import (
        ANATOMICAL_PROFILES,
        anatomical_joint_angles,
    )

    r = Rotation.from_euler("YXY", [20, -60, 30], degrees=True).as_matrix()
    result = anatomical_joint_angles(r, "isb_glenohumeral", degrees=True)
    np.testing.assert_allclose(result.values, [20, -60, 30], atol=1e-12)
    assert ANATOMICAL_PROFILES["isb_elbow"].sequence == "ZXY"
    assert ANATOMICAL_PROFILES["isb_hip"].source_url.startswith("https://")


def test_grood_suntay_floating_axis_and_singular_diagnostic():
    from src.shared.python.biomechanics.joint_conventions import grood_suntay

    p = AnatomicalFrame(np.eye(3), np.zeros(3), "femur")
    d = AnatomicalFrame(
        Rotation.from_euler("XYZ", [30, 20, 10], degrees=True).as_matrix(),
        np.zeros(3),
        "tibia",
    )
    result = grood_suntay(p, d, proximal_axis="X", distal_axis="Z", degrees=True)
    np.testing.assert_allclose(result.angles, [30, 20, 10], atol=1e-12)
    assert abs(np.dot(result.floating_axis_world, p.rotation_world[:, 0])) < 1e-12
    assert abs(np.dot(result.floating_axis_world, d.rotation_world[:, 2])) < 1e-12
    with pytest.raises(ValueError):
        grood_suntay(p, d, proximal_axis="Y", distal_axis="Y")


def test_orientation_series_preserves_missing_frames_and_reports_validity():
    from src.shared.python.biomechanics.joint_conventions import (
        convert_orientation_series,
    )

    result = convert_orientation_series(
        [[0, 0, 0], [np.nan] * 3, [0.1, 0.2, 0.3]], "euler", "quaternion_wxyz"
    )
    np.testing.assert_array_equal(result.valid, [True, False, True])
    assert np.isnan(result.values[1]).all()
    with pytest.raises(ValueError):
        convert_orientation_series([[0, np.nan, 0]], "euler", "matrix")
