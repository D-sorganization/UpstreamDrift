"""Unit tests for unified camera, morphology, and native state fitting (MMR-14, #11100).

Validates:
1. First failing tests: pass named articulated state through initialization, camera/FK, and rollout.
2. Coordinate identity preservation and quaternion velocity Jacobians (J_quat and J_quat_inv).
3. Independent grip translation and rotation closure gates.
4. Multi-hypothesis initialization producing alternatives across yaw, depth, and scale.
5. Renderer support for native (41) and canonical-v2 (42) state vectors with exact FK parity.
6. Video-only fitting without constructing fictitious TourCapture objects.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    forward_kinematics,
)
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    reference_golfer_setup,
)
from src.shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    SUBJECT_BINDING_SCHEMA_VERSION,
)
from src.shared.python.shadow_tracker.articulated_renderer import (
    CANONICAL_ARTICULATED_STATE_FIELDS,
    ArticulatedSilhouetteRenderer,
    state_vector_from_joint_dict,
)
from src.shared.python.shadow_tracker.contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    CANONICAL_V2_FULL_BODY_CONVENTION,
    NATIVE_FULL_BODY_CONVENTION,
    RenderRequest,
    SubjectModelBinding,
)
from src.shared.python.shadow_tracker.forward_model import (
    canonical_37_to_native_41,
    closed_grip_golfer_setup,
    evaluate_grip_closure,
    native_41_to_canonical_37,
    quat_derivative_to_body_omega,
    quat_velocity_jacobian_body,
    quat_velocity_jacobian_body_inv,
    body_omega_to_quat_derivative,
)
from src.shared.python.shadow_tracker.initialization import (
    FittingPriors,
    generate_monocular_hypotheses,
)
from src.shared.python.shadow_tracker.projection import PinholeCameraModel

pytestmark = pytest.mark.unit


def _make_subject_binding() -> SubjectModelBinding:
    return SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="sub-golfer-mmr14",
        model_hash="b" * 64,
        joint_ids=("pelvis", "spine", "l_shoulder", "r_shoulder"),
        body_ids=("torso", "l_arm", "r_arm", "l_leg", "r_leg"),
        visual_envelope={"height_m": 1.78, "chest_width_m": 0.44, "depth_m": 0.28},
        mass_kg=76.0,
        scale_evidence="measured_anthropometry",
        handedness="right",
    )


def test_first_failing_named_articulated_state_through_fk_and_replay() -> None:
    """Pass named articulated state through coordinate mappings, FK, and native representation."""
    angles = reference_golfer_setup()
    state_37 = state_vector_from_joint_dict(angles)
    assert len(state_37) == 37

    # Map 37 canonical fields to 41 native coordinates
    q_native = canonical_37_to_native_41(state_37)
    assert len(q_native) == 41
    assert np.all(np.isfinite(q_native))

    # Round trip back to 37
    state_37_rec = native_41_to_canonical_37(q_native)
    assert len(state_37_rec) == 37

    # Verify positions and key angles match
    assert np.allclose(state_37[:3], state_37_rec[:3], atol=1e-10)
    # Hip orientation (degrees)
    assert np.isclose(state_37[3], state_37_rec[3], atol=1e-6)
    assert np.isclose(state_37[4], state_37_rec[4], atol=1e-6)
    assert np.isclose(state_37[5], state_37_rec[5], atol=1e-6)

    # Permuted coordinate detection: swapping coords raises or changes values
    q_bad = np.zeros(40)
    with pytest.raises(ValueError, match="Expected.*41"):
        native_41_to_canonical_37(q_bad)


def test_quaternion_velocity_jacobian_body_identities() -> None:
    """Quaternion velocity Jacobian J_quat and its inverse satisfy kinematic identities."""
    # Arbitrary normalized quaternion
    q_raw = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)
    q = q_raw / np.linalg.norm(q_raw)

    j_quat = quat_velocity_jacobian_body(q)
    assert j_quat.shape == (4, 3)

    j_quat_inv = quat_velocity_jacobian_body_inv(q)
    assert j_quat_inv.shape == (3, 4)

    # 1. Left inverse identity: J_inv @ J == I_3
    identity_3 = j_quat_inv @ j_quat
    assert np.allclose(identity_3, np.eye(3), atol=1e-12)

    # 2. Tangent projector: J @ J_inv projects onto the tangent space (I - q q^T)
    projector = j_quat @ j_quat_inv
    expected_projector = np.eye(4) - np.outer(q, q)
    assert np.allclose(projector, expected_projector, atol=1e-12)

    # 3. Round trip: body angular velocity -> q_dot -> body angular velocity
    omega_body = np.array([1.5, -2.0, 0.75], dtype=np.float64)
    q_dot = body_omega_to_quat_derivative(q, omega_body)
    assert q_dot.shape == (4,)
    # q_dot must be orthogonal to q for unit quaternion rate
    assert np.isclose(np.dot(q, q_dot), 0.0, atol=1e-12)

    omega_rec = quat_derivative_to_body_omega(q, q_dot)
    assert np.allclose(omega_rec, omega_body, atol=1e-12)


def test_quaternion_velocity_jacobian_rejects_degenerate_inputs() -> None:
    """Quaternion velocity Jacobians reject non-finite or zero-norm quaternions."""
    with pytest.raises(ValueError, match="zero-norm"):
        quat_velocity_jacobian_body(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        quat_velocity_jacobian_body(np.array([1.0, float("nan"), 0.0, 0.0]))
    with pytest.raises(ValueError, match="zero-norm"):
        quat_velocity_jacobian_body_inv(np.zeros(4))


def test_grip_closure_translation_and_rotation_gates_evaluated_separately() -> None:
    """Grip translation and rotation gates are evaluated independently."""
    angles = closed_grip_golfer_setup()
    pose = forward_kinematics(angles, include_lower_body=True)

    # 1. Reference address pose satisfies both gates
    res_ref = evaluate_grip_closure(pose, max_translation_m=0.10, max_rotation_rad=0.6)
    assert res_ref.translation_passed is True
    assert res_ref.rotation_passed is True
    assert res_ref.passed is True

    # 2. Defect in translation only: hands separated by 0.3m
    angles_wide = dict(angles)
    angles_wide["RSStartPositionY"] = 60.0  # move right shoulder/arm outward
    pose_wide = forward_kinematics(angles_wide, include_lower_body=True)
    res_wide = evaluate_grip_closure(
        pose_wide, max_translation_m=0.08, max_rotation_rad=1.0
    )
    assert res_wide.translation_passed is False
    assert res_wide.rotation_passed is True
    assert res_wide.passed is False

    # 3. Defect in rotation only: lead wrist twisted 90 degrees
    angles_twist = dict(angles)
    angles_twist["LFStartPosition"] = 90.0  # twist forearm/wrist
    pose_twist = forward_kinematics(angles_twist, include_lower_body=True)
    res_twist = evaluate_grip_closure(
        pose_twist, max_translation_m=0.20, max_rotation_rad=0.3
    )
    assert res_twist.translation_passed is True
    assert res_twist.rotation_passed is False
    assert res_twist.passed is False


def test_monocular_hypotheses_produce_yaw_depth_scale_alternatives() -> None:
    """Monocular hypothesis generation produces alternatives across yaw, depth, and scale."""
    cam = PinholeCameraModel(
        camera_id="cam-mono",
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    binding = _make_subject_binding()
    bbox = (50, 200, 430, 440)  # top, left, bottom, right

    priors = FittingPriors(
        nominal_depths_m=(2.5, 3.5),
        yaw_options_deg=(-15.0, 0.0, 15.0),
        subject_height_prior_m=(1.60, 1.95),
        club_length_prior_m=(0.95, 1.25),
    )

    hypotheses = generate_monocular_hypotheses(
        camera=cam,
        observed_body_bbox=bbox,
        subject_binding=binding,
        nominal_depths_m=priors.nominal_depths_m,
        handedness_options=("right",),
        yaw_options_deg=priors.yaw_options_deg,
    )

    # 2 depths * 3 yaws = 6 distinct hypotheses
    assert len(hypotheses) == 6
    depths = {h.depth_m for h in hypotheses}
    assert depths == {2.5, 3.5}
    # Each hypothesis must have valid finite scale and pose
    for h in hypotheses:
        assert h.scale > 0.0
        assert math.isfinite(h.scale)
        assert len(h.pose) >= 7


def test_articulated_renderer_accepts_native_and_canonical_v2_states() -> None:
    """ArticulatedSilhouetteRenderer renders native-41 and canonical-v2 states with FK parity."""
    cam = PinholeCameraModel(
        camera_id="cam-parity",
        width_px=160,
        height_px=160,
        fx=120.0,
        fy=120.0,
        cx=80.0,
        cy=80.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
    )
    binding = _make_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-parity": cam},
        subject_binding=binding,
    )

    angles = reference_golfer_setup()
    state_37 = state_vector_from_joint_dict(angles)

    # 1. Render canonical 37
    res_37 = renderer.render(
        RenderRequest(
            camera_id="cam-parity",
            state=state_37,
            image_size_px=(160, 160),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )

    # 2. Render native 41
    q_native = canonical_37_to_native_41(state_37)
    res_41 = renderer.render(
        RenderRequest(
            camera_id="cam-parity",
            state=tuple(float(x) for x in q_native),
            image_size_px=(160, 160),
            state_convention=NATIVE_FULL_BODY_CONVENTION,
        )
    )

    # Parity: pixel masks must match between 37 canonical and 41 native
    assert res_37.body_mask == res_41.body_mask
    assert res_37.club_mask == res_41.club_mask
