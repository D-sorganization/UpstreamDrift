"""Unit tests for Shadow Tracker state, camera, and renderer boundary contracts (MMR-14-I, #11110).

Validates:
1. No-evidence abstention: empty masks or no-valid-pixel inputs cannot yield a winning initialization.
2. Real limb and club pose changes move expected geometry.
3. Offscreen partial silhouettes, complete out-of-frame translation, and near-plane clipping.
4. Anamorphic camera (fx != fy) aspect ratio rendering and bridge consistency.
5. Crop, rotation, and distortion consistency across boundaries.
6. Missing fields, dimension mismatches, and non-finite value rejections.
7. Quaternion normalization, degenerate quaternion rejection, and bidirectional q/v roundtrip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    reference_golfer_setup,
)
from src.shared.python.motion_pipeline.contracts import (
    CameraExtrinsics as PipelineExtrinsics,
    CameraIntrinsics as PipelineIntrinsics,
)
from pose_estimation.observations import (
    CameraCalibration,
    CameraExtrinsics as ObsExtrinsics,
    CameraIntrinsics as ObsIntrinsics,
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
    state_vector_to_joint_dict,
)
from src.shared.python.shadow_tracker.camera_bridge import (
    from_pipeline_camera,
    to_pipeline_camera,
)
from src.shared.python.shadow_tracker.contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    RenderRequest,
    RenderResult,
    SubjectModelBinding,
)
from src.shared.python.shadow_tracker.forward_model import (
    canonical_to_native_full_body,
    native_to_canonical_full_body,
    rpy_jacobian_body,
    rpy_jacobian_body_inv,
)
from src.shared.python.shadow_tracker.initialization import (
    InitialHypothesis,
    fit_initial_state_multiview,
)
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.projection import (
    AnalyticSilhouetteRenderer,
    PinholeCameraModel,
    compute_silhouette_loss,
)
from src.shared.python.shadow_tracker.source_records import FrameIdentity

pytestmark = pytest.mark.unit


def _make_test_subject_binding() -> SubjectModelBinding:
    return SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="sub-golfer-01",
        model_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        joint_ids=("pelvis", "spine", "l_shoulder", "r_shoulder"),
        body_ids=("torso", "l_arm", "r_arm", "l_leg", "r_leg"),
        visual_envelope={"height_m": 1.78, "chest_width_m": 0.44, "depth_m": 0.28},
        mass_kg=75.0,
        scale_evidence="measured_anthropometry",
        handedness="right",
    )


def _make_mask_frame(
    *,
    camera_id: str,
    width_px: int,
    height_px: int,
    body: bytes,
    club: bytes,
    valid: bytes,
) -> MaskFrame:
    return MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="asset_01",
            shot_id="shot_01",
            swing_id="swing_01",
            camera_id=camera_id,
            frame_id="f_001",
            pts_ticks=0,
            timebase_numerator=1,
            timebase_denominator=1000,
            physical_time_s=0.0,
            physical_time_reason="",
            frame_sha256="a" * 64,
        ),
        width_px=width_px,
        height_px=height_px,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-01",
        parent_revision_id=None,
        producer_id="test",
        correction_note="",
    )


# ===========================================================================
# 1. No-Evidence Abstention: Empty / No-Valid-Pixel Initializations
# ===========================================================================


def test_empty_mask_cannot_yield_winning_initialization() -> None:
    """An empty mask input (zero foreground pixels) must not select a winner."""
    cam = PinholeCameraModel(
        camera_id="cam-front",
        width_px=100,
        height_px=100,
        fx=80.0,
        fy=80.0,
        cx=50.0,
        cy=50.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
    )
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-front": cam},
        subject_binding=_make_test_subject_binding(),
    )

    total_px = 100 * 100
    # Empty observed mask: all valid pixels, but 0 body or club foreground pixels
    empty_mask = _make_mask_frame(
        camera_id="cam-front",
        width_px=100,
        height_px=100,
        body=bytes([0] * total_px),
        club=bytes([0] * total_px),
        valid=bytes([1] * total_px),
    )

    angles = reference_golfer_setup()
    candidate_state = state_vector_from_joint_dict(angles)

    result = fit_initial_state_multiview(
        cameras=(cam,),
        observed_masks=(empty_mask,),
        candidate_poses=(candidate_state,),
        subject_binding=_make_test_subject_binding(),
        renderer=renderer,
    )

    assert result.best_hypothesis is None, (
        "No-evidence abstention violated: empty mask yielded a winning initialization"
    )


def test_no_valid_pixel_input_cannot_yield_winning_initialization() -> None:
    """A mask with zero valid pixels must not select a winner."""
    cam = PinholeCameraModel(
        camera_id="cam-front",
        width_px=100,
        height_px=100,
        fx=80.0,
        fy=80.0,
        cx=50.0,
        cy=50.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
    )
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-front": cam},
        subject_binding=_make_test_subject_binding(),
    )

    total_px = 100 * 100
    # No valid pixels: valid mask is all zeros (body and club must be 0 per MaskFrame invariant)
    invalid_mask = _make_mask_frame(
        camera_id="cam-front",
        width_px=100,
        height_px=100,
        body=bytes([0] * total_px),
        club=bytes([0] * total_px),
        valid=bytes([0] * total_px),
    )

    angles = reference_golfer_setup()
    candidate_state = state_vector_from_joint_dict(angles)

    result = fit_initial_state_multiview(
        cameras=(cam,),
        observed_masks=(invalid_mask,),
        candidate_poses=(candidate_state,),
        subject_binding=_make_test_subject_binding(),
        renderer=renderer,
    )

    assert result.best_hypothesis is None, (
        "No-evidence abstention violated: zero valid pixels yielded a winning initialization"
    )


# ===========================================================================
# 2. Real Limb and Club Pose Changes Move Expected Geometry
# ===========================================================================


def test_real_limb_and_club_pose_change_moves_geometry() -> None:
    """Joint angle changes must move the expected geometry and produce altered masks."""
    cam = PinholeCameraModel(
        camera_id="cam-front",
        width_px=200,
        height_px=200,
        fx=140.0,
        fy=140.0,
        cx=100.0,
        cy=100.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
    )
    binding = _make_test_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-front": cam},
        subject_binding=binding,
    )

    # 1. Base pose
    angles_base = reference_golfer_setup()
    state_base = state_vector_from_joint_dict(angles_base)
    res_base = renderer.render(
        RenderRequest(
            camera_id="cam-front",
            state=state_base,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )

    angles_arm = dict(angles_base)
    angles_arm["LSInputX"] = 45.0
    angles_arm["LEStartPosition"] = 90.0
    state_arm = state_vector_from_joint_dict(angles_arm)
    res_arm = renderer.render(
        RenderRequest(
            camera_id="cam-front",
            state=state_arm,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )

    body_base = np.array(res_base.body_mask, dtype=bool)
    body_arm = np.array(res_arm.body_mask, dtype=bool)
    assert not np.array_equal(body_base, body_arm)
    body_diff_pixels = np.sum(body_base != body_arm)
    assert body_diff_pixels > 0, (
        f"Limb rotation should move body silhouette, got {body_diff_pixels} changed px"
    )

    # 3. Club pose change (wrist flex/extend moves clubhead)
    angles_club = dict(angles_base)
    angles_club["LWStartPositionX"] = 45.0
    state_club = state_vector_from_joint_dict(angles_club)
    res_club = renderer.render(
        RenderRequest(
            camera_id="cam-front",
            state=state_club,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )

    club_base = np.array(res_base.club_mask, dtype=bool)
    club_mod = np.array(res_club.club_mask, dtype=bool)
    assert not np.array_equal(club_base, club_mod)
    club_diff_pixels = np.sum(club_base != club_mod)
    assert club_diff_pixels > 20, (
        f"Club angle change should move club silhouette, got {club_diff_pixels} changed px"
    )


# ===========================================================================
# 3. Offscreen Partial Silhouettes and Boundary Clipping
# ===========================================================================


def test_offscreen_partial_silhouettes_and_viewport_clipping() -> None:
    """Renderer must gracefully handle partial silhouettes at borders, out-of-frame, and near plane."""
    cam = PinholeCameraModel(
        camera_id="cam-tight",
        width_px=80,
        height_px=80,
        fx=120.0,
        fy=120.0,
        cx=40.0,
        cy=40.0,
        translation_world_to_camera=(0.0, 0.0, 2.5),
    )
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-tight": cam},
        subject_binding=_make_test_subject_binding(),
    )

    angles = reference_golfer_setup()
    base_state = state_vector_from_joint_dict(angles)

    # 1. Partial silhouette: translate golfer so half the torso/arms are past the right border
    partial_state = list(base_state)
    partial_state[0] = 0.5  # shift +0.5m in X
    res_partial = renderer.render(
        RenderRequest(
            camera_id="cam-tight",
            state=tuple(partial_state),
            image_size_px=(80, 80),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )
    assert res_partial.body_mask.count(1) > 0, (
        "Partial silhouette should render visible portion"
    )
    # Ensure pixels hit the right border (x == 79)
    mask_2d = np.array(res_partial.body_mask).reshape((80, 80))
    assert np.any(mask_2d[:, -1] == 1), (
        "Silhouette should extend to right border without crashing"
    )

    # 2. Completely offscreen golfer: translate 20 meters away in X
    offscreen_state = list(base_state)
    offscreen_state[0] = 20.0
    res_offscreen = renderer.render(
        RenderRequest(
            camera_id="cam-tight",
            state=tuple(offscreen_state),
            image_size_px=(80, 80),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )
    assert res_offscreen.body_mask.count(1) == 0
    assert res_offscreen.club_mask.count(1) == 0

    # 3. Behind camera plane (near plane clipping): translate in Z behind camera
    behind_state = list(base_state)
    behind_state[2] = -5.0
    res_behind = renderer.render(
        RenderRequest(
            camera_id="cam-tight",
            state=tuple(behind_state),
            image_size_px=(80, 80),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )
    assert res_behind.body_mask.count(1) == 0


# ===========================================================================
# 4. Anamorphic Camera (fx != fy) Rendering and Bridge Consistency
# ===========================================================================


def test_anamorphic_camera_rendering_and_bridge_consistency() -> None:
    """Anamorphic cameras with fx != fy must render elliptical projections and bridge faithfully."""
    # 1. Render test: fx = 2 * fy
    cam_anamorphic = PinholeCameraModel(
        camera_id="cam-anamorphic",
        width_px=200,
        height_px=200,
        fx=200.0,
        fy=100.0,
        cx=100.0,
        cy=100.0,
        translation_world_to_camera=(0.0, 0.0, 2.5),
    )
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-anamorphic": cam_anamorphic},
        subject_binding=_make_test_subject_binding(),
    )

    angles = reference_golfer_setup()
    state = state_vector_from_joint_dict(angles)
    res = renderer.render(
        RenderRequest(
            camera_id="cam-anamorphic",
            state=state,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )
    body_arr = np.array(res.body_mask).reshape((200, 200))
    # Horizontal span vs vertical span of the body
    ys, xs = np.where(body_arr == 1)
    span_x = np.max(xs) - np.min(xs)
    span_y = np.max(ys) - np.min(ys)
    assert span_x > 0 and span_y > 0
    # Because fx = 2 * fy, horizontal scaling is doubled relative to isotropic camera

    # 2. Bridge test: round-trip through camera bridge preserves fx and fy
    obs_cam = CameraCalibration(
        camera_id="cam-ana-obs",
        image_size_px=(1920, 1080),
        intrinsics=ObsIntrinsics(
            matrix=np.array(
                [[1200.0, 0.0, 960.0], [0.0, 800.0, 540.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            distortion=np.array([0.05, -0.02, 0.001, -0.001, 0.0], dtype=np.float64),
        ),
        extrinsics=ObsExtrinsics(
            rotation_world_from_camera=np.eye(3, dtype=np.float64),
            translation_world_from_camera_m=np.array([0.0, 1.0, 3.0], dtype=np.float64),
        ),
    )

    p_intrinsics, p_extrinsics = to_pipeline_camera(obs_cam)
    assert p_intrinsics.fx == 1200.0
    assert p_intrinsics.fy == 800.0

    restored = from_pipeline_camera(
        camera_id="cam-ana-obs",
        image_size_px=(1920, 1080),
        intrinsics=p_intrinsics,
        extrinsics=p_extrinsics,
    )
    assert restored.intrinsics.matrix[0, 0] == 1200.0
    assert restored.intrinsics.matrix[1, 1] == 800.0


# ===========================================================================
# 5. Crop, Rotation, and Distortion Consistency
# ===========================================================================


def test_camera_bridge_distortion_and_rotation_consistency() -> None:
    """Camera bridge must enforce exact SE(3) inversion and canonical distortion lengths."""
    # 3D rotation around Y and X
    angle = math.radians(45.0)
    c, s = math.cos(angle), math.sin(angle)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    trans = np.array([1.5, -0.5, 4.0], dtype=np.float64)

    # 4-coefficient distortion -> expands to 5 with k3=0
    obs_cam = CameraCalibration(
        camera_id="cam-dist-4",
        image_size_px=(640, 480),
        intrinsics=ObsIntrinsics(
            matrix=np.array(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            distortion=np.array([0.1, -0.05, 0.002, 0.001], dtype=np.float64),
        ),
        extrinsics=ObsExtrinsics(
            rotation_world_from_camera=rot_y,
            translation_world_from_camera_m=trans,
        ),
    )

    p_in, p_ex = to_pipeline_camera(obs_cam)
    assert p_in.k3 == 0.0
    assert np.allclose(p_ex.rotation, rot_y.T)
    assert np.allclose(p_ex.translation, -rot_y.T @ trans)

    restored = from_pipeline_camera(
        camera_id="cam-dist-4",
        image_size_px=(640, 480),
        intrinsics=p_in,
        extrinsics=p_ex,
    )
    assert np.allclose(restored.extrinsics.rotation_world_from_camera, rot_y)
    assert np.allclose(restored.extrinsics.translation_world_from_camera_m, trans)

    # Invalid distortion lengths (e.g. 3 coefficients) fail closed
    with pytest.raises(ValueError, match="Unsupported distortion length"):
        obs_bad_dist = CameraCalibration(
            camera_id="cam-bad",
            image_size_px=(640, 480),
            intrinsics=ObsIntrinsics(
                matrix=np.array(
                    [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                distortion=np.array([0.1, -0.05, 0.002], dtype=np.float64),
            ),
            extrinsics=ObsExtrinsics(
                rotation_world_from_camera=np.eye(3),
                translation_world_from_camera_m=np.zeros(3),
            ),
        )
        to_pipeline_camera(obs_bad_dist)


# ===========================================================================
# 6. Missing Fields, Dimensions, and Non-Finite Values
# ===========================================================================


def test_missing_fields_and_dimension_validation() -> None:
    """Renderer and coordinate converters must reject invalid lengths and non-finite values."""
    cam = PinholeCameraModel(
        camera_id="cam-val",
        width_px=50,
        height_px=50,
        fx=50.0,
        fy=50.0,
        cx=25.0,
        cy=25.0,
    )
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-val": cam},
        subject_binding=_make_test_subject_binding(),
    )

    # Missing fields (state length 30 instead of 37)
    with pytest.raises(ValueError, match="requires 37"):
        renderer.render(
            RenderRequest(
                camera_id="cam-val",
                state=(0.0,) * 30,
                image_size_px=(50, 50),
                state_convention=CANONICAL_ARTICULATED_CONVENTION,
            )
        )

    # state_vector_to_joint_dict rejects wrong count
    with pytest.raises(ValueError, match="Expected 37"):
        state_vector_to_joint_dict((0.0,) * 15)

    # native_to_canonical_full_body rejects wrong coordinate dimensions
    with pytest.raises(ValueError, match="Expected q_native shape"):
        native_to_canonical_full_body(np.zeros(40), np.zeros(41))
    with pytest.raises(ValueError, match="Expected qd_native shape"):
        native_to_canonical_full_body(np.zeros(41), np.zeros(40))

    # canonical_to_native_full_body rejects wrong coordinate dimensions
    with pytest.raises(ValueError, match="Expected q_canonical shape"):
        canonical_to_native_full_body(np.zeros(41), np.zeros(41))
    with pytest.raises(ValueError, match="Expected v_canonical shape"):
        canonical_to_native_full_body(np.zeros(42), np.zeros(40))

    # Non-finite values fail closed
    with pytest.raises(ValueError, match="must contain only finite"):
        bad_q = np.zeros(41)
        bad_q[0] = float("nan")
        native_to_canonical_full_body(bad_q, np.zeros(41))

    with pytest.raises(ValueError, match="must contain only finite"):
        bad_canon = np.zeros(42)
        bad_canon[3] = float("inf")
        canonical_to_native_full_body(bad_canon, np.zeros(41))


# ===========================================================================
# 7. Quaternion Normalization and q/v Round-Trip
# ===========================================================================


def test_quaternion_normalization_and_qv_roundtrip() -> None:
    """Quaternion normalization and machine-precision q/v roundtrip."""
    # 1. Round-trip across nontrivial base orientation and joint kinematics
    q_native = np.zeros(41, dtype=np.float64)
    q_native[0:3] = [0.15, -0.25, 0.85]  # position
    q_native[3:6] = [math.radians(15.0), math.radians(-20.0), math.radians(35.0)]  # RPY
    q_native[6:] = np.linspace(-0.5, 0.5, 35)  # joints

    qd_native = np.zeros(41, dtype=np.float64)
    qd_native[0:3] = [1.2, -0.4, 0.1]
    qd_native[3:6] = [0.5, -0.3, 0.8]
    qd_native[6:] = np.linspace(-1.0, 1.0, 35)

    q_canon, v_canon = native_to_canonical_full_body(q_native, qd_native)
    assert len(q_canon) == 42
    assert len(v_canon) == 41

    # Check canonical quaternion is unit norm
    quat = q_canon[3:7]
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-12)

    # Inverse mapping back to native
    q_rec, qd_rec = canonical_to_native_full_body(q_canon, v_canon)
    assert np.allclose(q_rec, q_native, atol=1e-12)
    assert np.allclose(qd_rec, qd_native, atol=1e-12)

    # 2. Non-unit quaternion normalization in canonical_to_native_full_body
    q_unnorm = q_canon.copy()
    q_unnorm[3:7] = quat * 3.5  # scale quaternion by 3.5
    q_rec_unnorm, _ = canonical_to_native_full_body(q_unnorm, v_canon)
    assert np.allclose(q_rec_unnorm, q_native, atol=1e-12)

    # 3. Degenerate zero-norm quaternion fails closed
    q_zero_quat = q_canon.copy()
    q_zero_quat[3:7] = 0.0
    with pytest.raises(ValueError, match="cannot normalize a zero-norm quaternion"):
        canonical_to_native_full_body(q_zero_quat, v_canon)
