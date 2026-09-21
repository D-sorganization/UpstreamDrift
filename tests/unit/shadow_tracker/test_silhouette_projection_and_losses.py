"""Unit tests for Silhouette Rendering, Analytic Projection, and Residual Losses (ST-05, #10128).

Tests verify:
- Analytic landmark projection accuracy <= 0.5 px (satisfying Gate G1).
- Parity across camera transformations (crop, lens distortion, mirroring, points behind camera).
- Fulfillment of the SilhouetteRenderer protocol producing slotted RenderResult records.
- Valid-pixel aware silhouette residual and loss computation (occluded regions ignored).
- Safe abstention on all-invalid masks (no division by zero or NaN).
- Strictly monotonic loss increase under known contour displacement / shifts.
- Independent evaluation of thin-club and body channels to prevent body overlap masking club error.
"""

from __future__ import annotations

import math
import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    reference_golfer_setup,
)
from shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    SUBJECT_BINDING_SCHEMA_VERSION,
)
from shared.python.shadow_tracker.articulated_renderer import (
    CANONICAL_ARTICULATED_CONVENTION,
    CANONICAL_ARTICULATED_STATE_FIELDS,
    ArticulatedSilhouetteRenderer,
    state_vector_from_joint_dict,
    state_vector_to_joint_dict,
)
from shared.python.shadow_tracker.contracts import (
    RenderRequest,
    RenderResult,
    SilhouetteRenderer,
    SubjectModelBinding,
)
from shared.python.shadow_tracker.mask_records import MaskFrame
from shared.python.shadow_tracker.projection import (
    AnalyticSilhouetteRenderer,
    PinholeCameraModel,
    SilhouetteLossResult,
    compute_silhouette_loss,
    project_point_to_pixel,
)
from shared.python.shadow_tracker.source_records import FrameIdentity

pytestmark = pytest.mark.unit


@pytest.fixture
def base_frame_identity() -> FrameIdentity:
    return FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-proj-01",
        shot_id="shot-01",
        swing_id="swing-01",
        camera_id="cam-face-on",
        frame_id="f-001",
        pts_ticks=0,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.0,
        physical_time_reason="",
        frame_sha256="c" * 64,
    )


# ---------------------------------------------------------------------------
# 1. Gate G1: Analytic Landmark Projection Accuracy
# ---------------------------------------------------------------------------


def test_analytic_landmark_projection_g1_accuracy() -> None:
    # Camera at origin looking along +Z: R = I, t = [0, 0, 0]
    # fx = fy = 500.0, cx = 320.0, cy = 240.0, W = 640, H = 480
    camera = PinholeCameraModel(
        camera_id="cam-face-on",
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )

    # Known 3D landmark: X = 0.5 m, Y = -0.2 m, Z = 2.0 m
    # Analytic expected:
    # x_norm = 0.5 / 2.0 = 0.25
    # y_norm = -0.2 / 2.0 = -0.10
    # u = 500.0 * 0.25 + 320.0 = 125.0 + 320.0 = 445.0 px
    # v = 500.0 * (-0.10) + 240.0 = -50.0 + 240.0 = 190.0 px
    point_world = (0.5, -0.2, 2.0)
    u, v, is_visible = project_point_to_pixel(point_world, camera)

    assert is_visible is True
    assert abs(u - 445.0) <= 0.5  # Gate G1: <= 0.5 px tolerance
    assert abs(v - 190.0) <= 0.5


# ---------------------------------------------------------------------------
# 2. Camera Transforms Parity: Distortion, Mirroring, Crop, Occlusion
# ---------------------------------------------------------------------------


def test_projection_camera_transforms_parity() -> None:
    # 1. Points behind camera (Z <= 0) must be flagged non-visible
    camera_base = PinholeCameraModel(
        camera_id="cam-01",
        width_px=100,
        height_px=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
    )
    _, _, is_vis_behind = project_point_to_pixel((0.0, 0.0, -1.0), camera_base)
    assert is_vis_behind is False

    # 2. Points offscreen outside image boundaries must be flagged non-visible
    _, _, is_vis_offscreen = project_point_to_pixel((10.0, 0.0, 1.0), camera_base)
    assert is_vis_offscreen is False

    # 3. Mirror parity: horizontal flip maps u -> (W - 1) - u
    camera_mirrored = PinholeCameraModel(
        camera_id="cam-mirrored",
        width_px=100,
        height_px=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        is_mirrored=True,
    )
    u_orig, v_orig, _ = project_point_to_pixel((0.1, 0.1, 1.0), camera_base)
    u_mirr, v_mirr, _ = project_point_to_pixel((0.1, 0.1, 1.0), camera_mirrored)
    assert v_mirr == pytest.approx(v_orig)
    assert u_mirr == pytest.approx((100 - 1) - u_orig)

    # 4. Crop box parity: crop_box = (10, 20, 80, 70) offsets coords by min_x, min_y
    camera_cropped = PinholeCameraModel(
        camera_id="cam-cropped",
        width_px=100,
        height_px=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        crop_box=(10, 20, 80, 70),
    )
    u_crop, v_crop, is_vis_crop = project_point_to_pixel(
        (0.1, 0.1, 1.0), camera_cropped
    )
    assert u_crop == pytest.approx(u_orig - 10)
    assert v_crop == pytest.approx(v_orig - 20)
    assert is_vis_crop is True


# ---------------------------------------------------------------------------
# 3. SilhouetteRenderer Protocol Fulfillment
# ---------------------------------------------------------------------------


def test_analytic_renderer_fulfills_silhouette_renderer_protocol() -> None:
    camera = PinholeCameraModel(
        camera_id="cam-face-on",
        width_px=4,
        height_px=4,
        fx=10.0,
        fy=10.0,
        cx=2.0,
        cy=2.0,
    )

    renderer = AnalyticSilhouetteRenderer(cameras={"cam-face-on": camera})
    assert isinstance(renderer, SilhouetteRenderer)

    # State: 6 floats (e.g. 3D body center [X, Y, Z] + 3D clubhead [X, Y, Z])
    # Body centered at (0.0, 0.0, 5.0) -> projects to (2.0, 2.0)
    # Club at (0.5, 0.5, 5.0) -> projects to (3.0, 3.0)
    req = RenderRequest(
        camera_id="cam-face-on",
        state=(0.0, 0.0, 5.0, 0.5, 0.5, 5.0),
        image_size_px=(4, 4),
    )

    result = renderer.render(req)
    assert isinstance(result, RenderResult)
    assert len(result.body_mask) == 16
    assert len(result.club_mask) == 16
    assert len(result.visibility_mask) == 16
    assert all(p in (0, 1) for p in result.body_mask)
    assert all(p in (0, 1) for p in result.club_mask)
    assert all(p in (0, 1) for p in result.visibility_mask)


# ---------------------------------------------------------------------------
# 4. Valid-Pixel Aware Silhouette Loss & Occlusion Handling
# ---------------------------------------------------------------------------


def test_valid_pixel_aware_silhouette_loss(base_frame_identity: FrameIdentity) -> None:
    # 4 pixels (2x2)
    # idx 0: valid=1, observed body=1, rendered body=1 (match)
    # idx 1: valid=1, observed body=0, rendered body=1 (candidate false positive -> loss!)
    # idx 2: valid=0 (occluded barrier!), observed body=0, rendered body=1 (must NOT penalize candidate!)
    # idx 3: valid=1, observed body=0, rendered body=0 (true negative)
    width, height = 2, 2
    total_px = width * height

    valid = bytes([1, 1, 0, 1])
    obs_body = bytes([1, 0, 0, 0])
    obs_club = bytes([0, 0, 0, 0])

    observed = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=obs_body,
        club=obs_club,
        valid=valid,
        revision_id="rev-loss-test",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="",
    )

    rendered = RenderResult(
        body_mask=(1, 1, 1, 0),  # pixel 2 is 1, but pixel 2 is valid=0!
        club_mask=(0, 0, 0, 0),
        visibility_mask=(1, 1, 1, 1),
    )

    # Over valid pixels {0, 1, 3}:
    # observed body: {0}
    # rendered body: {0, 1}
    # intersection: {0} -> 1
    # union: {0, 1} -> 2
    # body IoU = 1 / 2 = 0.5
    loss_res = compute_silhouette_loss(rendered, observed)
    assert isinstance(loss_res, SilhouetteLossResult)
    assert loss_res.is_valid is True
    assert loss_res.valid_pixel_count == 3
    assert loss_res.body_iou == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. All-Invalid Mask Abstention
# ---------------------------------------------------------------------------


def test_all_invalid_mask_abstention_and_safety(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 2, 2
    total_px = width * height

    # 100% occluded view (e.g. blackout or obstacle)
    valid_empty = bytes([0] * total_px)
    obs_body = bytes([0] * total_px)
    obs_club = bytes([0] * total_px)

    observed = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=obs_body,
        club=obs_club,
        valid=valid_empty,
        revision_id="rev-blackout",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="Camera blackout",
    )

    rendered = RenderResult(
        body_mask=(1, 1, 0, 0),
        club_mask=(0, 0, 1, 0),
        visibility_mask=(1, 1, 1, 1),
    )

    loss_res = compute_silhouette_loss(rendered, observed)
    assert loss_res.is_valid is False
    assert loss_res.valid_pixel_count == 0
    assert not math.isnan(loss_res.combined_loss)


# ---------------------------------------------------------------------------
# 6. Monotonic Contour Shift Property
# ---------------------------------------------------------------------------


def test_monotonic_contour_shift_loss_increase(
    base_frame_identity: FrameIdentity,
) -> None:
    # 5x5 grid
    width, height = 5, 5
    total_px = width * height
    valid = bytes([1] * total_px)

    # Target: 3x3 block in the center (rows 1..3, cols 1..3)
    target_body = bytearray(total_px)
    for r in range(1, 4):
        for c in range(1, 4):
            target_body[r * width + c] = 1

    observed = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=bytes(target_body),
        club=bytes([0] * total_px),
        valid=valid,
        revision_id="rev-target",
        parent_revision_id=None,
        producer_id="gold",
        correction_note="",
    )

    losses = []
    # Shift candidate horizontally by dx = 0, 1, 2 pixels
    for dx in (0, 1, 2):
        cand_body = [0] * total_px
        for r in range(1, 4):
            for c in range(1, 4):
                new_c = c + dx
                if 0 <= new_c < width:
                    cand_body[r * width + new_c] = 1

        rendered = RenderResult(
            body_mask=tuple(cand_body),
            club_mask=tuple([0] * total_px),
            visibility_mask=tuple([1] * total_px),
        )
        res = compute_silhouette_loss(rendered, observed)
        losses.append(res.combined_loss)

    # Loss must be strictly monotonically increasing as displacement dx increases
    assert losses[0] < losses[1] < losses[2]
    assert losses[0] == pytest.approx(0.0)  # Perfect overlap -> zero loss


# ---------------------------------------------------------------------------
# 7. Thin-Club Separation
# ---------------------------------------------------------------------------


def test_thin_club_separation_independent_loss(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 4, 4
    total_px = width * height
    valid = bytes([1] * total_px)

    # Identical body overlap (both have body at pixel 0)
    body = bytes([1 if i == 0 else 0 for i in range(total_px)])
    # Club observed at pixel 5
    club_gold = bytes([1 if i == 5 else 0 for i in range(total_px)])

    observed = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club_gold,
        valid=valid,
        revision_id="rev-club-test",
        parent_revision_id=None,
        producer_id="gold",
        correction_note="",
    )

    # 1. Candidate with matching body but mismatched club (club at pixel 6 instead of 5)
    rendered_bad_club = RenderResult(
        body_mask=tuple(body),
        club_mask=tuple(1 if i == 6 else 0 for i in range(total_px)),
        visibility_mask=tuple([1] * total_px),
    )

    res_bad = compute_silhouette_loss(rendered_bad_club, observed)
    assert res_bad.body_iou == pytest.approx(1.0)  # Perfect body
    assert res_bad.club_iou == pytest.approx(0.0)  # Disjoint club
    assert res_bad.combined_loss > 0.0  # Combined loss catches club error!


# ---------------------------------------------------------------------------
# 8. Filled Primitive Area & Camera/Pose Invariants (Review Findings)
# ---------------------------------------------------------------------------


def test_renderer_renders_filled_primitive_area() -> None:
    """ST-05 / P1 finding: projected primitive must occupy independently calculated filled area.

    A sphere/ellipsoid/capsule primitive with radius > 0 at depth Z must project to a filled
    disk of radius r_px = fx * R / Z, NOT a single pixel!
    """
    camera = PinholeCameraModel(
        camera_id="cam-area",
        width_px=100,
        height_px=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
    )
    renderer = AnalyticSilhouetteRenderer(
        cameras={"cam-area": camera},
        body_radius_m=0.2,  # 20 cm radius body primitive
        club_radius_m=0.05,  # 5 cm radius clubhead primitive
    )

    # State: 3-element reference position (bx, by, bz)
    # Placed at (0.0, 0.0, 2.0) -> center projects to (cx, cy) = (50, 50)
    # Expected radius: r_px = 100 * 0.2 / 2.0 = 10 pixels!
    # Expected area: pi * 10^2 ~ 314 pixels.
    req = RenderRequest(
        camera_id="cam-area",
        state=(0.0, 0.0, 2.0),
        image_size_px=(100, 100),
    )
    result = renderer.render(req)

    foreground_count = result.body_mask.count(1)
    # Must NOT be 1 pixel! Must be a filled area consistent with primitive geometry
    assert foreground_count > 50
    assert 250 <= foreground_count <= 350


def test_pinhole_camera_model_rejects_zero_or_invalid_rotation() -> None:
    """ST-05 / P2 finding: PinholeCameraModel must reject all-zero rotation matrices."""
    with pytest.raises(
        ValueError, match="rotation matrix must be non-singular|invalid rotation"
    ):
        PinholeCameraModel(
            camera_id="cam-bad-rot",
            width_px=100,
            height_px=100,
            fx=100.0,
            fy=100.0,
            cx=50.0,
            cy=50.0,
            rotation_world_to_camera=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# 9. Correct Clipping, Non-Square Pixels, Crop Parity & Articulation (#10232)
# ---------------------------------------------------------------------------


def test_renderer_clips_partially_visible_shapes_with_offscreen_center() -> None:
    """ST-05 / #10232: shapes with centers outside image bounds must not be dropped.

    Reproduced review probe:
    Camera: 10x10, fx=fy=10, cx=5, cy=5.
    Body center: (-1.2, 0, 2), radius: 0.4 m.
    The center projects to u = -1.0, v = 5.0 (offscreen to the left).
    Radius in pixels is r_px = 10 * 0.4 / 2 = 2.0 px.
    The disk extends from u = -3.0 to u = 1.0, intersecting col 0 at rows 4, 5, 6.
    Must rasterize intersecting foreground pixels instead of returning 0.
    """
    camera = PinholeCameraModel(
        camera_id="cam-clip",
        width_px=10,
        height_px=10,
        fx=10.0,
        fy=10.0,
        cx=5.0,
        cy=5.0,
    )
    renderer = AnalyticSilhouetteRenderer(
        cameras={"cam-clip": camera},
        body_radius_m=0.4,
    )
    req = RenderRequest(
        camera_id="cam-clip",
        state=(-1.2, 0.0, 2.0),
        image_size_px=(10, 10),
    )
    result = renderer.render(req)
    fg_count = result.body_mask.count(1)
    assert fg_count > 0, (
        "Partially visible shape must not produce 0 pixels when center is offscreen"
    )
    # Specific pixels at col 0, rows 4, 5, 6 should be set
    assert result.body_mask[5 * 10 + 0] == 1
    assert result.body_mask[4 * 10 + 0] == 1
    assert result.body_mask[6 * 10 + 0] == 1


def test_renderer_anamorphic_projection_oracle_fx_not_equal_fy() -> None:
    """ST-05 / #10232: fx != fy must project spheres to ellipses, not circles using fx for both axes."""
    camera = PinholeCameraModel(
        camera_id="cam-anamorphic",
        width_px=100,
        height_px=100,
        fx=200.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
    )
    renderer = AnalyticSilhouetteRenderer(
        cameras={"cam-anamorphic": camera},
        body_radius_m=0.1,
    )
    req = RenderRequest(
        camera_id="cam-anamorphic",
        state=(0.0, 0.0, 2.0),
        image_size_px=(100, 100),
    )
    result = renderer.render(req)
    # rx = 200 * 0.1 / 2 = 10 px
    # ry = 100 * 0.1 / 2 = 5 px
    # Expected area: pi * 10 * 5 ~ 157 px
    fg_count = result.body_mask.count(1)
    assert 130 <= fg_count <= 180, f"Expected ellipse area ~157 px, got {fg_count}"

    # Check width vs height of rendered shape:
    row_50_cols = [c for c in range(100) if result.body_mask[50 * 100 + c] == 1]
    col_50_rows = [r for r in range(100) if result.body_mask[r * 100 + 50] == 1]
    width_px = len(row_50_cols)
    height_px = len(col_50_rows)
    assert 19 <= width_px <= 23, f"Expected width ~21, got {width_px}"
    assert 9 <= height_px <= 13, f"Expected height ~11, got {height_px}"


def test_renderer_enforces_camera_crop_dimensions_parity() -> None:
    """ST-05 / #10232: render request image_size_px must match camera effective dimensions."""
    camera_uncropped = PinholeCameraModel(
        camera_id="cam-dim-1",
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    camera_cropped = PinholeCameraModel(
        camera_id="cam-dim-2",
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        crop_box=(100, 50, 300, 250),  # width: 200, height: 200
    )
    renderer = AnalyticSilhouetteRenderer(
        cameras={"cam-dim-1": camera_uncropped, "cam-dim-2": camera_cropped},
        body_radius_m=0.1,
    )
    # Mismatch for uncropped camera:
    with pytest.raises(ValueError, match="effective dimensions|image_size_px"):
        renderer.render(
            RenderRequest(
                camera_id="cam-dim-1",
                state=(0.0, 0.0, 2.0),
                image_size_px=(320, 240),
            )
        )
    # Match for uncropped camera:
    res1 = renderer.render(
        RenderRequest(
            camera_id="cam-dim-1",
            state=(0.0, 0.0, 2.0),
            image_size_px=(640, 480),
        )
    )
    assert len(res1.body_mask) == 640 * 480

    # Mismatch for cropped camera:
    with pytest.raises(ValueError, match="effective dimensions|image_size_px"):
        renderer.render(
            RenderRequest(
                camera_id="cam-dim-2",
                state=(0.0, 0.0, 2.0),
                image_size_px=(640, 480),
            )
        )
    # Match for cropped camera:
    res2 = renderer.render(
        RenderRequest(
            camera_id="cam-dim-2",
            state=(0.0, 0.0, 2.0),
            image_size_px=(200, 200),
        )
    )
    assert len(res2.body_mask) == 200 * 200


def test_analytic_renderer_rejects_unsupported_states() -> None:
    """ST-05 / #10232: AnalyticSilhouetteRenderer must explicitly reject unsupported states."""
    camera = PinholeCameraModel(
        camera_id="cam-test",
        width_px=100,
        height_px=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
    )
    renderer = AnalyticSilhouetteRenderer(
        cameras={"cam-test": camera}, body_radius_m=0.1
    )
    # 7-element state without articulated binding must be rejected:
    with pytest.raises(
        ValueError, match="Unsupported state|AnalyticSilhouetteRenderer"
    ):
        renderer.render(
            RenderRequest(
                camera_id="cam-test",
                state=(0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0),
                image_size_px=(100, 100),
            )
        )
    # 4-element state must be rejected:
    with pytest.raises(
        ValueError, match="Unsupported state|AnalyticSilhouetteRenderer"
    ):
        renderer.render(
            RenderRequest(
                camera_id="cam-test",
                state=(0.0, 0.0, 2.0, 1.0),
                image_size_px=(100, 100),
            )
        )


def _make_test_subject_binding() -> SubjectModelBinding:
    return SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="sub_test_01",
        model_hash="0" * 64,
        joint_ids=("root", "spine", "shoulder_left", "shoulder_right"),
        body_ids=("torso", "head", "arm_left", "arm_right"),
        visual_envelope={"height_m": 1.80, "chest_width_m": 0.45, "depth_m": 0.28},
        mass_kg=78.5,
        scale_evidence="measured_anthropometry",
        handedness="right",
    )


# ---------------------------------------------------------------------------
# 10. Articulated Golfer Silhouette Rendering & Kinematic Binding (#10232)
# ---------------------------------------------------------------------------


def test_articulated_renderer_binds_subject_and_renders_separate_masks() -> None:
    """ST-05 / #10232: ArticulatedSilhouetteRenderer renders body and club into distinct channels."""
    camera = PinholeCameraModel(
        camera_id="cam-front",
        width_px=200,
        height_px=200,
        fx=150.0,
        fy=150.0,
        cx=100.0,
        cy=100.0,
        translation_world_to_camera=(0.0, -0.5, 2.5),
    )
    binding = _make_test_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-front": camera},
        subject_binding=binding,
    )
    assert isinstance(renderer, SilhouetteRenderer)

    angles = reference_golfer_setup()
    state_vec = state_vector_from_joint_dict(angles)
    assert len(state_vec) == 37
    assert len(state_vec) == len(CANONICAL_ARTICULATED_STATE_FIELDS)

    req = RenderRequest(
        camera_id="cam-front",
        state=state_vec,
        image_size_px=(200, 200),
        state_convention=CANONICAL_ARTICULATED_CONVENTION,
    )
    res = renderer.render(req)
    assert isinstance(res, RenderResult)
    assert len(res.body_mask) == 40000
    assert len(res.club_mask) == 40000

    body_px = res.body_mask.count(1)
    club_px = res.club_mask.count(1)
    assert body_px > 500, f"Expected substantial body silhouette, got {body_px} px"
    assert club_px > 50, f"Expected club silhouette, got {club_px} px"

    # Verify channels are separate: club pixels are distinct from pure body pixels
    overlap = sum(
        1 for b, c in zip(res.body_mask, res.club_mask, strict=True) if b and c
    )
    # Club extends out from hands, so club should have pixels not in body
    club_unique = sum(
        1 for b, c in zip(res.body_mask, res.club_mask, strict=True) if c and not b
    )
    assert club_unique > 30, (
        f"Club channel must not be completely swallowed by body channel: unique={club_unique}"
    )


def test_articulated_renderer_limb_joint_rotation_changes_body_mask() -> None:
    """ST-05 / #10232: Rotating arm/elbow joint directly alters the body silhouette mask."""
    camera = PinholeCameraModel(
        camera_id="cam-front",
        width_px=200,
        height_px=200,
        fx=150.0,
        fy=150.0,
        cx=100.0,
        cy=100.0,
        translation_world_to_camera=(0.0, -0.5, 2.5),
    )
    binding = _make_test_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-front": camera},
        subject_binding=binding,
    )

    angles_base = reference_golfer_setup()
    state_base = state_vector_from_joint_dict(angles_base)

    # Pose with 90 degree left elbow flexion
    angles_flex = dict(angles_base)
    angles_flex["LEStartPosition"] = 90.0
    state_flex = state_vector_from_joint_dict(angles_flex)

    res_base = renderer.render(
        RenderRequest(
            camera_id="cam-front",
            state=state_base,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )
    res_flex = renderer.render(
        RenderRequest(
            camera_id="cam-front",
            state=state_flex,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )

    # Compute IoU between base and flexed body masks
    intersection = sum(
        1
        for b1, b2 in zip(res_base.body_mask, res_flex.body_mask, strict=True)
        if b1 and b2
    )
    union = sum(
        1
        for b1, b2 in zip(res_base.body_mask, res_flex.body_mask, strict=True)
        if b1 or b2
    )
    iou = intersection / union
    # Joint rotation MUST change the mask silhouette: IoU must not be 1.0!
    assert iou < 0.95, f"Joint rotation must change body silhouette: got IoU={iou}"


def test_articulated_renderer_wrist_rotation_changes_club_mask() -> None:
    """ST-05 / #10232: Rotating wrist angle alters the club silhouette mask."""
    camera = PinholeCameraModel(
        camera_id="cam-front",
        width_px=200,
        height_px=200,
        fx=150.0,
        fy=150.0,
        cx=100.0,
        cy=100.0,
        translation_world_to_camera=(0.0, -0.5, 2.5),
    )
    binding = _make_test_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-front": camera},
        subject_binding=binding,
    )

    angles_base = reference_golfer_setup()
    state_base = state_vector_from_joint_dict(angles_base)

    # Pose with wrist angle altered by 45 degrees
    angles_hinge = dict(angles_base)
    angles_hinge["LFStartPosition"] = 45.0
    angles_hinge["LWStartPositionX"] = 30.0
    state_hinge = state_vector_from_joint_dict(angles_hinge)

    res_base = renderer.render(
        RenderRequest(
            camera_id="cam-front",
            state=state_base,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )
    res_hinge = renderer.render(
        RenderRequest(
            camera_id="cam-front",
            state=state_hinge,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )

    inter_club = sum(
        1
        for c1, c2 in zip(res_base.club_mask, res_hinge.club_mask, strict=True)
        if c1 and c2
    )
    union_club = sum(
        1
        for c1, c2 in zip(res_base.club_mask, res_hinge.club_mask, strict=True)
        if c1 or c2
    )
    iou_club = inter_club / union_club
    assert iou_club < 0.85, (
        f"Wrist/club rotation must change club silhouette: got IoU={iou_club}"
    )


def test_articulated_renderer_clips_geometry_extending_offscreen() -> None:
    """ST-05 / #10232: Limbs extending outside camera bounds are clipped cleanly without crashing."""
    # Small 40x40 viewport positioned close to the pelvis
    camera_tight = PinholeCameraModel(
        camera_id="cam-tight",
        width_px=40,
        height_px=40,
        fx=150.0,
        fy=150.0,
        cx=20.0,
        cy=20.0,
        translation_world_to_camera=(0.0, 0.0, 1.5),
    )
    binding = _make_test_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-tight": camera_tight},
        subject_binding=binding,
    )
    angles = reference_golfer_setup()
    state_vec = state_vector_from_joint_dict(angles)

    req = RenderRequest(
        camera_id="cam-tight",
        state=state_vec,
        image_size_px=(40, 40),
        state_convention=CANONICAL_ARTICULATED_CONVENTION,
    )
    res = renderer.render(req)
    # The golfer's torso/arms extend far outside this 40x40 viewport, but visible pixels must be rasterized
    assert res.body_mask.count(1) > 0


def test_articulated_renderer_rejects_unsupported_convention_and_lengths() -> None:
    """ST-05 / #10232: ArticulatedSilhouetteRenderer strictly validates state convention and length."""
    camera = PinholeCameraModel(
        camera_id="cam-valid",
        width_px=50,
        height_px=50,
        fx=50.0,
        fy=50.0,
        cx=25.0,
        cy=25.0,
    )
    binding = _make_test_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-valid": camera},
        subject_binding=binding,
    )
    # Wrong convention:
    with pytest.raises(
        ValueError, match="requires state_convention='canonical_articulated_v1'"
    ):
        renderer.render(
            RenderRequest(
                camera_id="cam-valid",
                state=(0.0, 0.0, 2.0),
                image_size_px=(50, 50),
                state_convention="point_landmarks",
            )
        )
    # Wrong length (e.g. 7 elements):
    with pytest.raises(
        ValueError,
        match="requires 37 \\(or 27\\) state elements for 'canonical_articulated_v1'",
    ):
        renderer.render(
            RenderRequest(
                camera_id="cam-valid",
                state=(0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0),
                image_size_px=(50, 50),
                state_convention=CANONICAL_ARTICULATED_CONVENTION,
            )
        )


def test_articulated_renderer_lower_limb_region_isolation() -> None:
    """ST-05 / #10232: Knee flexion rotates lower-limb segments while upper body is strictly isolated."""
    from shared.python.shadow_tracker.articulated_renderer import _BODY_SEGMENTS

    # Verify segments explicitly contain thighs, shins, and feet
    segment_pairs = {(p1, p2) for p1, p2, _ in _BODY_SEGMENTS}
    assert ("l_hip", "l_knee") in segment_pairs
    assert ("r_hip", "r_knee") in segment_pairs
    assert ("l_knee", "l_ankle") in segment_pairs
    assert ("r_knee", "r_ankle") in segment_pairs
    assert ("l_ankle", "l_foot") in segment_pairs
    assert ("r_ankle", "r_foot") in segment_pairs

    # Frontal camera framing whole golfer (height 200, width 200)
    camera = PinholeCameraModel(
        camera_id="cam-full",
        width_px=200,
        height_px=200,
        fx=130.0,
        fy=130.0,
        cx=100.0,
        cy=90.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
    )
    binding = _make_test_subject_binding()
    renderer = ArticulatedSilhouetteRenderer(
        cameras={"cam-full": camera},
        subject_binding=binding,
    )

    angles_base = reference_golfer_setup()
    state_base = state_vector_from_joint_dict(angles_base)

    angles_knee = dict(angles_base)
    # Flex left knee by 60 degrees
    angles_knee["LKneeStartPosition"] = 60.0
    state_knee = state_vector_from_joint_dict(angles_knee)

    res_base = renderer.render(
        RenderRequest(
            camera_id="cam-full",
            state=state_base,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )
    res_knee = renderer.render(
        RenderRequest(
            camera_id="cam-full",
            state=state_knee,
            image_size_px=(200, 200),
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
        )
    )

    # Pelvis is positioned at y=0, projects near cy=90 in this camera setup.
    # In rows 0..70 (upper torso, head, shoulders, arms), pixels must be IDENTICAL.
    upper_diffs = 0
    lower_diffs = 0
    for r in range(200):
        for c in range(200):
            idx = r * 200 + c
            b1 = res_base.body_mask[idx]
            b2 = res_knee.body_mask[idx]
            if b1 != b2:
                if r < 75:
                    upper_diffs += 1
                else:
                    lower_diffs += 1

    assert upper_diffs == 0, (
        f"Upper body pixels must not move when knee rotates: got {upper_diffs} diffs"
    )
    assert lower_diffs > 20, (
        f"Lower body pixels must change when knee rotates: got {lower_diffs} diffs"
    )


def test_articulated_renderer_near_plane_adversarial_bounded_work() -> None:
    """ST-05 / #10232: Segments crossing or extremely near the camera plane do not produce unbounded work."""
    import time
    from shared.python.shadow_tracker.articulated_renderer import _rasterize_3d_segment
    import numpy as np

    camera = PinholeCameraModel(
        camera_id="cam-near",
        width_px=100,
        height_px=100,
        fx=200.0,
        fy=200.0,
        cx=50.0,
        cy=50.0,
    )
    mask = [0] * (100 * 100)

    # Adversarial segment: starts at z = 0.0001 (0.1 mm in front of optical center) and ends at z = 2.0 m
    # Without bounded viewport work, projected distance is > 2,000,000 pixels.
    p1 = np.array([0.5, 0.5, 0.0001])
    p2 = np.array([0.0, 0.0, 2.0])

    t0 = time.perf_counter()
    _rasterize_3d_segment(mask, p1, p2, 0.05, camera, 100, 100)
    elapsed = time.perf_counter() - t0

    # Must complete in well under 100 milliseconds
    assert elapsed < 0.1, f"Near-plane rasterization took too long: {elapsed:.4f}s"
    assert sum(mask) > 0, "Visible portion of segment must be rasterized"


def test_articulated_renderer_coherent_subject_morphology_scaling() -> None:
    """ST-05 / #10232: Subject height scales both envelope radii and segment lengths coherently."""
    camera = PinholeCameraModel(
        camera_id="cam-scale",
        width_px=200,
        height_px=200,
        fx=100.0,
        fy=100.0,
        cx=100.0,
        cy=100.0,
        translation_world_to_camera=(0.0, 0.0, 3.5),
    )

    binding_short = SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="subj-short",
        model_hash="0" * 64,
        joint_ids=("root", "spine", "shoulder_left", "shoulder_right"),
        body_ids=("torso", "head", "arm_left", "arm_right"),
        visual_envelope={"height_m": 1.50, "chest_width_m": 0.40, "depth_m": 0.25},
        mass_kg=55.0,
        scale_evidence="measured_anthropometry",
        handedness="right",
    )
    binding_tall = SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="subj-tall",
        model_hash="0" * 64,
        joint_ids=("root", "spine", "shoulder_left", "shoulder_right"),
        body_ids=("torso", "head", "arm_left", "arm_right"),
        visual_envelope={"height_m": 2.05, "chest_width_m": 0.50, "depth_m": 0.32},
        mass_kg=95.0,
        scale_evidence="measured_anthropometry",
        handedness="right",
    )

    renderer_short = ArticulatedSilhouetteRenderer(
        cameras={"cam-scale": camera},
        subject_binding=binding_short,
    )
    renderer_tall = ArticulatedSilhouetteRenderer(
        cameras={"cam-scale": camera},
        subject_binding=binding_tall,
    )

    angles = reference_golfer_setup()
    state = state_vector_from_joint_dict(angles)
    req = RenderRequest(
        camera_id="cam-scale",
        state=state,
        image_size_px=(200, 200),
        state_convention=CANONICAL_ARTICULATED_CONVENTION,
    )

    res_short = renderer_short.render(req)
    res_tall = renderer_tall.render(req)

    area_short = res_short.body_mask.count(1)
    area_tall = res_tall.body_mask.count(1)

    assert area_short > 0
    assert area_tall > area_short * 1.3, (
        f"Tall subject must produce significantly larger silhouette than short subject: "
        f"tall={area_tall}, short={area_short}"
    )
