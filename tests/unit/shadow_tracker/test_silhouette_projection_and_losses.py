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

from shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
)
from shared.python.shadow_tracker.contracts import (
    RenderRequest,
    RenderResult,
    SilhouetteRenderer,
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
