"""Unit tests for OpenCV video glyph renderer (FTO-8, #11293)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
)
from src.shared.python.force_overlay.renderers.opencv_glyphs import (
    HypothesisProjector,
    PinholeProjector,
    VideoGlyphReceipt,
    VideoGlyphStyle,
    draw_glyphs_on_frame,
    draw_legend_box,
)
from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection


def _make_dummy_arrow(
    label: str = "test_arrow",
    tail_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tip_m: tuple[float, float, float] = (1.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
) -> ArrowGlyph:
    tail_arr = np.asarray(tail_m, dtype=np.float64)
    tip_arr = np.asarray(tip_m, dtype=np.float64)
    head_base_arr = tip_arr - 0.2 * (tip_arr - tail_arr)
    head_base_m = (
        float(head_base_arr[0]),
        float(head_base_arr[1]),
        float(head_base_arr[2]),
    )
    return ArrowGlyph(
        label=label,
        kind="applied",
        tail_m=tail_m,
        tip_m=tip_m,
        head_base_m=head_base_m,
        shaft_radius_m=0.01,
        head_radius_m=0.024,
        rgba=rgba,
        magnitude=500.0,
        units="N",
        clamped=False,
    )


def _make_glyph_set(
    arrows: tuple[ArrowGlyph, ...] = (),
    torque_arcs: tuple[TorqueArcGlyph, ...] = (),
    legend: LegendSpec | None = None,
) -> GlyphSet:
    return GlyphSet(
        time_s=0.0,
        arrows=arrows,
        torque_arcs=torque_arcs,
        legend=legend or LegendSpec(force_reference_n=500.0),
    )


def _synthetic_camera(
    w: int = 640,
    h: int = 480,
    pos: tuple[float, float, float] = (0.0, 0.0, 5.0),
    tgt: tuple[float, float, float] = (0.0, 0.0, 0.0),
    distortion: np.ndarray | None = None,
) -> PinholeCamera:
    K = intrinsics_from_fov(w, h, 60.0)
    R = look_at(np.asarray(pos), np.asarray(tgt))
    return PinholeCamera(
        camera_id="synthetic_test_cam",
        matrix=K,
        rotation_world_from_camera=R,
        translation_world_from_camera_m=np.asarray(pos),
        image_size_px=(w, h),
        distortion=distortion,
    )


@pytest.mark.unit
def test_projected_endpoints_match_pinhole_and_pixels_exist() -> None:
    """A camera looking down -z at a +x arrow: tail and tip pixels match pinhole values, and arrow pixels exist."""
    w, h = 640, 480
    cam = _synthetic_camera(w, h, pos=(0.0, 0.0, 5.0), tgt=(0.0, 0.0, 0.0))
    projector = PinholeProjector(cam, world_frame="adr0041_world")

    arrow = _make_dummy_arrow(
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(1.0, 0.0, 0.0),
        rgba=(1.0, 0.0, 0.0, 1.0),  # Pure red in RGB, BGR is (0, 0, 255)
    )
    glyph_set = _make_glyph_set(arrows=(arrow,))

    frame = np.full((h, w, 3), 200, dtype=np.uint8)  # Light gray background
    out_frame, receipt = draw_glyphs_on_frame(
        frame,
        glyph_set,
        projector,
        world_frame="adr0041_world",
        inplace=False,
    )

    assert receipt.drawn == 1
    assert receipt.skipped_behind_camera == 0
    assert receipt.skipped_out_of_frame == 0

    # Hand-computed pinhole values:
    # Camera at (0, 0, 5) looking at (0, 0, 0): depth = 5.0
    # Tail (0, 0, 0) -> (cx, cy) = (320, 240)
    # Tip (1, 0, 0) -> (fx * (1/5) + cx, cy)
    K = cam.matrix
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    expected_tail = np.array([cx, cy])
    expected_tip = np.array([fx * (1.0 / 5.0) + cx, cy])

    # Check that red pixels exist along the line between expected_tail and expected_tip
    # BGR pure red has high BGR[2] and low BGR[0], BGR[1]
    red_mask = (
        (out_frame[:, :, 2] > 200)
        & (out_frame[:, :, 0] < 50)
        & (out_frame[:, :, 1] < 50)
    )
    assert np.any(red_mask), "Arrow-coloured red pixels must exist along the shaft"

    # Red pixels should be concentrated near the line y = 240, between x = 320 and x = expected_tip[0]
    ys, xs = np.where(red_mask)
    assert np.allclose(np.mean(ys), cy, atol=2.0)
    assert np.min(xs) >= int(expected_tail[0]) - 5
    assert np.max(xs) <= int(expected_tip[0]) + 5


@pytest.mark.unit
def test_halo_pixels_dark_flank_the_shaft() -> None:
    """Halo pixels (#101010 dark) flank the arrow shaft."""
    w, h = 640, 480
    cam = _synthetic_camera(w, h, pos=(0.0, 0.0, 5.0), tgt=(0.0, 0.0, 0.0))
    projector = PinholeProjector(cam, world_frame="adr0041_world")

    arrow = _make_dummy_arrow(
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(1.0, 0.0, 0.0),
        rgba=(1.0, 0.0, 0.0, 1.0),
    )
    glyph_set = _make_glyph_set(arrows=(arrow,))

    frame = np.full((h, w, 3), 255, dtype=np.uint8)  # White background
    out_frame, receipt = draw_glyphs_on_frame(
        frame,
        glyph_set,
        projector,
        world_frame="adr0041_world",
        inplace=False,
    )
    assert receipt.drawn == 1

    # Halo is dark (#101010 with alpha 0.7 blended onto white 255):
    # 0.7 * 16 + 0.3 * 255 = 11.2 + 76.5 = ~88
    # The shaft is y=240. Pixels at y=238 and y=242 along x=360 should be significantly darker than 255
    col_x = 360
    # Center pixel is red
    assert out_frame[240, col_x, 2] > 200
    # Flanking pixels (y = 237 or 243) should be darker than white
    assert out_frame[237, col_x, 0] < 200 or out_frame[243, col_x, 0] < 200


@pytest.mark.unit
def test_arrow_behind_camera_is_skipped_and_counted() -> None:
    """An arrow behind the camera is skipped and counted in skipped_behind_camera."""
    cam = _synthetic_camera(pos=(0.0, 0.0, 5.0), tgt=(0.0, 0.0, 0.0))
    projector = PinholeProjector(cam, world_frame="adr0041_world")

    # Point at z=10 is behind camera (camera is at z=5 looking toward z=0)
    arrow = _make_dummy_arrow(
        tail_m=(0.0, 0.0, 10.0),
        tip_m=(1.0, 0.0, 10.0),
    )
    glyph_set = _make_glyph_set(arrows=(arrow,), legend=LegendSpec())

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out_frame, receipt = draw_glyphs_on_frame(
        frame,
        glyph_set,
        projector,
        world_frame="adr0041_world",
    )
    assert receipt.drawn == 0
    assert receipt.skipped_behind_camera == 1
    assert np.array_equal(out_frame, frame)


@pytest.mark.unit
def test_arrow_crossing_edge_is_clipped_and_drawn() -> None:
    """An arrow crossing the image edge is clipped, drawn and counted as drawn."""
    w, h = 640, 480
    cam = _synthetic_camera(w, h, pos=(0.0, 0.0, 5.0), tgt=(0.0, 0.0, 0.0))
    projector = PinholeProjector(cam, world_frame="adr0041_world")

    # Arrow starting in center (0, 0, 0) and extending far right to (10, 0, 0)
    arrow = _make_dummy_arrow(
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(10.0, 0.0, 0.0),
        rgba=(0.0, 1.0, 0.0, 1.0),  # Green
    )
    glyph_set = _make_glyph_set(arrows=(arrow,))

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    out_frame, receipt = draw_glyphs_on_frame(
        frame,
        glyph_set,
        projector,
        world_frame="adr0041_world",
    )
    assert receipt.drawn == 1
    # Green pixels must exist up to the right boundary
    green_mask = (out_frame[:, :, 1] > 200) & (out_frame[:, :, 0] < 50)
    assert np.any(green_mask)
    assert np.max(np.where(green_mask)[1]) >= w - 5


@pytest.mark.unit
def test_frame_mismatch_raises_value_error() -> None:
    """A projector frame mismatch raises ValueError."""
    cam = _synthetic_camera()
    projector = PinholeProjector(cam, world_frame="adr0041_world")
    glyph_set = _make_glyph_set()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Frame mismatch"):
        draw_glyphs_on_frame(
            frame,
            glyph_set,
            projector,
            world_frame="other_frame",
        )


@pytest.mark.unit
def test_4k_frame_gets_thicker_lines_than_720p() -> None:
    """A 4K frame gets thicker lines than a 720p frame."""
    style_720p = VideoGlyphStyle.for_height(720)
    style_4k = VideoGlyphStyle.for_height(2160)
    assert style_4k.line_px > style_720p.line_px
    assert style_4k.head_px > style_720p.head_px


@pytest.mark.unit
def test_input_array_unchanged_when_inplace_false() -> None:
    """The input array is unchanged when inplace=False."""
    cam = _synthetic_camera()
    projector = PinholeProjector(cam, world_frame="adr0041_world")
    arrow = _make_dummy_arrow()
    glyph_set = _make_glyph_set(arrows=(arrow,))

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame_orig = frame.copy()

    out_frame, _ = draw_glyphs_on_frame(
        frame,
        glyph_set,
        projector,
        world_frame="adr0041_world",
        inplace=False,
    )
    assert np.array_equal(frame, frame_orig)
    assert not np.array_equal(out_frame, frame)


@pytest.mark.unit
def test_distortion_path_compares_with_cv2_project_points() -> None:
    """A distortion-enabled PinholeCamera uses the distortion path and matches cv2.projectPoints within 0.5 px."""
    w, h = 640, 480
    dist = np.array([-0.1, 0.05, 0.001, -0.001, 0.0], dtype=np.float64)
    cam = _synthetic_camera(
        w, h, pos=(0.0, 0.0, 5.0), tgt=(0.0, 0.0, 0.0), distortion=dist
    )
    projector = PinholeProjector(cam, world_frame="adr0041_world")

    pts_w = np.array([[0.5, 0.5, 0.0], [1.0, -0.5, 0.0]], dtype=np.float64)
    px_proj, valid = projector.project(pts_w)

    # Reference cv2.projectPoints
    r_w2c = cam.rotation_world_from_camera.T
    t_w2c = -r_w2c @ cam.translation_world_from_camera_m
    rvec, _ = cv2.Rodrigues(r_w2c)
    px_cv, _ = cv2.projectPoints(pts_w, rvec, t_w2c, cam.matrix, dist)
    px_cv = px_cv.reshape(-1, 2)

    np.testing.assert_allclose(px_proj, px_cv, atol=0.5)


@pytest.mark.unit
def test_legend_box_differs_from_background() -> None:
    """The legend box text appears and alters the bottom-left image region."""
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    orig = frame.copy()

    legend = LegendSpec(
        force_reference_n=500.0,
        torque_reference_nm=50.0,
        kinds_present=("applied",),
        unavailable_labels=("reaction_forces",),
        engine="mujoco",
    )
    style = VideoGlyphStyle.for_height(480)

    draw_legend_box(
        frame, legend, style, qualification_note="unqualified camera hypothesis"
    )

    # Bottom-left box region (e.g. x: [0..250], y: [350..480]) must differ from orig
    box_diff = np.abs(
        frame[350:480, 0:250].astype(int) - orig[350:480, 0:250].astype(int)
    )
    assert np.sum(box_diff > 10) > 100, (
        "Legend box must alter pixels in the bottom-left"
    )


@pytest.mark.unit
def test_hypothesis_projector_behind_camera_handling() -> None:
    """HypothesisProjector handles behind-camera errors per-point gracefully."""
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array(
        [0.0, 0.0, -1.0]
    )  # Camera at (0, 0, 1) in world, points with z < 1.0 are behind
    proj = CameraProjection(intrinsics=K, rotation=R, translation=t)
    projector = HypothesisProjector(proj, world_frame="hypothesis_world")

    pts = np.array(
        [
            [0.0, 0.0, 5.0],  # In front (z=5, z+t_z = 4 > 0)
            [0.0, 0.0, -5.0],  # Behind (z=-5, z+t_z = -6 <= 0)
        ]
    )
    px, valid = projector.project(pts)
    assert valid[0] is True or valid[0] == 1
    assert valid[1] is False or valid[1] == 0
    assert np.isnan(px[1, 0])
