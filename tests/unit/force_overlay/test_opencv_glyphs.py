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
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.opencv_glyphs import (
    HypothesisProjector,
    PinholeProjector,
    VideoGlyphReceipt,
    VideoGlyphStyle,
    draw_glyphs_on_frame,
    draw_legend_box,
)


class _MockCameraProjection:
    """Mock projection for testing HypothesisProjector error-handling without heavy stacks."""

    def __init__(
        self, intrinsics: np.ndarray, rotation: np.ndarray, translation: np.ndarray
    ) -> None:
        self.intrinsics = intrinsics
        self.rotation = rotation
        self.translation = translation

    def project(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        cam = (p @ self.rotation.T) + self.translation
        if np.any(cam[:, 2] <= 0):
            raise ValueError("Point behind camera")
        uv_norm = cam[:, :2] / cam[:, 2:3]
        return uv_norm @ self.intrinsics[:2, :2].T + self.intrinsics[:2, 2]


@pytest.fixture
def synthetic_camera() -> PinholeCamera:
    """Camera at (0, 0, 5) looking down -z at target (0, 0, 0) with up (0, 1, 0).

    In ADR-0041:
    pos = (0, 0, 5), tgt = (0, 0, 0).
    Image: 1920x1080, FOV 60 deg.
    Principal point: (960, 540).
    Depth to origin is 5.0m.
    """
    w, h = 1920, 1080
    k = intrinsics_from_fov(w, h, 60.0)
    pos = np.array([0.0, 0.0, 5.0])
    tgt = np.array([0.0, 0.0, 0.0])
    r = look_at(pos, tgt, up=np.array([0.0, 1.0, 0.0]))
    return PinholeCamera(
        camera_id="cam_test",
        matrix=k,
        rotation_world_from_camera=r,
        translation_world_from_camera_m=pos,
        image_size_px=(w, h),
    )


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Blank light grey image 1920x1080 BGR."""
    return np.full((1080, 1920, 3), 200, dtype=np.uint8)


def _make_glyph_set(
    arrows: list[ArrowGlyph] | None = None,
    arcs: list[TorqueArcGlyph] | None = None,
    unavailable: tuple[str, ...] = (),
) -> GlyphSet:
    return GlyphSet(
        time_s=0.0,
        arrows=tuple(arrows or []),
        torque_arcs=tuple(arcs or []),
        legend=LegendSpec(
            force_reference_n=100.0,
            force_reference_length_m=0.1,
            torque_reference_nm=10.0,
            torque_reference_radius_m=0.05,
            kinds_present=("reaction",),
            unavailable_labels=unavailable,
            engine="pinocchio",
            source_labels=("test",),
        ),
    )


@pytest.mark.unit
def test_pinhole_projection_and_arrow_colors(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """A camera looking down -z at a +x arrow: tail and tip match pinhole values within 0.5px."""
    projector = PinholeProjector(synthetic_camera)
    # Hand calculation:
    # Origin (0, 0, 0) projects to principal point (960, 540)
    # Point (1, 0, 0) in ADR-0041:
    # Camera coords: x_c = 1.0, y_c = 0.0, z_c = 5.0
    # u = 960 + fx * (1.0 / 5.0), v = 540
    fx = synthetic_camera.matrix[0, 0]
    expected_u_tip = 960.0 + fx * (1.0 / 5.0)

    tail = (0.0, 0.0, 0.0)
    tip = (1.0, 0.0, 0.0)
    head_base = (0.8, 0.0, 0.0)
    arrow = ArrowGlyph(
        label="test_arrow",
        kind="reaction",
        tail_m=tail,
        tip_m=tip,
        head_base_m=head_base,
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(1.0, 0.0, 0.0, 1.0),  # Pure red in RGBA -> BGR is (0, 0, 255)
        magnitude=100.0,
        units="N",
        clamped=False,
    )
    glyphs = _make_glyph_set(arrows=[arrow])

    # Check projector projection directly
    pts = np.array([tail, tip])
    pixels, valid = projector.project(pts)
    assert valid.all()
    np.testing.assert_allclose(pixels[0], [960.0, 540.0], atol=0.5)
    np.testing.assert_allclose(pixels[1], [expected_u_tip, 540.0], atol=0.5)

    # Render on frame
    out_frame, receipt = draw_glyphs_on_frame(synthetic_frame, glyphs, projector)
    assert receipt.drawn == 1
    assert receipt.skipped_behind_camera == 0
    assert receipt.skipped_out_of_frame == 0

    # Verify red pixels exist along the line y=540, between x=960 and expected_u_tip
    mid_x = int(round((960.0 + expected_u_tip) / 2.0))
    # Red in BGR is (0, 0, 255)
    sample_bgr = out_frame[540, mid_x]
    assert sample_bgr[2] > 200  # High red
    assert sample_bgr[0] < 50  # Low blue


@pytest.mark.unit
def test_halo_pixels_flank_shaft(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """Halo pixels (dark) flank the shaft above and below horizontal line."""
    projector = PinholeProjector(synthetic_camera)
    arrow = ArrowGlyph(
        label="test_arrow",
        kind="reaction",
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(1.0, 0.0, 0.0),
        head_base_m=(0.8, 0.0, 0.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(0.0, 1.0, 0.0, 1.0),  # Green
        magnitude=50.0,
        units="N",
        clamped=False,
    )
    glyphs = _make_glyph_set(arrows=[arrow])
    style = VideoGlyphStyle(
        line_px=3, halo_px=7, halo_color_bgr=(16, 16, 16), halo_alpha=1.0
    )
    out_frame, receipt = draw_glyphs_on_frame(
        synthetic_frame, glyphs, projector, style=style
    )
    assert receipt.drawn == 1

    fx = synthetic_camera.matrix[0, 0]
    mid_x = int(round(960.0 + fx * 0.1))  # along the shaft
    # Center pixel y=540 should be green
    assert out_frame[540, mid_x, 1] > 200
    # Flanking pixels (y=540 +/- 2 or 3) should be darker than background (200) due to dark halo
    flank_pixel_above = out_frame[540 - 2, mid_x]
    flank_pixel_below = out_frame[540 + 2, mid_x]
    assert np.mean(flank_pixel_above) < 180
    assert np.mean(flank_pixel_below) < 180


@pytest.mark.unit
def test_arrow_behind_camera_skipped_and_counted(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """An arrow located behind the camera (z > 5) is skipped and counted."""
    projector = PinholeProjector(synthetic_camera)
    arrow = ArrowGlyph(
        label="behind_camera",
        kind="reaction",
        tail_m=(
            0.0,
            0.0,
            10.0,
        ),  # camera is at z=5 looking at z=0, so z=10 is behind camera
        tip_m=(1.0, 0.0, 10.0),
        head_base_m=(0.8, 0.0, 10.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(1.0, 0.0, 0.0, 1.0),
        magnitude=50.0,
        units="N",
        clamped=False,
    )
    glyphs = _make_glyph_set(arrows=[arrow])
    out_frame, receipt = draw_glyphs_on_frame(synthetic_frame, glyphs, projector)
    assert receipt.drawn == 0
    assert receipt.skipped_behind_camera == 1
    assert receipt.skipped_out_of_frame == 0


@pytest.mark.unit
def test_arrow_crossing_edge_clipped_and_drawn(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """An arrow crossing the image edge is clipped, drawn, and counted as drawn."""
    projector = PinholeProjector(synthetic_camera)
    # Long arrow from origin (960, 540) far out to the right (+x = 10m -> u ~ 960 + 10*fx/5 ~ 4000px)
    arrow = ArrowGlyph(
        label="crossing_edge",
        kind="reaction",
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(10.0, 0.0, 0.0),
        head_base_m=(9.5, 0.0, 0.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(0.0, 0.0, 1.0, 1.0),
        magnitude=500.0,
        units="N",
        clamped=False,
    )
    glyphs = _make_glyph_set(arrows=[arrow])
    out_frame, receipt = draw_glyphs_on_frame(synthetic_frame, glyphs, projector)
    assert receipt.drawn == 1
    assert receipt.skipped_out_of_frame == 0


@pytest.mark.unit
def test_arrow_completely_out_of_frame_counted(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """An arrow far to the side with no clipped intersection is counted as skipped_out_of_frame."""
    projector = PinholeProjector(synthetic_camera)
    arrow = ArrowGlyph(
        label="far_offscreen",
        kind="reaction",
        tail_m=(50.0, 0.0, 0.0),
        tip_m=(51.0, 0.0, 0.0),
        head_base_m=(50.8, 0.0, 0.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(0.0, 0.0, 1.0, 1.0),
        magnitude=50.0,
        units="N",
        clamped=False,
    )
    glyphs = _make_glyph_set(arrows=[arrow])
    out_frame, receipt = draw_glyphs_on_frame(synthetic_frame, glyphs, projector)
    assert receipt.drawn == 0
    assert receipt.skipped_out_of_frame == 1


@pytest.mark.unit
def test_frame_mismatch_raises(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """A projector frame mismatch raises ValueError."""
    projector = PinholeProjector(synthetic_camera, world_frame="adr0041")
    glyphs = _make_glyph_set()
    with pytest.raises(ValueError, match="Projector world frame mismatch"):
        draw_glyphs_on_frame(
            synthetic_frame, glyphs, projector, world_frame="canonical_z_up"
        )


@pytest.mark.unit
def test_4k_vs_720p_line_thickness() -> None:
    """A 4K frame gets thicker lines than a 720p frame."""
    style = VideoGlyphStyle()
    line_4k = style.resolve_line_px(2160)
    line_720p = style.resolve_line_px(720)
    assert line_4k > line_720p
    assert line_4k == 5
    assert line_720p == 2


@pytest.mark.unit
def test_inplace_false_leaves_input_unchanged(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """The input array is unchanged when inplace=False."""
    projector = PinholeProjector(synthetic_camera)
    arrow = ArrowGlyph(
        label="test",
        kind="reaction",
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(0.5, 0.0, 0.0),
        head_base_m=(0.4, 0.0, 0.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(1.0, 0.0, 0.0, 1.0),
        magnitude=10.0,
        units="N",
        clamped=False,
    )
    glyphs = _make_glyph_set(arrows=[arrow])
    frame_copy = synthetic_frame.copy()
    out_frame, receipt = draw_glyphs_on_frame(
        synthetic_frame, glyphs, projector, inplace=False
    )
    np.testing.assert_array_equal(synthetic_frame, frame_copy)
    assert not np.array_equal(out_frame, synthetic_frame)


@pytest.mark.unit
def test_distortion_camera_matches_cv2_project_points() -> None:
    """A distortion-enabled PinholeCamera uses the distortion path; compare to cv2.projectPoints within 0.5px."""
    w, h = 1920, 1080
    k = intrinsics_from_fov(w, h, 60.0)
    pos = np.array([0.0, 0.0, 4.0])
    tgt = np.array([0.0, 0.0, 0.0])
    r = look_at(pos, tgt, up=np.array([0.0, 1.0, 0.0]))
    dist = np.array([-0.1, 0.05, 0.001, -0.001, 0.0])
    cam = PinholeCamera(
        camera_id="dist_cam",
        matrix=k,
        rotation_world_from_camera=r,
        translation_world_from_camera_m=pos,
        image_size_px=(w, h),
        distortion=dist,
    )
    projector = PinholeProjector(cam)
    points_world = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [-0.4, -0.2, 0.5]], dtype=float
    )
    pixels, valid = projector.project(points_world)
    assert valid.all()

    # Compare against cv2.projectPoints
    r_c2w = r
    r_w2c = r_c2w.T
    rvec, _ = cv2.Rodrigues(r_w2c)
    tvec = -r_w2c @ pos
    expected_px, _ = cv2.projectPoints(points_world, rvec, tvec, k, dist)
    expected_px = expected_px.reshape(-1, 2)

    np.testing.assert_allclose(pixels, expected_px, atol=0.5)


@pytest.mark.unit
def test_legend_box_text_appears(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """The legend box appears in the bottom-left and modifies pixels in that region."""
    projector = PinholeProjector(synthetic_camera)
    glyphs = _make_glyph_set(unavailable=("motor_torque",))
    frame_orig = synthetic_frame.copy()
    out_frame, _ = draw_glyphs_on_frame(
        synthetic_frame,
        glyphs,
        projector,
        qualification="unqualified camera hypothesis",
    )
    # Bottom-left quadrant: y in [700, 1080], x in [0, 600]
    bl_orig = frame_orig[700:1080, 0:600]
    bl_out = out_frame[700:1080, 0:600]
    assert not np.array_equal(bl_orig, bl_out)


@pytest.mark.unit
def test_hypothesis_projector() -> None:
    """HypothesisProjector wraps CameraProjection catching behind-camera errors."""
    k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    r = np.eye(3)
    t = np.array([0.0, 0.0, 2.0])  # camera at origin looking along +z, shifted by +2
    proj = _MockCameraProjection(intrinsics=k, rotation=r, translation=t)
    projector = HypothesisProjector(proj)

    # Point in front: (0, 0, 1) -> z_c = 3.0 > 0
    # Point behind: (0, 0, -5) -> z_c = -3.0 < 0
    pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -5.0]], dtype=float)
    pixels, valid = projector.project(pts)
    assert valid[0]
    assert not valid[1]
    assert np.isnan(pixels[1, 0])
    assert np.isclose(pixels[0, 0], 320.0)
    assert np.isclose(pixels[0, 1], 240.0)


@pytest.mark.unit
def test_receipt_serialization() -> None:
    """VideoGlyphReceipt is frozen and serializable via to_dict."""
    receipt = VideoGlyphReceipt(
        drawn=3,
        skipped_behind_camera=1,
        skipped_out_of_frame=2,
        unavailable_labels=("load_cell",),
    )
    d = receipt.to_dict()
    assert d == {
        "drawn": 3,
        "skipped_behind_camera": 1,
        "skipped_out_of_frame": 2,
        "unavailable_labels": ["load_cell"],
    }
    # Test 2-tuple unpacking
    frame, rec = receipt
    assert rec is receipt


@pytest.mark.unit
def test_torque_arc_drawing(
    synthetic_camera: PinholeCamera, synthetic_frame: np.ndarray
) -> None:
    """A torque arc with polyline and head is drawn onto the frame."""
    projector = PinholeProjector(synthetic_camera)
    thetas = np.linspace(0, 1.5 * np.pi, 16)
    poly = tuple(
        (float(0.2 * np.cos(th)), float(0.2 * np.sin(th)), 0.0) for th in thetas
    )
    tip = (
        float(0.2 * np.cos(thetas[-1] + 0.1)),
        float(0.2 * np.sin(thetas[-1] + 0.1)),
        0.0,
    )
    base = poly[-1]
    arc = TorqueArcGlyph(
        label="test_arc",
        kind="reaction",
        center_m=(0.0, 0.0, 0.0),
        axis_unit=(0.0, 0.0, 1.0),
        radius_m=0.2,
        polyline_m=poly,
        head_tip_m=tip,
        head_base_m=base,
        rgba=(0.0, 1.0, 1.0, 1.0),  # Cyan
        magnitude=25.0,
        units="N·m",
        clamped=False,
    )
    glyphs = _make_glyph_set(arcs=[arc])
    out_frame, receipt = draw_glyphs_on_frame(synthetic_frame, glyphs, projector)
    assert receipt.drawn == 1
    assert receipt.skipped_behind_camera == 0
    assert receipt.skipped_out_of_frame == 0
