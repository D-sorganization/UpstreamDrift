"""Tests for bidirectional video/model overlay and camera perspective matching (Epic #15660, #15669).

Reuses #11246 (CameraProjection) and #11356 (ShapeOverlayOptions, SurfaceLayer) seams.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection
from src.shared.python.playback.camera_perspective import (
    CameraViewParameters,
    frustum_quad_world_m,
    match_perspective_from_projection,
    projection_from_viewer_perspective,
    unproject_pixel_to_plane,
    unproject_pixel_to_ray,
)
from src.shared.python.playback.bidirectional_overlay import (
    BidirectionalOverlayConfig,
    OverlayMode,
    composite_bidirectional_overlay,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_projection() -> CameraProjection:
    """A standard calibrated pinhole camera facing the origin from (0, -2, 1)."""
    k = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=float
    )
    pos, tgt, up = (
        np.array([0.0, -2.0, 1.0]),
        np.array([0.0, 0.0, 0.8]),
        np.array([0.0, 0.0, 1.0]),
    )
    f = (tgt - pos) / np.linalg.norm(tgt - pos)
    r = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    d = np.cross(f, r)
    r_cw = np.column_stack([r, d, f]).T
    return CameraProjection(intrinsics=k, rotation=r_cw, translation=-r_cw @ pos)


def test_match_perspective_from_projection(sample_projection: CameraProjection) -> None:
    """Matched view recovers position, aspect, field of view, and look-at depth."""
    v = match_perspective_from_projection(sample_projection, (640, 480), 2.0, 0.1, 50.0)
    assert isinstance(v, CameraViewParameters)
    np.testing.assert_allclose(v.position_m, [0.0, -2.0, 1.0], atol=1e-6)
    assert v.aspect_ratio == pytest.approx(640 / 480)
    assert 0.0 < v.fov_horizontal_deg < 180.0 and 0.0 < v.fov_vertical_deg < 180.0
    depth = sample_projection.camera_points(np.array(v.target_m)[None, :])[0, 2]
    assert depth == pytest.approx(2.0, rel=1e-5)


def test_perspective_matching_round_trip(sample_projection: CameraProjection) -> None:
    """Viewer parameters rebuilt from a projection recover the original camera."""
    v = match_perspective_from_projection(sample_projection, (640, 480), 2.5)
    rec = projection_from_viewer_perspective(
        v.position_m, v.target_m, v.up_vector, (640, 480), v.fov_vertical_deg
    )
    np.testing.assert_allclose(rec.intrinsics, sample_projection.intrinsics, atol=1e-4)
    np.testing.assert_allclose(rec.rotation, sample_projection.rotation, atol=1e-5)
    np.testing.assert_allclose(
        rec.translation, sample_projection.translation, atol=1e-5
    )


def test_unproject_pixel_to_ray_and_plane(sample_projection: CameraProjection) -> None:
    """A principal-point ray projects back, and a hit behind the camera is rejected."""
    orig, ray = unproject_pixel_to_ray(sample_projection, (320.0, 240.0))
    np.testing.assert_allclose(orig, [0.0, -2.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        sample_projection.project((orig + 3.0 * ray)[None, :])[0],
        [320.0, 240.0],
        atol=1e-5,
    )

    pt_3d = unproject_pixel_to_plane(
        sample_projection, (320.0, 240.0), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]
    )
    assert pt_3d[1] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(
        sample_projection.project(pt_3d[None, :])[0], [320.0, 240.0], atol=1e-4
    )

    with pytest.raises(ValueError, match="behind"):
        unproject_pixel_to_plane(
            sample_projection, (320.0, 240.0), [0.0, -1.0, 0.0], [0.0, -5.0, 0.0]
        )


def test_frustum_quad_world_m(sample_projection: CameraProjection) -> None:
    """Frame corners at a fixed depth project back onto the image rectangle."""
    quad = frustum_quad_world_m(sample_projection, (640, 480), 2.0)
    assert quad.shape == (4, 3)
    np.testing.assert_allclose(
        sample_projection.project(quad),
        [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]],
        atol=1e-4,
    )


def test_composite_model_on_video() -> None:
    """Model-on-video compositing honors zero, full, and half model opacity."""
    vid = np.full((50, 50, 3), 100, dtype=np.uint8)
    mod = np.zeros((50, 50, 4), dtype=np.uint8)
    mod[:25, :25, 0], mod[:25, :25, 3] = 200, 255

    cfg_zero = BidirectionalOverlayConfig(OverlayMode.MODEL_ON_VIDEO, model_opacity=0.0)
    np.testing.assert_array_equal(
        composite_bidirectional_overlay(vid, mod, cfg_zero), vid
    )

    cfg_full = BidirectionalOverlayConfig(OverlayMode.MODEL_ON_VIDEO, model_opacity=1.0)
    res_full = composite_bidirectional_overlay(vid, mod, cfg_full)
    assert np.all(res_full[:25, :25, 0] == 200) and np.all(res_full[25:, 25:] == 100)

    cfg_half = BidirectionalOverlayConfig(OverlayMode.MODEL_ON_VIDEO, model_opacity=0.5)
    res_half = composite_bidirectional_overlay(vid, mod, cfg_half)
    assert res_half[0, 0, 0] == pytest.approx(150, abs=1)


def test_composite_video_on_model_and_split() -> None:
    """Video-on-model blends by opacity, and a vertical split keeps each side."""
    vid = np.full((40, 60, 3), 200, dtype=np.uint8)
    mod = np.full((40, 60, 3), 50, dtype=np.uint8)

    cfg_vom = BidirectionalOverlayConfig(OverlayMode.VIDEO_ON_MODEL, video_opacity=0.5)
    assert composite_bidirectional_overlay(vid, mod, cfg_vom)[0, 0, 0] == pytest.approx(
        125, abs=1
    )

    cfg_split = BidirectionalOverlayConfig(
        OverlayMode.SPLIT_SCREEN, split_ratio=0.5, split_orientation="vertical"
    )
    res_split = composite_bidirectional_overlay(vid, mod, cfg_split)
    assert np.all(res_split[:, :29] == 200) and np.all(res_split[:, 31:] == 50)

    with pytest.raises(ValueError, match="[Dd]imensions|[Ss]hape"):
        composite_bidirectional_overlay(
            vid, np.full((30, 60, 3), 50, dtype=np.uint8), cfg_vom
        )


def test_shape_overlay_options_interop() -> None:
    """Shape overlay opacity becomes a model-on-video config at the same opacity."""
    cfg = BidirectionalOverlayConfig.from_shape_options(ShapeOverlayOptions(0.75))
    assert (
        cfg.mode == OverlayMode.MODEL_ON_VIDEO
        and cfg.model_opacity == pytest.approx(0.75)
    )
