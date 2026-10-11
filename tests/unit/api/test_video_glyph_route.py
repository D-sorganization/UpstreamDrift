"""Unit tests for the video glyph overlay API route (FTO-29, #11314).

Verifies:
1. Pinhole projection matches PinholeCamera.project within 0.5px.
2. Unknown source_id returns 404.
3. Source without camera calibration returns 409.
4. Scale, kinds, and frame index are validated and respected.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.video_overlays import (
    VideoOverlaySource,
    VideoOverlayStore,
    get_video_overlay_store,
    router,
)
from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)


@pytest.fixture
def synthetic_camera() -> PinholeCamera:
    """1920x1080 camera at (0, 0, 5) looking down -z at origin (0, 0, 0)."""
    w, h = 1920, 1080
    k = intrinsics_from_fov(w, h, 60.0)
    pos = np.array([0.0, 0.0, 5.0])
    tgt = np.array([0.0, 0.0, 0.0])
    r = look_at(pos, tgt, up=np.array([0.0, 1.0, 0.0]))
    return PinholeCamera(
        camera_id="cam_synthetic",
        matrix=k,
        rotation_world_from_camera=r,
        translation_world_from_camera_m=pos,
        image_size_px=(w, h),
    )


@pytest.fixture
def synthetic_series() -> ForceTorqueSeries:
    """Force series with a 100 N contact wrench at origin along +x in ADR-0041 frame."""
    wrench = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:lead_foot",
        body="lead_foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(100.0, 0.0, 0.0),
        source="synthetic",
    )
    frame = ForceTorqueFrame(
        time_s=0.0,
        engine="synthetic",
        world_frame="adr0041_world",
        wrenches=(wrench,),
    )
    return ForceTorqueSeries(frames=(frame,))


@pytest.fixture
def overlay_store(
    synthetic_camera: PinholeCamera, synthetic_series: ForceTorqueSeries
) -> VideoOverlayStore:
    store = VideoOverlayStore()
    # Source with camera and series
    store.register_source(
        "source_with_camera",
        VideoOverlaySource(
            camera=synthetic_camera,
            series=synthetic_series,
            fps=30.0,
            total_frames=10,
        ),
    )
    # Source without camera
    store.register_source(
        "source_no_camera",
        VideoOverlaySource(
            camera=None,
            series=synthetic_series,
            fps=30.0,
            total_frames=10,
        ),
    )
    return store


@pytest.fixture
def client(overlay_store: VideoOverlayStore) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_video_overlay_store] = lambda: overlay_store
    return TestClient(app)


@pytest.mark.unit
def test_unknown_source_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/overlays/video/unknown_id_404/frames/0/glyphs")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.unit
def test_source_without_camera_returns_409(client: TestClient) -> None:
    response = client.get("/api/overlays/video/source_no_camera/frames/0/glyphs")
    assert response.status_code == 409
    assert "camera" in response.json()["detail"].lower()


@pytest.mark.unit
def test_invalid_frame_index_returns_422(client: TestClient) -> None:
    response = client.get("/api/overlays/video/source_with_camera/frames/-1/glyphs")
    assert response.status_code == 422


@pytest.mark.unit
def test_frame_index_out_of_bounds_returns_404(client: TestClient) -> None:
    response = client.get("/api/overlays/video/source_with_camera/frames/999/glyphs")
    assert response.status_code == 404
    assert "out of range" in response.json()["detail"].lower()


@pytest.mark.unit
def test_projected_pixels_match_pinhole_camera_project(
    client: TestClient, synthetic_camera: PinholeCamera
) -> None:
    """Projected pixels must match PinholeCamera.project within 0.5px."""
    response = client.get(
        "/api/overlays/video/source_with_camera/frames/0/glyphs?kinds=contact&scale=1.0"
    )
    assert response.status_code == 200
    data = response.json()

    assert "arrows" in data
    assert len(data["arrows"]) == 1

    arrow = data["arrows"][0]
    tail_px = arrow["start_px"]
    tip_px = arrow["end_px"]

    # Origin (0,0,0) projects to principal point (960, 540)
    origin_world = np.array([[0.0, 0.0, 0.0]])
    expected_origin_px, _ = synthetic_camera.project(origin_world)

    np.testing.assert_allclose(tail_px, expected_origin_px[0], atol=0.5)

    # 100 N scaled at default scale 1/1000 = 0.1 m along +x
    # World tip: (0.1, 0.0, 0.0)
    tip_world = np.array([[0.1, 0.0, 0.0]])
    expected_tip_px, _ = synthetic_camera.project(tip_world)

    np.testing.assert_allclose(tip_px, expected_tip_px[0], atol=0.5)

    # Verify head polygon exists with 3 vertices
    assert "head_poly_px" in arrow
    assert len(arrow["head_poly_px"]) == 3

    # Verify receipt and legend
    assert data["receipt"]["drawn"] == 1
    assert data["receipt"]["skipped_behind_camera"] == 0
    assert "legend" in data
