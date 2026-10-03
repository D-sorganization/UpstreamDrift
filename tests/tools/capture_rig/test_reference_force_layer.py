"""Tests for force/torque arrow layer in reference comparison (FTO-25, #11310)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.motion_capture.reference.comparison import (
    ComparisonLayer,
    ForceLayer,
)
from src.motion_capture.reference.model import (
    ReferenceMotion,
    ReferenceSource,
)
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
)
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.series import ForceTorqueSeries
from src.tools.capture_rig.reference_export import (
    ComparisonVideoExportOptions,
    export_comparison_video,
)
from src.tools.capture_rig.reference_rendering import (
    ComparisonRenderContext,
    ComparisonRenderer,
)
from tests.motion_capture.rig.test_ingest import _bundle

pytestmark = [pytest.mark.unit]

SIZE = (640, 480)


def _synthetic_camera(
    name: str = "cam_test", size: tuple[int, int] = SIZE
) -> PinholeCamera:
    k = intrinsics_from_fov(size[0], size[1], 60.0)
    position = np.array([0.0, 1.0, 3.0])
    return PinholeCamera(
        name, k, look_at(position, np.array([0.0, 1.0, 0.0])), position, size
    )


def _synthetic_motion_asset() -> ReferenceMotion:
    return ReferenceMotion.model_validate(
        {
            "id": str(uuid4()),
            "title": "Synthetic Motion",
            "source": ReferenceSource(
                path="synthetic.c3d", sha256="0" * 64, format="c3d"
            ),
            "source_units": "m",
            "source_axes": ("+X", "+Y", "+Z"),
            "source_names": ("origin", "top"),
            "joint_names": ("origin", "top"),
            "edges": ((0, 1),),
            "time_s": (0.0, 0.05, 0.1, 0.15, 0.2),
            "points_m": (
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.5)),
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.5)),
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.5)),
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.5)),
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.5)),
            ),
        }
    )


def _synthetic_force_series() -> ForceTorqueSeries:
    frames = []
    for i in range(5):
        w = OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:ground_0",
            body="ground",
            point_m=(0.0, 0.0, 1.0),
            force_n=(0.0, 0.0, 500.0),
            torque_nm=None,
            source="synthetic_source",
        )
        frames.append(
            ForceTorqueFrame(
                time_s=float(i * 0.05),
                engine="synthetic",
                world_frame="adr0041_world",
                wrenches=(w,),
            )
        )
    return ForceTorqueSeries(engine="synthetic", frames=tuple(frames))


def test_reference_force_layer_renders_arrow_at_projected_location() -> None:
    camera = _synthetic_camera()
    gradient = np.tile(np.linspace(30, 200, SIZE[0], dtype=np.uint8), (SIZE[1], 1))
    frame = cv2.merge([gradient, gradient, gradient])

    asset = _synthetic_motion_asset()
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="manual")
    series = _synthetic_force_series()

    layer = ForceLayer(series=series, opacity=1.0)
    ctx = ComparisonRenderContext(
        view="cam_test",
        asset=asset,
        registration=reg,
        layer=layer,
    )

    with ComparisonRenderer(asset) as renderer:
        out = renderer.overlay(frame.copy(), 0.0, ctx, camera)
        assert renderer.last_receipt is not None
        assert renderer.last_receipt.drawn >= 1

        # Placed world point is at (0, 1, 0)
        px, visible = camera.project(np.array([[0.0, 1.0, 0.0]]))
        assert visible[0]
        u, v = int(round(px[0, 0])), int(round(px[0, 1]))
        assert 0 <= u < SIZE[0] and 0 <= v < SIZE[1]

        # Non-gradient pixels appear where arrow was drawn
        assert not np.array_equal(out, frame)
        patch_out = out[
            max(0, v - 30) : min(SIZE[1], v + 30), max(0, u - 30) : min(SIZE[0], u + 30)
        ]
        patch_frame = frame[
            max(0, v - 30) : min(SIZE[1], v + 30), max(0, u - 30) : min(SIZE[0], u + 30)
        ]
        assert not np.array_equal(patch_out, patch_frame)


def test_reference_force_layer_opacity_zero_preserves_pixels() -> None:
    camera = _synthetic_camera()
    gradient = np.tile(np.linspace(30, 200, SIZE[0], dtype=np.uint8), (SIZE[1], 1))
    frame = cv2.merge([gradient, gradient, gradient])

    asset = _synthetic_motion_asset()
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="manual")
    series = _synthetic_force_series()

    layer = ForceLayer(series=series, opacity=0.0)
    ctx = ComparisonRenderContext(
        view="cam_test",
        asset=asset,
        registration=reg,
        layer=layer,
    )

    with ComparisonRenderer(asset) as renderer:
        out = renderer.overlay(frame.copy(), 0.0, ctx, camera)
        np.testing.assert_array_equal(out, frame)


def test_reference_force_export_writes_sidecar_with_receipts(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    asset = _synthetic_motion_asset()
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="manual")
    series = _synthetic_force_series()
    camera = _synthetic_camera(name="a", size=(64, 48))

    layer = ForceLayer(series=series, opacity=1.0)
    out_video = tmp_path / "comparison_force.mp4"
    options = ComparisonVideoExportOptions(camera=camera)

    metadata = export_comparison_video(root, "a", asset, reg, layer, out_video, options)
    assert out_video.is_file()

    sidecar_path = out_video.with_suffix(".json")
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    receipts = sidecar.get("glyph_receipts") or sidecar.get("receipts")
    assert receipts is not None
    assert len(receipts) >= 1
    assert any(r["drawn"] >= 1 for r in receipts)
    assert "force_series_hash" in sidecar or "series_source_hash" in sidecar
