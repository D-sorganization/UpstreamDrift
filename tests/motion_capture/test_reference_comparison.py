"""Unit tests for reference comparison models, session storage and sidecar exports (#9866)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.motion_capture.reference.comparison import (
    COMPARISON_EXPORT_SCHEMA,
    COMPARISON_SESSION_SCHEMA,
    ComparisonExportSidecarSpec,
    ComparisonLayer,
    ComparisonSession,
    build_comparison_sidecar,
    comparison_session_path,
    load_comparison_session,
    save_comparison_session,
)
from src.motion_capture.reference.model import (
    ReferenceMotion,
    ReferenceSource,
    ReferenceVideo,
)
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    ReferenceTransform,
    TimeMapping,
)
from src.motion_capture.rig.edits import CropRect

pytestmark = pytest.mark.unit


def dummy_motion() -> ReferenceMotion:
    return ReferenceMotion.model_validate(
        {
            "id": str(uuid4()),
            "title": "Pro Model Benchmark",
            "source": ReferenceSource(
                path="mock_pro.c3d", sha256="a" * 64, format="c3d"
            ),
            "source_units": "m",
            "source_axes": ("+X", "+Y", "+Z"),
            "source_names": ("pelvis", "lead_wrist"),
            "joint_names": ("pelvis", "wrist"),
            "edges": ((0, 1),),
            "time_s": (0.0, 0.5, 1.0),
            "points_m": (
                ((0.0, 0.0, 1.0), (0.2, 0.0, 1.0)),
                ((0.0, 0.0, 1.0), (-0.2, 0.0, 1.6)),
                ((0.0, 0.0, 1.0), (0.1, 0.0, 1.2)),
            ),
        }
    )


def dummy_video(tmp_path: Path) -> ReferenceVideo:
    src_file = tmp_path / "expert.mp4"
    src_file.write_bytes(b"dummy video content for hashing")
    return ReferenceVideo.model_validate(
        {
            "id": str(uuid4()),
            "title": "Expert 2D View",
            "source": ReferenceSource(
                path=str(src_file),
                sha256="b" * 64,
                format="video",
            ),
            "width": 1920,
            "height": 1080,
            "frames": 60,
            "fps": 30.0,
        }
    )


def test_comparison_layer_defaults_and_validation() -> None:
    layer = ComparisonLayer()
    assert layer.colour == "#00dcff"
    assert layer.opacity == 1.0
    assert layer.visible is True
    assert layer.line_width == 2

    custom = ComparisonLayer(colour="#ff5500", opacity=0.75, line_width=4)
    assert custom.colour == "#ff5500"
    assert custom.opacity == 0.75
    assert custom.line_width == 4

    with pytest.raises(ValueError):
        ComparisonLayer(colour="not-a-colour")
    with pytest.raises(ValueError):
        ComparisonLayer(opacity=1.5)


def test_comparison_session_serialization_and_roundtrip(tmp_path: Path) -> None:
    ref = dummy_motion()
    reg = ReferenceRegistration(
        reference_id=ref.id,
        calibration_id="calib_session_1",
        transform=ReferenceTransform(scale=1.05),
        time_mapping=TimeMapping(offset_s=0.25),
        is_calibrated=True,
    )
    session = ComparisonSession(
        session_root=str(tmp_path),
        view="down_the_line",
        reference_id=ref.id,
        reference_kind="motion",
        registration=reg,
        layer=ComparisonLayer(colour="#00ff00", opacity=0.8),
        notes="Testing reference comparison session persistence",
    )
    assert session.schema_version == COMPARISON_SESSION_SCHEMA

    saved_path = save_comparison_session(session, tmp_path)
    assert saved_path == comparison_session_path(tmp_path, "down_the_line", ref.id)
    assert saved_path.is_file()

    loaded = load_comparison_session(saved_path)
    assert loaded.id == session.id
    assert loaded.view == "down_the_line"
    assert loaded.reference_id == ref.id
    assert loaded.registration is not None
    assert loaded.registration.transform.scale == 1.05
    assert loaded.layer.colour == "#00ff00"
    assert loaded.notes == "Testing reference comparison session persistence"


def test_build_comparison_sidecar_3d_calibrated(tmp_path: Path) -> None:
    src_video = tmp_path / "player_take.mp4"
    src_video.write_bytes(b"player video stream bytes")
    out_video = tmp_path / "comparison_export.mp4"

    ref = dummy_motion()
    reg = ReferenceRegistration(
        reference_id=ref.id,
        calibration_id="calib_1",
        transform=ReferenceTransform(),
        time_mapping=TimeMapping(offset_s=-0.1),
        is_calibrated=True,
    )
    sidecar = build_comparison_sidecar(
        ComparisonExportSidecarSpec(
            video_out=out_video,
            source_media=src_video,
            reference_asset=ref,
            view="face_on",
            fps=30.0,
            frame_count=3,
            output_frame_times=[0.0, 0.033, 0.066],
            registration=reg,
            crop=CropRect(x=10, y=10, width=640, height=480),
            layer=ComparisonLayer(colour="#ff00ff", opacity=0.9),
        )
    )

    assert sidecar["schema_version"] == COMPARISON_EXPORT_SCHEMA
    assert sidecar["video_file"] == "comparison_export.mp4"
    assert sidecar["source"]["view"] == "face_on"
    assert sidecar["reference"]["id"] == ref.id
    assert sidecar["reference"]["alignment_status"] == "calibrated_3d_projection"
    assert sidecar["reference"]["is_3d"] is True
    assert sidecar["reference"]["is_calibrated"] is True
    assert sidecar["reference"]["missing_alignment_evidence"] is False
    assert sidecar["playback"]["frame_count"] == 3
    assert sidecar["crop"]["width"] == 640


def test_build_comparison_sidecar_2d_video_no_3d_claim(tmp_path: Path) -> None:
    src_video = tmp_path / "player_take.mp4"
    src_video.write_bytes(b"player video stream bytes 2")
    out_video = tmp_path / "comparison_export_2d.mp4"

    ref_vid = dummy_video(tmp_path)
    sidecar = build_comparison_sidecar(
        ComparisonExportSidecarSpec(
            video_out=out_video,
            source_media=src_video,
            reference_asset=ref_vid,
            view="face_on",
            fps=30.0,
            frame_count=5,
            output_frame_times=[0.0, 0.033, 0.066, 0.1, 0.133],
            time_mapping=TimeMapping(offset_s=0.5),
        )
    )

    assert sidecar["schema_version"] == COMPARISON_EXPORT_SCHEMA
    assert (
        sidecar["reference"]["alignment_status"] == "manual_2d_homography_no_3d_claim"
    )
    assert sidecar["reference"]["is_3d"] is False
    assert sidecar["reference"]["missing_alignment_evidence"] is True
