"""Raster exports exercise native FK and a real readable two-frame codec fixture."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def video_case(native_fit_case, tmp_path):
    cv2 = pytest.importorskip("cv2")
    library, source, payload = native_fit_case
    rows = []
    archive_path = tmp_path / "video-capture.zip"
    with ZipFile(archive_path, "w") as archive:
        for i in range(2):
            image = np.full((240, 320, 3), 90 + i * 20, dtype=np.uint8)
            ok, png = cv2.imencode(".png", image)
            assert ok
            frame = {
                "schema_version": "shadow-tracker/frame/1.1.0",
                "asset_id": "source-" + "a" * 64,
                "shot_id": "video-shot",
                "swing_id": "practice",
                "camera_id": "source-camera",
                "frame_id": f"video-{i}",
                "frame_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "pts_ticks": i,
                "timebase_numerator": 1,
                "timebase_denominator": 30,
                "physical_time_s": None,
                "physical_time_reason": "Synthetic source physical clock unknown",
                "timing_mode": "container_pts",
                "is_timing_exact": True,
                "clock_evidence": "Synthetic exact fixture PTS",
                "decoder_name": "synthetic",
                "decoder_version": "1",
                "pixel_format": "bgr24",
            }
            rows.append(
                {
                    "frame": frame,
                    "image": f"frame-{i}.png",
                    "observation": {
                        "status": "detected",
                        "landmarks": {
                            "origin": {"x": 0.5, "y": 0.5, "visibility": None}
                        },
                    },
                }
            )
            archive.writestr(f"frame-{i}.png", png.tobytes())
        archive.writestr(
            "receipt.json",
            json.dumps(
                {
                    "frame_count": 2,
                    "source": {"width_px": 320, "height_px": 240, "sha256": "a" * 64},
                }
            ),
        )
        archive.writestr(
            "observations.jsonl", "\n".join(json.dumps(row) for row in rows)
        )
    capture = library._save_asset(
        "video-capture",
        "practice",
        archive_path,
        "image_capture",
        {"schema": "necromatcher/image-capture/1", "source_sha256": "a" * 64},
    )
    payload.update(
        capture_id=capture.dataset_id,
        capture_hash=capture.metadata["hash"],
        frame_indices=[0, 1],
        frames=[row["frame"] for row in rows],
    )
    source.write_text(json.dumps(payload))
    library.add_fit("video-fit", "practice", source)
    return library, capture, payload


def test_video_and_selected_pngs_preserve_bindings_and_source_clock(
    video_case, tmp_path
):
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, capture, payload = video_case
    before = Path(capture.path).read_bytes()
    output = tmp_path / "export"
    manifest = export_fit_video(library, "video-fit", output, selected_frames=(0, 1))
    assert manifest["schema"] == "necromatcher/source-overlay-video/1"
    assert manifest["capture_hash"] == payload["capture_hash"]
    assert manifest["model_hash"] == payload["model_hash"]
    assert manifest["physical_time_qualified"] is False
    assert manifest["source_frame_rate"] == {"numerator": 30, "denominator": 1}
    assert len(manifest["frames"]) == 2
    assert manifest["frames"][0]["matched_marker_count"] == 1
    assert (output / "frame-000000.png").exists()
    assert Path(capture.path).read_bytes() == before
    assert json.loads((output / "manifest.json").read_text()) == manifest
    import cv2

    reader = cv2.VideoCapture(str(output / "overlay.mp4"))
    try:
        decoded = []
        while True:
            ok, image = reader.read()
            if not ok:
                break
            decoded.append(image)
        assert len(decoded) == 2
        assert decoded[0].shape == (240, 320, 3)
    finally:
        reader.release()


def test_video_refuses_existing_output_and_changed_parent(video_case, tmp_path):
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, capture, _ = video_case
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        export_fit_video(library, "video-fit", destination)
    Path(capture.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_fit_video(library, "video-fit", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("indices,frames", [([0, 2], [0, 1]), ([0, 1, 2], [0, 1, 3])])
def test_clock_conversion_rejects_noncontiguous_or_nonuniform(indices, frames):
    from src.shared.python.workspace.necromatcher_video import source_frame_rate

    identities = [
        {"pts_ticks": tick, "timebase_numerator": 1, "timebase_denominator": 30}
        for tick in frames
    ]
    with pytest.raises(ValueError, match="uniform|contiguous"):
        source_frame_rate(indices, identities)


def test_joint_wireframe_uses_declared_native_body_origins(video_case, tmp_path):
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, _, payload = video_case
    manifest = export_fit_video(library, "video-fit", tmp_path / "rig")
    edges = manifest["rigid_body_segments"]
    declared = {
        (joint["parent"], joint["child"])
        for joint in payload["provenance"]["native_definition"]["joints"]
    }
    assert edges
    assert {(edge["a"], edge["b"]) for edge in edges} <= declared
    assert "tibia_l" in {name for edge in edges for name in (edge["a"], edge["b"])}
    assert "LAnkleOut" in manifest["missing_anatomical_offsets"]
    assert "LAnkleOut" not in manifest["anatomical_marker_names"]


def test_invalid_native_projection_does_not_publish_partial_output(
    video_case, tmp_path, monkeypatch
):
    from src.shared.python.workspace.necromatcher_video import export_fit_video
    from src.shared.python.motion_matching.historical_fit import CameraProjection

    monkeypatch.setattr(
        CameraProjection,
        "project",
        lambda self, points: np.full((len(points), 2), np.nan),
    )
    with pytest.raises(ValueError, match="finite"):
        export_fit_video(video_case[0], "video-fit", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    assert not list(tmp_path.glob("necromatcher-video-*"))


def test_missing_observation_cannot_produce_a_residual():
    from src.shared.python.workspace.necromatcher_video import _observations

    assert (
        _observations(
            {
                "image_width": 10,
                "image_height": 10,
                "observation": {
                    "status": "missing",
                    "landmarks": {"left_wrist": {"x": 0.2, "y": 0.3}},
                },
            }
        )
        == {}
    )


def test_unreadable_selected_png_cannot_publish(video_case, tmp_path, monkeypatch):
    import cv2
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    def broken_png(path, image):
        Path(path).write_bytes(b"invalid PNG")
        return True

    monkeypatch.setattr(cv2, "imwrite", broken_png)
    with pytest.raises(ValueError, match="PNG"):
        export_fit_video(
            video_case[0], "video-fit", tmp_path / "broken", selected_frames=(0,)
        )
    assert not (tmp_path / "broken").exists()
