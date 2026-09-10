"""Model-only analysis persists and exports without creating capture recordings."""

import json
from pathlib import Path

import pytest

from src.motion_capture.coaching import Drawing
from src.tools.capture_rig.model_coaching_source import ModelCoachingSource
from src.tools.capture_rig.model_frame_source import ModelFrameSource
from src.tools.capture_rig.player import VideoReader
from tests.motion_capture.test_reference_registration import sample_motion

pytestmark = pytest.mark.unit


def test_model_analysis_roundtrip_and_exports_without_recordings(
    tmp_path: Path,
) -> None:
    recipe = ModelFrameSource.from_motion(
        sample_motion(), fps=5, size=(160, 120)
    ).recipe
    source = ModelCoachingSource(recipe, tmp_path / "analysis")
    layer = source.drawings.with_shape(
        Drawing(kind="line", start=(10, 10), end=(100, 80))
    )
    source.save(layer)
    reopened = ModelCoachingSource(recipe, tmp_path / "analysis")
    assert reopened.drawings == layer
    assert reopened.time_at(2) == 0.4
    source.still(layer, 2, tmp_path / "model.png")
    sidecar = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert sidecar["source_kind"] == "virtual_model"
    assert sidecar["recipe"]["asset"]["id"] == recipe.asset.id
    job = source.export_job(layer)
    job(tmp_path / "motion.avi", lambda: False, lambda done, total: None)
    with VideoReader(tmp_path / "motion.avi") as video:
        assert video.frame_count == source.reader.frame_count
        assert video.read(0) is not None
    assert not list(tmp_path.rglob("recordings.json"))
    with pytest.raises(InterruptedError):
        job(tmp_path / "cancelled.avi", lambda: True, lambda done, total: None)
    assert not (tmp_path / "cancelled.avi").exists()
    source.reader.close()
    reopened.reader.close()


def test_model_export_verifies_encoded_frames_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.tools.capture_rig.model_coaching_source as implementation

    recipe = ModelFrameSource.from_motion(
        sample_motion(), fps=5, size=(160, 120)
    ).recipe
    source = ModelCoachingSource(recipe, tmp_path / "analysis")

    def broken_writer(
        reader: ModelFrameSource, clip: object, out: Path, **options: object
    ) -> dict[str, float | int]:
        out.write_bytes(b"invalid video")
        return {"frames": reader.frame_count, "fps": reader.fps}

    monkeypatch.setattr(implementation, "write_frame_clip", broken_writer)
    with pytest.raises(ValueError):
        source.export_job(source.drawings)(
            tmp_path / "broken.avi", lambda: False, lambda done, total: None
        )
    assert not (tmp_path / "broken.avi").exists()
    assert not (tmp_path / "broken.json").exists()
    source.reader.close()
