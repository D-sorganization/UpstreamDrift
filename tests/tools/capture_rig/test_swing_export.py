"""Saved swing exports preserve source selection, pixels and originals (#9860)."""

from pathlib import Path
import json

import pytest

from src.motion_capture.rig.edits import CropRect, SessionEdits, ViewEdit, save_edits
from src.tools.capture_rig.player import VideoReader
from src.tools.capture_rig.swing_export import export_swing
from tests.motion_capture.rig.test_ingest import _bundle

pytestmark = pytest.mark.unit


def test_export_saved_selection_and_odd_crop_with_provenance(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    from src.motion_capture.rig.bundle import load_bundle

    entry = load_bundle(root)[1].recordings[0]
    source = root / entry.file
    original = source.read_bytes()
    edit = ViewEdit(first=1, last=3, crop=CropRect(x=8, y=6, width=31, height=23))
    save_edits(root, SessionEdits(views={entry.view: edit}))
    output = tmp_path / "swing.avi"
    progress: list[tuple[int, int]] = []
    result = export_swing(
        root, entry.view, output, progress=lambda a, b: progress.append((a, b))
    )
    assert result["frames"] == 3
    assert result["edit"] == edit.model_dump(mode="json")
    assert result["padding"] == {"right": 1, "bottom": 1}
    assert len(result["source_sha256"]) == 64
    assert progress[-1] == (3, 3)
    with VideoReader(output) as reader:
        assert (reader.width, reader.height, reader.frame_count) == (32, 24, 3)
        assert reader.read(0) is not None
    assert source.read_bytes() == original
    assert (
        json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["first"]
        == 1
    )


def test_export_cancellation_and_existing_output_leave_files_intact(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    from src.motion_capture.rig.bundle import load_bundle

    view = load_bundle(root)[1].recordings[0].view
    output = tmp_path / "swing.avi"
    with pytest.raises(InterruptedError):
        export_swing(root, view, output, cancelled=lambda: True)
    assert not output.exists() and not output.with_suffix(".json").exists()
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        export_swing(root, view, output)
    assert output.read_bytes() == b"existing"


def test_decode_shortfall_does_not_publish_a_partial_swing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path)
    from src.motion_capture.rig.bundle import load_bundle

    view = load_bundle(root)[1].recordings[0].view
    original = VideoReader.read
    monkeypatch.setattr(
        VideoReader,
        "read",
        lambda self, index: None if index == 2 else original(self, index),
    )
    output = tmp_path / "short.avi"
    with pytest.raises(ValueError, match="decode"):
        export_swing(root, view, output)
    assert not output.exists() and not output.with_suffix(".json").exists()
