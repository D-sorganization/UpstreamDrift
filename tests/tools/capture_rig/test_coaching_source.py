"""Capture and model analysis share a media contract rather than fake bundles."""

from pathlib import Path

import pytest

from src.motion_capture.coaching import Drawing
from src.motion_capture.coaching.storage import load_layer
from src.tools.capture_rig.coaching_source import CaptureCoachingSource
from tests.motion_capture.rig.test_ingest import _bundle

pytestmark = pytest.mark.unit


def test_capture_source_preserves_existing_drawing_and_export_paths(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    source = CaptureCoachingSource(root, "a")
    layer = source.drawings.with_shape(Drawing(kind="line", start=(2, 2), end=(30, 20)))
    assert source.time_at(0) == 0
    source.save(layer)
    assert load_layer(root, "a", 64, 48, 6) == layer
    source.still(layer, 0, tmp_path / "frame.png")
    assert (tmp_path / "frame.json").is_file()
    job = source.export_job(layer)
    with pytest.raises(InterruptedError):
        job(tmp_path / "cancelled.avi", lambda: True, lambda done, total: None)
    assert not (tmp_path / "cancelled.avi").exists()
    source.reader.close()


def test_coaching_dialog_accepts_a_media_source(tmp_path: Path) -> None:
    from src.tools.capture_rig.coaching_dialog import CoachingDialog
    from tests.tools.capture_rig.test_pane_layout import _app

    _app()
    root = _bundle(tmp_path)
    source = CaptureCoachingSource(root, "a")
    dialog = CoachingDialog(root, "a", media=source)
    assert dialog.reader is source.reader
    dialog.tool.setCurrentText("Line")
    dialog.add_center()
    assert dialog.save()
    dialog.close()
