"""Reference comparison launches the common drawing editor and reloads edits."""

import json
from pathlib import Path

import pytest

from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.coaching_dialog import CoachingDialog
from src.tools.capture_rig.model_frame_source import ModelFrameSource
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_comparison_launches_shared_editor_and_reloads_saved_drawings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    root = _bundle(tmp_path)
    motion = synthetic_motion()
    virtual = ModelFrameSource.from_motion(motion, size=(64, 48))
    snapshot = virtual.recipe.camera.model_copy(update={"camera_id": "a"})
    virtual.close()
    camera_path = root / "reconstruct" / "reconstruction.json"
    camera_path.parent.mkdir()
    camera_path.write_text(json.dumps({"cameras": [snapshot.record().to_dict()]}))
    library = ReferenceLibrary(tmp_path / "references")
    library.save(motion)
    opened = []

    def edit(dialog: CoachingDialog) -> int:
        opened.append(dialog)
        dialog.tool.setCurrentText("Line")
        dialog.add_center()
        assert dialog.save()
        dialog.close()
        return 0

    monkeypatch.setattr(CoachingDialog, "exec", edit)
    comparison = ReferenceComparisonDialog(root, "a", library)
    comparison.edit_drawings()
    assert len(opened) == 1
    assert len(comparison._drawings.shapes) == 1
    comparison.close()
