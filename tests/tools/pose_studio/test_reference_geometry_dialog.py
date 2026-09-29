"""Native reference editing reuses the same controls and portable scene document."""

from pathlib import Path
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PyQt6")

from src.tools.pose_studio.reference_geometry import ReferenceGeometryDialog
from src.tools.pose_studio.widgets.view_3d import View3D
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_native_editor_applies_and_roundtrips_shared_geometry(tmp_path: Path):
    _app()
    view = View3D()
    dialog = ReferenceGeometryDialog(view)
    baseline = len(view._ax.collections)
    dialog.controls.add_plane()
    assert len(view._ax.collections) == baseline + 1
    path = tmp_path / "references.json"
    dialog.controls.save(path)
    dialog.controls.delete_selected()
    assert len(view._ax.collections) == baseline
    dialog.controls.load(path)
    assert len(view._ax.collections) == baseline + 1
    dialog.close()
    assert len(view._ax.collections) == baseline + 1
    view.close()


def test_pose_studio_entry_reopens_existing_reference_editor():
    from src.tools.pose_studio.gui import MainWidget

    _app()
    widget = MainWidget()
    widget.btn_references.click()
    dialog = widget.references_dialog
    assert isinstance(dialog, ReferenceGeometryDialog)
    dialog.controls.add_point()
    dialog.close()
    widget.btn_references.click()
    assert widget.references_dialog is dialog
    assert len(dialog.controls.document.points) == 1
    dialog.close()
    widget.close()
