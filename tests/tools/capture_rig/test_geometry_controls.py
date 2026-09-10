"""Metric reference edits are validated, reversible and portable across views."""

from pathlib import Path

import pytest

from src.motion_capture.coaching import ReferenceGeometry, ReferencePoint
from src.tools.capture_rig.geometry_controls import GeometryControls
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_plane_edit_validation_undo_and_portable_save(tmp_path: Path) -> None:
    app = _app()
    controls = GeometryControls(ReferenceGeometry(scene_id="scene"))
    controls.show()
    app.processEvents()
    controls.add_plane()
    assert len(controls.document.planes) == 1
    controls.fields["origin_m"][2].setValue(0.5)
    controls.opacity.setValue(0.7)
    controls.apply_selected()
    assert controls.document.planes[0].origin_m[2] == 0.5
    assert controls.document.planes[0].opacity == 0.7
    controls.undo()
    assert controls.document.planes[0].origin_m[2] == 0
    controls.redo()
    path = tmp_path / "geometry.json"
    controls.save(path)
    reopened = GeometryControls(ReferenceGeometry(scene_id="scene"))
    reopened.load(path)
    assert reopened.document == controls.document
    for field in controls.fields["along_m"]:
        field.setValue(0)
    for field in controls.fields["across_m"]:
        field.setValue(0)
    before = controls.document
    controls.apply_selected()
    assert controls.document == before
    assert "collinear" in controls.status.text()
    assert controls.pending
    with pytest.raises(ValueError, match="collinear"):
        controls.save(path)
    controls.close()
    reopened.close()


def test_point_delete_and_wrong_scene_load_preserve_state(tmp_path: Path) -> None:
    _app()
    controls = GeometryControls(ReferenceGeometry(scene_id="scene"))
    controls.add_point()
    assert len(controls.document.points) == 1
    controls.delete_selected()
    assert not controls.document.points
    controls.undo()
    assert len(controls.document.points) == 1
    wrong = tmp_path / "wrong.json"
    ReferenceGeometry(scene_id="wrong").save(wrong)
    before = controls.document
    with pytest.raises(ValueError, match="scene"):
        controls.load(wrong)
    assert controls.document == before
    controls.close()


def test_opening_and_saving_preserves_sub_display_precision(tmp_path: Path) -> None:
    _app()
    original = ReferenceGeometry(
        scene_id="scene",
        points=(ReferencePoint(position_m=(0.123456789, 2, 3), opacity=0.1234567),),
    )
    controls = GeometryControls(original)
    assert not controls.pending
    path = tmp_path / "precise.json"
    controls.save(path)
    assert ReferenceGeometry.load(path, scene_id="scene") == original
    controls.title.setText("Edited Name")
    controls.fields["position_m"][1].setValue(4)
    controls.apply_selected()
    assert controls.document.points[0].position_m == (0.123456789, 4, 3)
    controls.close()
