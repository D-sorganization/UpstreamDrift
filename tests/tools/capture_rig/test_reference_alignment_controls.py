"""Instructor placement and event editing contracts (#9883)."""

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox

from src.motion_capture.reference.registration import ReferenceRegistration, TimeMapping
from src.tools.capture_rig.reference_controls import SpatialControls, TimeControls
from tests.tools.capture_rig.test_pane_layout import _app
from tests.tools.capture_rig.test_reference_comparison_ui import synthetic_motion
from tests.motion_capture.rig.test_ingest import _bundle
from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_spatial_apply_preserves_time_and_uses_shared_rotation() -> None:
    app = _app()
    asset = synthetic_motion()
    reg = ReferenceRegistration(
        reference_id=asset.id,
        calibration_id="manual",
        time_mapping=TimeMapping(offset_s=0.3),
    )
    panel = SpatialControls(reg, "motion", (640, 480))
    changes = []
    panel.changed.connect(changes.append)
    panel.translation[0].setValue(1.25)
    panel.rotation[2].setValue(90)
    panel.apply_placement()
    changed = changes[-1]
    assert changed.time_mapping == reg.time_mapping
    assert changed.transform.translation_m == (1.25, 0, 0)
    np.testing.assert_allclose(
        changed.transform.rotation, ((0, -1, 0), (1, 0, 0), (0, 0, 1)), atol=1e-9
    )
    panel.close()
    app.processEvents()


def test_image_adjustment_preserves_existing_perspective() -> None:
    app = _app()
    asset = synthetic_motion()
    base = ((1.0, 0.0, 4.0), (0.0, 1.0, 5.0), (0.0001, 0.0, 1.0))
    reg = ReferenceRegistration(
        reference_id=asset.id, calibration_id="manual", image_transform_2d=base
    )
    panel = SpatialControls(reg, "video", (640, 480), reference_size=(640, 480))
    changes = []
    panel.changed.connect(changes.append)
    panel.image_x.setValue(10)
    panel.image_y.setValue(-5)
    panel.apply_image()
    expected = np.array(((1, 0, 10), (0, 1, -5), (0, 0, 1))) @ np.asarray(base)
    np.testing.assert_allclose(changes[-1].image_transform_2d, expected)
    assert changes[-1].transform == reg.transform
    panel.close()
    app.processEvents()


def test_event_table_rejects_invalid_pairs_without_changing_registration() -> None:
    app = _app()
    asset = synthetic_motion()
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="manual")
    panel = TimeControls(reg)
    changes = []
    panel.changed.connect(changes.append)
    panel.events.setRowCount(2)
    for row, values in enumerate((("Top", "0.1", "0.2"), ("Impact", "0.2", "0.3"))):
        for column, value in enumerate(values):
            panel.events.setItem(row, column, QTableWidgetItem(value))
    panel.apply_events()
    assert len(changes) == 1
    assert changes[0].time_mapping.scene_to_reference(0.25) == pytest.approx(0.15)
    panel.events.item(1, 2).setText("0.15")
    panel.apply_events()
    assert len(changes) == 1
    assert panel.problem.text()
    panel.close()
    app.processEvents()


def test_dialog_guards_pending_fields_notes_switch_and_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    for name in ("A", "B"):
        library.save(synthetic_motion().changed(title=name))
    dialog = ReferenceComparisonDialog(root, "a", library)
    dialog.show()
    app.processEvents()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel
    )
    dialog.spatial.translation[0].setValue(1.2)
    dialog.notes.setPlainText("Compare hip turn at impact")
    dialog.asset_selector.setCurrentIndex(1)
    assert dialog.asset_selector.currentIndex() == 0
    assert dialog._current_asset.title == "A"
    dialog.reject()
    assert dialog.isVisible()
    assert dialog.save()
    assert dialog._session.registration.transform.translation_m[0] == 1.2
    assert dialog._session.notes == "Compare hip turn at impact"
    assert not dialog._dirty()
    before = dialog._session
    dialog.reset_alignment()
    assert dialog._session.registration.transform.translation_m == (0, 0, 0)
    dialog.undo_change()
    assert dialog._session == before
    dialog.close()
    assert not dialog.isVisible()
    app.processEvents()


def test_laptop_layout_keeps_preview_and_controls_in_window(tmp_path: Path) -> None:
    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    library.save(synthetic_motion())
    dialog = ReferenceComparisonDialog(root, "a", library)
    dialog.resize(900, 740)
    dialog.show()
    app.processEvents()
    try:
        assert dialog.width() == 900
        assert dialog.minimumSizeHint().width() <= 900
        assert dialog.splitter.orientation() == Qt.Orientation.Vertical
        assert dialog.canvas.height() >= 220
        for index in range(dialog.inspector.count()):
            dialog.inspector.setCurrentIndex(index)
            app.processEvents()
            assert dialog.inspector.geometry().right() <= dialog.width()
        dialog.resize(1280, 800)
        app.processEvents()
        assert dialog.splitter.orientation() == Qt.Orientation.Horizontal
        assert dialog.canvas.width() > dialog.inspector.width()
    finally:
        dialog.close()
        app.processEvents()


def test_independent_expert_clock_pairs_visible_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PyQt6.QtWidgets import QInputDialog

    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    library.save(synthetic_motion())
    dialog = ReferenceComparisonDialog(root, "a", library)
    dialog.show()
    dialog.inspector.setCurrentIndex(1)
    timeline = dialog.expert_timeline
    timeline.follow(1000)
    assert timeline.outside
    assert "No expert sample" in timeline.clock.text()
    timeline.linked.setChecked(False)
    timeline.slider.setValue(1)
    reference_time = timeline.reference_time
    scene_time = dialog._clock.player_time(dialog.slider.value() / dialog.fps)
    monkeypatch.setattr(
        QInputDialog, "getText", lambda *args, **kwargs: ("Impact", True)
    )
    dialog._pair_frames()
    mapping = dialog._session.registration.time_mapping
    assert mapping.reference_to_scene(reference_time) == pytest.approx(scene_time)
    assert not timeline.outside
    assert dialog.save()
    dialog.close()
    app.processEvents()


def test_changed_asset_can_be_reviewed_without_losing_previous_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.motion_capture.reference.comparison import comparison_session_path

    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    asset = synthetic_motion()
    library.save(asset)
    dialog = ReferenceComparisonDialog(root, "a", library)
    assert dialog.save()
    dialog.close()
    path = comparison_session_path(root, "a", asset.id)
    previous = path.read_bytes()
    points = [
        [tuple(p) if p is not None else None for p in frame] for frame in asset.points_m
    ]
    points[0][0] = (0.25, 0.5, 1.0)
    library.save(asset.changed(points_m=points))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
    )
    reviewed = ReferenceComparisonDialog(root, "a", library, review_stale=True)
    assert reviewed._dirty()
    assert path.read_bytes() == previous
    assert reviewed.save()
    backups = list(path.parent.glob(f"{path.stem}.before-review-*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous
    reviewed.close()
    app.processEvents()


def test_alignment_undo_preserves_notes_written_after_placement(tmp_path: Path) -> None:
    app = _app()
    root = _bundle(tmp_path)
    library = ReferenceLibrary(tmp_path / "references")
    library.save(synthetic_motion())
    dialog = ReferenceComparisonDialog(root, "a", library)
    dialog.spatial.translation[0].setValue(1)
    dialog.spatial.apply_placement()
    dialog.notes.setPlainText("Retain this lesson observation")
    dialog.undo_change()
    assert dialog._session.registration.transform.translation_m[0] == 0
    assert dialog._session.notes == "Retain this lesson observation"
    assert dialog.save()
    dialog.close()
    app.processEvents()
