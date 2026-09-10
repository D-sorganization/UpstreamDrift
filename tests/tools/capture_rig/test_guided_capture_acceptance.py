"""Guided restart retains capture evidence but renews camera-setting review."""

import json
from pathlib import Path

import pytest

from src.motion_capture.provenance import sha256_of
from src.motion_capture.reference import load_comparison_session
from src.motion_capture.reference.storage import ReferenceLibrary
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import SessionEdits, save_edits
from src.motion_capture.rig.equipment import model_equipment_context
from src.shared.python.club_data.player_clubs import PlayerBag
from src.tools.capture_rig.capture_library import CaptureLibrary
from src.tools.capture_rig.club_editor import ClubEditorDialog
from src.tools.capture_rig.equipment import (
    load_capture_club,
    save_bag,
    save_capture_club,
)
from src.tools.capture_rig.gui import CaptureRigWidget
from src.tools.capture_rig.library_actions import LIBRARY_ROOT_KEY
from src.tools.capture_rig.reference_comparison import ReferenceComparisonDialog
from src.tools.capture_rig.reference_import import load_reference_video
from src.tools.capture_rig.wizard_evidence import CalibrationReview
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_equipment import club
from tests.tools.capture_rig.test_pane_layout import _app, _settings
from tests.tools.capture_rig.test_wizard_actions import _ready
from tests.tools.capture_rig.test_wizard_evidence import _reviewed_calibration

pytestmark = [pytest.mark.unit, pytest.mark.ui]


@pytest.mark.parametrize("known_length", [False, True])
def test_restart_preserves_club_and_capture_but_renews_calibration_review(
    tmp_path: Path, known_length: bool
) -> None:
    _app()
    settings = _settings(tmp_path)
    library = tmp_path / "library"
    settings.setValue(LIBRARY_ROOT_KEY, str(library))
    root = _bundle(tmp_path)
    CaptureLibrary(library).register(root)
    editor = ClubEditorDialog(club())
    try:
        editor.notes.setPlainText(
            "Lesson capture; retain this club's original evidence"
        )
        if known_length:
            editor.length.mode.setCurrentText("Measured")
            editor.length.input.set_value(37, "in")
        selected = editor.record()
    finally:
        editor.close()
    bag = PlayerBag(clubs=(selected,))
    save_bag(library / "player_clubs.json", bag)
    snapshot = save_capture_club(root, selected)
    save_edits(root, SessionEdits())
    host = CaptureRigWidget(settings=settings)
    try:
        host._open_library_capture(root)
        review, calibration = _reviewed_calibration(root, tmp_path)
        host.process.start_edit.setText(str(calibration))
        actions = host.wizard_actions
        actions.show()
        wizard = actions.dialog
        assert wizard is not None
        wizard.choose(["edit", "fit_model"])
        wizard.next()
        _ready(actions)
        actions.review = review
        actions.refresh()
        _ready(actions)
        wizard.navigate("step.intrinsics")
        _ready(actions)
        assert wizard.step_pages["clubs.bag"].isComplete()
        assert wizard.step_pages["step.intrinsics"].isComplete()
        actions.save()
        saved = (root / "capture_workflow.json").read_bytes()
        capture_bytes = {
            p: p.read_bytes()
            for p in (
                root / "capture_club.json",
                root / "swing_edits.json",
                root / "session_manifest.json",
            )
        }
    finally:
        host.shutdown()

    # A later bag edit must not rewrite the club used by an earlier capture.
    changed = selected.model_copy(update={"notes": "Changed after the lesson"})
    save_bag(
        library / "player_clubs.json",
        PlayerBag(clubs=(changed,)),
        expected_revision=bag.revision,
    )
    reopened = CaptureRigWidget(settings=settings)
    try:
        reopened._open_library_capture(root)
        reopened.process.start_edit.setText(str(calibration))
        actions = reopened.wizard_actions
        actions.show()
        _ready(actions)
        actions.resume()
        _ready(actions)
        wizard = actions.dialog
        assert wizard is not None and wizard.current_step == "step.intrinsics"
        assert actions.review is None
        assert not wizard.step_pages["step.intrinsics"].isComplete()
        assert wizard.step_pages["step.intrinsics"].open_button.isEnabled()
        assert wizard.step_pages["clubs.bag"].isComplete()
        assert load_capture_club(root) == snapshot
        context = model_equipment_context(root)
        assert context is not None and context["club_number"] == "7"
        expected = pytest.approx(0.9398) if known_length else None
        assert context["eligible_values"]["assembled_length_si"] == expected
        assert context["constraints_applied"] == []
        assert (root / "capture_workflow.json").read_bytes() == saved
        assert all(path.read_bytes() == value for path, value in capture_bytes.items())
        entries = reopened.library_actions.library().catalog_entries()
        assert len(entries) == 1 and entries[0].capture_id == snapshot.capture_id
        actions.review = CalibrationReview.confirmed(root, calibration)
        actions.refresh()
        _ready(actions)
        assert wizard.step_pages["step.intrinsics"].isComplete()
        changed_profile = json.loads(calibration.read_text(encoding="utf-8"))
        changed_profile["profile_selections"][0]["setup"]["zoom"] = "ring mark 3"
        write_document(calibration, changed_profile)
        actions.refresh()
        _ready(actions)
        assert not wizard.step_pages["step.intrinsics"].isComplete()
        assert wizard.step_pages["step.intrinsics"].open_button.isEnabled()
        wizard.restart()
        wizard.choose(["edit"])
        wizard.next()
        _ready(actions)
        assert wizard.step_pages["capture.selection"].isComplete()
        assert load_capture_club(root) == snapshot
    finally:
        reopened.shutdown()


def test_instructor_video_alignment_completes_guided_step_without_calibration(
    tmp_path: Path,
) -> None:
    _app()
    root = _bundle(tmp_path)
    settings = _settings(tmp_path)
    library = tmp_path / "library"
    settings.setValue(LIBRARY_ROOT_KEY, str(library))
    host = CaptureRigWidget(settings=settings)
    try:
        host._open_library_capture(root)
        assert host.media is not None
        source = host.media.views[0].recording
        assert source is not None
        original_digest = sha256_of(source)
        asset = load_reference_video(source)
        references = ReferenceLibrary(library / "references")
        references.save(asset)
        save_edits(root, SessionEdits())
        comparison = ReferenceComparisonDialog(root, "a", references)
        try:
            comparison.offset_spin.setValue(0.05)
            comparison.opacity_spin.setValue(0.65)
            assert comparison.save()
        finally:
            comparison.close()
        saved = load_comparison_session(root / "comparisons" / f"a_{asset.id}.json")
        assert saved.registration is not None
        assert saved.registration.time_mapping.offset_s == pytest.approx(0.05)
        assert not saved.registration.is_calibrated
        actions = host.wizard_actions
        actions.show()
        wizard = actions.dialog
        assert wizard is not None
        wizard.choose(["edit", "compare_video"])
        wizard.next()
        _ready(actions)
        assert (
            wizard.route is not None and "step.intrinsics" not in wizard.route.step_ids
        )
        assert wizard.step_pages["compare.video"].isComplete()
        wizard.navigate("compare.video")
        _ready(actions)
        actions.save()
        progress = json.loads(
            (root / "capture_workflow.json").read_text(encoding="utf-8")
        )
        assert "compare_video" in progress["goals"]
        assert sha256_of(source) == original_digest
    finally:
        host.shutdown()
