"""Capture status is tied to saved edits, identity and current calibration bytes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import SessionEdits, ViewEdit, save_edits
from src.tools.capture_rig.capture_library import CaptureLibrary, read_notes
from src.tools.capture_rig.goal_catalog import load_catalog
from src.tools.capture_rig.goal_planner import CaptureProgress, resolve, restore
from src.tools.capture_rig.session import load_session
from src.tools.capture_rig.wizard_evidence import CalibrationReview, inspect_capture
from src.tools.capture_rig.wizard_storage import (
    input_revision,
    load_progress,
    save_progress,
)
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_calibration_profiles import calibration, setup
from src.motion_capture.rig.bundle import load_bundle

pytestmark = pytest.mark.unit


def _capture(tmp_path: Path):
    root = _bundle(tmp_path)
    library = CaptureLibrary(tmp_path / "library")
    library.register(root)
    return root, library, load_session(root)


def test_imported_video_is_ready_to_edit_without_devices_or_calibration(
    tmp_path,
) -> None:
    root, library, media = _capture(tmp_path)
    route = resolve(load_catalog(), ["edit"])
    before = inspect_capture(route, media, library.root)
    assert before.states["capture.library"].status == "done"
    assert before.states["capture.selection"].status == "ready"
    save_edits(root, SessionEdits(views={"a": ViewEdit(first=1, last=4)}))
    after = inspect_capture(route, media, library.root)
    assert after.states["capture.selection"].status == "done"
    assert after.input_revision != before.input_revision
    assert read_notes(root).capture_id in after.identity


def test_progress_is_capture_owned_and_preserves_corrupt_existing_metadata(
    tmp_path,
) -> None:
    root, _, media = _capture(tmp_path)
    graph = load_catalog()
    progress = CaptureProgress(
        capture_id=read_notes(root).capture_id,
        goals=("edit",),
        catalog_revision=graph.revision,
        input_revision=input_revision(media),
        current_step="capture.selection",
    )
    save_progress(root, progress)
    assert load_progress(root) == progress
    restored = restore(
        graph,
        progress,
        capture_id=progress.capture_id,
        input_revision=input_revision(media),
    )
    assert restored.goals == ("edit",)
    with (root / "a_1.avi").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="inputs changed"):
        restore(
            graph,
            progress,
            capture_id=progress.capture_id,
            input_revision=input_revision(media),
        )
    (root / "capture_workflow.json").write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        save_progress(root, progress)
    assert (root / "capture_workflow.json").read_text() == "broken"


def test_optional_skip_cannot_hide_a_required_detection_step(tmp_path) -> None:
    root, _, media = _capture(tmp_path)
    graph = load_catalog()
    progress = CaptureProgress(
        capture_id=read_notes(root).capture_id,
        goals=("fit_model",),
        catalog_revision=graph.revision,
        input_revision=input_revision(media),
        current_step="step.detect",
        skipped=("step.detect",),
    )
    with pytest.raises(ValueError, match="required"):
        restore(
            graph,
            progress,
            capture_id=progress.capture_id,
            input_revision=input_revision(media),
        )


def test_old_detection_or_model_identity_never_marks_a_new_fit_done(tmp_path) -> None:
    root, library, media = _capture(tmp_path)
    save_edits(root, SessionEdits())
    observations = root / "observations"
    observations.mkdir()
    write_document(
        observations / "observations.json",
        {
            "provenance": {
                "estimator": "openpose_dnn",
                "edits": {"schema_version": "swing-edits/1.0.0", "views": {"old": {}}},
            }
        },
    )
    path = observations / "a.json"
    path.write_text("{}", encoding="utf-8")
    views = (replace(media.views[0], observations=path), media.views[1])
    media = replace(
        media,
        views=views,
        model_fit={"provenance": {"parameters": {"model": "another"}}},
    )
    evidence = inspect_capture(
        resolve(load_catalog(), ["fit_model"]), media, library.root
    )
    assert evidence.states["step.detect"].status == "blocked"
    assert "Swing Edits Changed" in evidence.states["step.detect"].reason
    assert evidence.states["step.fit_model"].status == "blocked"


def _reviewed_calibration(root: Path, tmp_path: Path) -> tuple[CalibrationReview, Path]:
    path = root / "intrinsics-selected-test.json"
    camera = json.loads(calibration(tmp_path / "lens.json").read_text())[0]
    plan, index, _ = load_bundle(root)
    cameras, selections = [], []
    for binding, recording in zip(plan.cameras, index.recordings, strict=True):
        size = (recording.width, recording.height)
        cameras.append(camera | {"camera_id": binding.view, "image_size_px": size})
        settings = setup().model_copy(
            update={"camera_identity": binding.identity, "image_size_px": size}
        )
        selections.append(
            {
                "view": binding.view,
                "setup": settings.model_dump(mode="json"),
                "confirmed_utc": "2026-09-10T00:00:00+00:00",
            }
        )
    write_document(path, {"cameras": cameras, "profile_selections": selections})
    return CalibrationReview.confirmed(root, path), path


def test_review_is_invalidated_by_changed_calibration_plan_or_capture(tmp_path) -> None:
    root, _, media = _capture(tmp_path)
    review, path = _reviewed_calibration(root, tmp_path)
    assert review.matches(media, path)
    path.write_text("changed", encoding="utf-8")
    assert not review.matches(media, path)
    assert not review.matches(media, root / "another.json")


@pytest.mark.parametrize("goal", ["edit", "reconstruct"])
def test_missing_reviewed_calibration_only_blocks_dependent_work(tmp_path, goal):
    root, library, media = _capture(tmp_path)
    review, path = _reviewed_calibration(root, tmp_path)
    path.unlink()
    route = resolve(load_catalog(), [goal])
    evidence = inspect_capture(
        route, media, library.root, review=review, start_file=path
    )
    assert evidence.states["capture.library"].status == "done"
    assert evidence.states["capture.selection"].status == "ready"
    if goal == "reconstruct":
        assert evidence.states["step.intrinsics"].status == "blocked"
        assert "calibration" in evidence.states["step.intrinsics"].reason.lower()
        assert evidence.states["step.reconstruct"].status == "blocked"


def test_empty_calibration_file_cannot_be_treated_as_reviewed(tmp_path) -> None:
    root, _, _ = _capture(tmp_path)
    path = root / "intrinsics.json"
    write_document(path, {"cameras": []})
    with pytest.raises(ValueError, match="every camera"):
        CalibrationReview.confirmed(root, path)


@pytest.mark.parametrize("content", [b"{broken", b"\xff\xfe"])
def test_unreadable_comparison_keeps_capture_steps_available(tmp_path, content) -> None:
    root, library, media = _capture(tmp_path)
    save_edits(root, SessionEdits())
    comparisons = root / "comparisons"
    comparisons.mkdir()
    broken = comparisons / "unreadable.json"
    broken.write_bytes(content)
    from hashlib import sha256
    from src.motion_capture.reference.model import ReferenceSource, ReferenceVideo
    from src.motion_capture.reference.storage import ReferenceLibrary

    recording = media.views[0].recording
    assert recording is not None
    ReferenceLibrary(library.root / "references").save(
        ReferenceVideo(
            title="Expert",
            source=ReferenceSource(
                path=str(recording),
                sha256=sha256(recording.read_bytes()).hexdigest(),
                format="video",
            ),
            width=16,
            height=16,
            frames=5,
            fps=30,
        )
    )
    evidence = inspect_capture(
        resolve(load_catalog(), ["compare_video"]), media, library.root
    )
    assert evidence.states["capture.library"].status == "done"
    assert evidence.states["capture.selection"].status == "done"
    assert "unreadable.json" in evidence.states["compare.video"].reason
    assert "review" in evidence.states["compare.video"].reason.lower()
    assert broken.read_bytes() == content


def test_removed_pose_output_changes_revision_without_blocking_editing(
    tmp_path,
) -> None:
    root, library, media = _capture(tmp_path)
    save_edits(root, SessionEdits())
    output = root / "observations" / "a.json"
    output.parent.mkdir()
    output.write_text("{}", encoding="utf-8")
    media = replace(
        media, views=(replace(media.views[0], observations=output), media.views[1])
    )
    before = input_revision(media)
    output.unlink()
    evidence = inspect_capture(resolve(load_catalog(), ["edit"]), media, library.root)
    assert evidence.input_revision != before
    assert evidence.states["capture.library"].status == "done"
    assert evidence.states["capture.selection"].status == "done"
    assert input_revision(media) == evidence.input_revision
