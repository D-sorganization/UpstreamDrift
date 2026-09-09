"""Capture history survives reload and never destroys unreadable evidence."""

from pathlib import Path
import json

import pytest

from src.tools.capture_rig.capture_activity import (
    ACTIVITY_FILE,
    CaptureAction,
    display_status,
    read_activity,
    save_action,
)

pytestmark = pytest.mark.unit


def test_history_preserves_context_and_updates_same_job(tmp_path: Path) -> None:
    action = CaptureAction(
        action="ingest", context="openpose_dnn · observations_openpose"
    )
    assert save_action(tmp_path, action)
    assert display_status(read_activity(tmp_path).actions[0]).startswith("Unconfirmed")
    assert display_status(action, action.id) == "Running"
    save_action(tmp_path, action.complete(1))
    history = read_activity(tmp_path)
    assert len(history.actions) == 1
    assert history.actions[0].status == "failed"
    assert history.actions[0].context == action.context
    assert history.actions[0].finished is not None


def test_history_does_not_create_capture_or_overwrite_invalid_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "new-take"
    action = CaptureAction(action="record")
    assert not save_action(root, action)
    assert not root.exists()
    path = tmp_path / ACTIVITY_FILE
    path.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        save_action(tmp_path, action)
    assert path.read_text(encoding="utf-8") == "broken"


def test_capture_histories_are_independent_and_cancellation_is_explicit(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    action = CaptureAction(action="fit_model", context="full-body · face-on")
    save_action(first, action.complete(0, cancelled=True))
    assert read_activity(first).actions[0].status == "cancelled"
    assert read_activity(second).actions == []


def test_detector_identity_and_changed_edits_are_not_inferred_from_directory(
    tmp_path: Path,
) -> None:
    from src.tools.capture_rig.capture_evidence import observation_evidence
    from tests.tools.capture_rig.test_core import _bundle

    _bundle(tmp_path)
    folder = tmp_path / "observations"
    folder.mkdir(exist_ok=True)
    output = folder / "face_on.json"
    output.write_text("{}", encoding="utf-8")
    index = folder / "observations.json"
    index.write_text(
        json.dumps(
            {
                "provenance": {
                    "estimator": "openpose_dnn",
                    "edits": {"schema_version": "swing-edits/1.0.0", "views": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    detector, status = observation_evidence(tmp_path, output)
    assert detector == "openpose_dnn"
    assert "Edits Match" in status
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["provenance"]["edits"]["views"] = {"old-view": {}}
    index.write_text(json.dumps(payload), encoding="utf-8")
    assert "Stale" in observation_evidence(tmp_path, output)[1]
    index.write_text("broken", encoding="utf-8")
    assert "Needs Review" in observation_evidence(tmp_path, output)[1]
