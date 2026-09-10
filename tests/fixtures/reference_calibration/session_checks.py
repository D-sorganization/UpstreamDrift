"""Player placement persistence retains camera, target and capture identities."""

from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np
from shared.python.sidekick.lab.mocap.reference_placements import (
    PlacementObservation,
    common_reference,
)

from src.tools.capture_rig.calibration_profiles import CameraSetup
from src.tools.capture_rig.reference_calibration.session import (
    CameraSelection,
    ReferenceSample,
    ReferenceSession,
    load_revision,
    save_revision,
)


def test_worker_marks_partial_points_and_preserves_saved_revision(
    tmp_path: Path,
) -> None:
    from src.motion_capture.rig.capture_notes import CaptureNotes
    from src.motion_capture.rig.documents import write_document
    from src.tools.capture_rig.reference_calibration.frames import archive_frame
    from src.tools.capture_rig.reference_calibration.worker import _dispatch

    original = session().revise(samples=())
    write_document(
        tmp_path / "capture_notes.json",
        CaptureNotes(
            capture_id=original.capture_id, title="Reference Practice"
        ).model_dump(mode="json"),
    )
    image = archive_frame(
        tmp_path,
        np.zeros((480, 640, 3), dtype=np.uint8),
        capture_id=original.capture_id,
        view="front",
        frame_index=2,
        timestamp_s=0.04,
        source_label="Original",
    )
    request = {
        "action": "mark",
        "workspace": str(tmp_path),
        "session": original.model_dump(mode="json"),
        "parameters": {
            "frame_path": image.path,
            "frame_sha256": image.sha256,
            "placement_id": "near-ball",
            "reference_id": "us-letter",
            "points": {"origin": [12, 24]},
            "notes": "One visible corner",
        },
    }
    marked = _dispatch(request)
    assert marked["samples"][0]["observation"]["point_ids"] == ["origin"]
    assert marked["parent_revision_id"] == str(original.revision_id)
    _dispatch(
        {
            "action": "save",
            "workspace": str(tmp_path),
            "capture_id": original.capture_id,
            "session": marked,
        }
    )
    old_path = tmp_path / "reference_calibration" / f"{marked['revision_id']}.json"
    before = old_path.read_bytes()
    request["session"] = marked
    request["parameters"]["points"] = {"origin": [15, 29], "along-arrow": [60, 29]}
    changed = _dispatch(request)
    assert len(changed["samples"]) == 1
    assert changed["revision_id"] != marked["revision_id"]
    assert old_path.read_bytes() == before
    (tmp_path / image.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        _dispatch(
            {
                "action": "save",
                "workspace": str(tmp_path),
                "capture_id": original.capture_id,
                "session": changed,
            }
        )


def camera() -> CameraSelection:
    return CameraSelection(
        view="front",
        setup=CameraSetup(
            camera_identity="camera-serial-1",
            lens="manual lens",
            zoom="unknown",
            focus="infinity",
            image_size_px=(640, 480),
            sensor_mode="full frame",
        ),
    )


def sample(selection: CameraSelection, scene: str) -> ReferenceSample:
    return ReferenceSample(
        observation=PlacementObservation(
            "ball",
            "us-letter",
            "front",
            selection.profile_key,
            ("origin", "along-arrow", "opposite", "across-width"),
            ((10, 10), (50, 10), (50, 60), (10, 60)),
            2,
            40000000,
        ),
        camera_signature=selection.signature(scene),
        source_frame="frames/front-2.png",
        source_sha256="a" * 64,
    )


def session() -> ReferenceSession:
    selection = camera()
    return ReferenceSession(
        capture_id="capture-1",
        title="Paper near the ball",
        scene_id="scene-1",
        cameras=(selection,),
        targets=(common_reference("us-letter"),),
        samples=(sample(selection, "scene-1"),),
    )


def test_nominal_target_and_unknown_manual_zoom_round_trip(tmp_path: Path) -> None:
    original = session()
    path = save_revision(tmp_path, original, capture_id="capture-1")
    restored = load_revision(path, capture_id="capture-1")
    assert restored == original
    assert restored.targets[0].object_points_m[2] == (0.2794, 0.0, 0.2159)
    assert restored.cameras[0].setup.zoom == "unknown"
    assert restored.cameras[0].profile is None
    assert restored.unreviewed_views == ("front",)


@pytest.mark.parametrize("change", ["zoom", "image_size_px", "scene"])
def test_changed_camera_settings_or_movement_reject_saved_observations(
    change: str,
) -> None:
    original = session()
    data = original.model_dump(mode="json")
    if change == "scene":
        data["scene_id"] = "camera-moved"
    else:
        data["cameras"][0]["setup"][change] = "2x" if change == "zoom" else [1280, 720]
    with pytest.raises(ValueError, match="camera settings or scene"):
        ReferenceSession.model_validate(data)


def test_repeating_and_disabling_placement_preserves_previous_revision(
    tmp_path: Path,
) -> None:
    original = session()
    previous = save_revision(tmp_path, original, capture_id="capture-1")
    first = original.samples[0]
    repeated = first.model_copy(
        update={"observation": replace(first.observation, placement_id="near-feet")}
    )
    changed = original.revise(
        samples=(first.model_copy(update={"enabled": False}), repeated)
    )
    current = save_revision(tmp_path, changed, capture_id="capture-1")
    assert current != previous
    assert load_revision(previous, capture_id="capture-1").samples[0].enabled
    assert not load_revision(current, capture_id="capture-1").samples[0].enabled
    assert changed.parent_revision_id == original.revision_id


def test_wrong_capture_and_conflicting_revision_cannot_replace_saved_work(
    tmp_path: Path,
) -> None:
    original = session()
    with pytest.raises(ValueError, match="another capture"):
        save_revision(tmp_path, original, capture_id="different-capture")
    path = save_revision(tmp_path, original, capture_id="capture-1")
    with pytest.raises(ValueError, match="another capture"):
        load_revision(path, capture_id="different-capture")
    changed = original.model_copy(update={"title": "Overwrite attempt"})
    with pytest.raises(ValueError, match="existing revision"):
        save_revision(tmp_path, changed, capture_id="capture-1")
    assert load_revision(path, capture_id="capture-1") == original


@pytest.mark.parametrize("invalid", ["duplicate", "point", "profile", "pixel"])
def test_ambiguous_or_incompatible_samples_are_rejected(invalid: str) -> None:
    original = session()
    data = original.model_dump(mode="json")
    observation = data["samples"][0]["observation"]
    if invalid == "duplicate":
        data["samples"].append(data["samples"][0])
    elif invalid == "point":
        observation["point_ids"][0] = "image-top-left"
    elif invalid == "profile":
        observation["profile_id"] = "different-lens"
    else:
        observation["pixels_px"][0][0] = 640
    with pytest.raises(ValueError):
        ReferenceSession.model_validate(data)


def test_corrupt_revision_is_reported_and_retained(tmp_path: Path) -> None:
    original = session()
    path = save_revision(tmp_path, original, capture_id="capture-1")
    path.write_bytes(b"{broken")
    with pytest.raises(ValueError):
        save_revision(tmp_path, original, capture_id="capture-1")
    assert path.read_bytes() == b"{broken"


def test_line_reference_is_saved_as_scale_only() -> None:
    original = session()
    line = common_reference("yardstick")
    changed = original.revise(targets=(line,), samples=())
    assert changed.targets[0].object_points_m[-1][0] == 0.9144
    assert changed.unreviewed_views == ("front",)


def test_frame_path_cannot_escape_revision_workspace() -> None:
    data = session().model_dump(mode="json")
    data["samples"][0]["source_frame"] = "../../private.png"
    with pytest.raises(ValueError, match="relative frame"):
        ReferenceSession.model_validate(data)
