"""Cross-capture reuse keeps the original review and portable evidence intact."""

import json
import shutil
from pathlib import Path

import pytest

from solver_checks import make_request
from src.motion_capture.rig.capture_notes import CaptureNotes
from src.motion_capture.rig.documents import write_document
from src.tools.capture_rig.calibration_profiles import validate_profile_set
from src.tools.capture_rig.reference_calibration.solver import (
    accept_result,
    solve_reference,
)


@pytest.fixture(scope="module")
def reviewed_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("reviewed-source")
    request, _, _ = make_request(root)
    solved = solve_reference(request)
    request["parameters"].update(
        result_id=solved["result"]["layout_id"],
        result_sha256=solved["result_sha256"],
        reviewed=True,
    )
    accepted = accept_result(request)
    return root, accepted["result_path"]


@pytest.fixture
def reuse_case(tmp_path, reviewed_source):
    source, relative = reviewed_source
    source_copy = tmp_path / "source"
    shutil.copytree(source, source_copy)
    target = tmp_path / "new-swing"
    target.mkdir()
    write_document(
        target / "capture_notes.json",
        CaptureNotes(capture_id="new-swing", title="Driver Swing 2").model_dump(
            mode="json"
        ),
    )
    document = json.loads((source_copy / relative).read_bytes())
    expected = {
        item["view"]: (
            item["setup"]["camera_identity"],
            tuple(item["setup"]["image_size_px"]),
        )
        for item in document["profile_selections"]
    }
    _write_target_bundle(target, expected)
    request = {
        "workspace": str(target),
        "capture_id": "new-swing",
        "parameters": {
            "source_path": str(source_copy / relative),
            "expected_cameras": expected,
            "settings_confirmed": True,
            "scene_confirmed": True,
        },
    }
    return request, source_copy, target, expected


def _write_target_bundle(target, expected):
    from src.motion_capture.rig.bundle import RecordingEntry, RecordingsIndex
    from src.motion_capture.rig.plan import RigPlan, CameraBinding, CaptureMode
    from src.motion_capture.rig.session import SessionManifest, CaptureOutcome

    cameras = tuple(
        CameraBinding(
            view=view, serial=identity, mode=CaptureMode(width=size[0], height=size[1])
        )
        for view, (identity, size) in expected.items()
    )
    RigPlan(name="Synthetic Reuse", cameras=cameras).save(target / "plan.json")
    index = RecordingsIndex(
        duration_s=1,
        recordings=tuple(
            RecordingEntry(
                view=camera.view,
                identity=camera.identity,
                file=f"{camera.view}.avi",
                bytes=0,
                returncode=1,
                requested_mode=camera.mode,
                requested_duration_s=1,
                width=camera.mode.width,
                height=camera.mode.height,
            )
            for camera in cameras
        ),
    )
    write_document(target / "recordings.json", index.model_dump(mode="json"))
    SessionManifest(
        plan_name="Synthetic Reuse",
        started_utc="2026-09-10T00:00:00+00:00",
        duration_s=1,
        cameras=(),
        outcome=CaptureOutcome.BLOCKED,
    ).save(target / "session_manifest.json")


def _preview_and_adopt(request):
    from src.tools.capture_rig.reference_calibration.reuse import (
        inspect_reuse,
        adopt_layout,
    )

    preview = inspect_reuse(request)
    request["parameters"]["source_sha256"] = preview["source_sha256"]
    return adopt_layout(request)


def test_new_capture_retains_portable_evidence_and_original_review(reuse_case):
    request, source, target, expected = reuse_case
    accepted = _preview_and_adopt(request)
    data = (target / accepted["result_path"]).read_bytes()
    document = json.loads(data)
    assert document["capture_id"] == "new-swing"
    assert document["schema_version"] == "capture-reference-assignment/1"
    assert document["source_capture_id"] == "synthetic-reference"
    assert document["scene_confirmed"] is True
    # Moving the original capture must not break the new swing's calibration.
    source.rename(source.with_name("archived-source"))
    validate_profile_set(data, expected, capture_root=target)
    from src.motion_capture.reconstruct.pipeline import start_cameras_from

    cameras = start_cameras_from(target / accepted["result_path"], capture_root=target)
    assert len(cameras) == len(expected)
    embedded = target / document["source_evidence"]["root"]
    original = json.loads(
        (embedded / document["source_evidence"]["result_path"]).read_bytes()
    )
    assert document["cameras"] == original["cameras"]
    assert original["capture_id"] == "synthetic-reference"
    assert (embedded / "reference_calibration/frames").is_dir()


@pytest.mark.parametrize("confirmation", ["settings_confirmed", "scene_confirmed"])
def test_reuse_requires_both_fresh_confirmations(reuse_case, confirmation):
    request, _, target, _ = reuse_case
    request["parameters"][confirmation] = False
    with pytest.raises(ValueError, match="Confirm"):
        _preview_and_adopt(request)
    assert not list(target.glob("reference_calibration/assignments/*.json"))


def test_reuse_rejects_changed_source_after_preview(reuse_case):
    from src.tools.capture_rig.reference_calibration.reuse import (
        inspect_reuse,
        adopt_layout,
    )

    request, _, _, _ = reuse_case
    preview = inspect_reuse(request)
    request["parameters"]["source_sha256"] = preview["source_sha256"]
    path = Path(request["parameters"]["source_path"])
    document = json.loads(path.read_bytes())
    document["limitations"].append("Changed after review")
    write_document(path, document)
    with pytest.raises(ValueError, match="changed since review"):
        adopt_layout(request)


def test_reuse_rejects_changed_camera_identity(reuse_case):
    request, _, _, _ = reuse_case
    view = next(iter(request["parameters"]["expected_cameras"]))
    request["parameters"]["expected_cameras"][view] = ("other-camera", (1920, 1080))
    with pytest.raises(ValueError, match="identity or recorded image size"):
        _preview_and_adopt(request)


@pytest.mark.parametrize("tamper", ["frame", "cameras", "capture", "path"])
def test_processing_revalidates_assignment_evidence(reuse_case, tamper):
    request, _, target, expected = reuse_case
    accepted = _preview_and_adopt(request)
    document = json.loads((target / accepted["result_path"]).read_bytes())
    if tamper == "frame":
        embedded = target / document["source_evidence"]["root"]
        next(embedded.glob("reference_calibration/frames/*.png")).write_bytes(
            b"changed"
        )
    elif tamper == "cameras":
        document["cameras"][0]["extrinsics"]["translation_world_from_camera_m"][0] += 1
    elif tamper == "capture":
        document["capture_id"] = "another-swing"
    else:
        document["source_evidence"]["root"] = "../source"
    with pytest.raises(ValueError):
        validate_profile_set(
            json.dumps(document).encode(),
            expected,
            capture_root=target,
            verify_reference_frames=True,
        )


def test_cli_rejects_another_capture_before_reconstruction(reuse_case):
    from argparse import Namespace
    from src.motion_capture.rig.__main__ import cmd_reconstruct

    request, _, target, _ = reuse_case
    accepted = _preview_and_adopt(request)
    path = target / accepted["result_path"]
    document = json.loads(path.read_bytes())
    document["capture_id"] = "different-swing"
    write_document(path, document)
    with pytest.raises(ValueError, match="selected capture"):
        cmd_reconstruct(
            Namespace(
                cameras=path, intrinsics=None, session=target, anchor=[], views=""
            )
        )


def test_foreground_review_does_not_read_all_archived_frames(reuse_case, monkeypatch):
    from src.tools.capture_rig.reference_calibration import reuse_evidence
    from src.motion_capture.reconstruct.pipeline import start_cameras_from

    request, _, target, expected = reuse_case
    accepted = _preview_and_adopt(request)
    path = target / accepted["result_path"]

    def frame_read_forbidden(*args, **kwargs):
        raise ValueError("Full frame verification belongs off the UI thread")

    monkeypatch.setattr(reuse_evidence, "verify_frame", frame_read_forbidden)
    validate_profile_set(path.read_bytes(), expected, capture_root=target)
    with pytest.raises(ValueError, match="Full frame verification"):
        start_cameras_from(path, capture_root=target)


def test_library_choices_use_named_archived_captures_and_report_bad_results(reuse_case):
    from uuid import uuid4
    from src.tools.capture_rig.capture_library import CaptureLibrary
    from src.tools.capture_rig.reference_calibration.reuse_catalog import (
        list_reviewed_layouts,
    )

    request, source, target, expected = reuse_case
    _write_target_bundle(source, expected)
    library = CaptureLibrary(target.parent / "library")
    library.register(source)
    library.register(target)
    library.update(source, title="Paper Calibration Take", archived=True)
    broken = source / "reference_calibration/results" / f"reviewed-{uuid4()}.json"
    broken.write_text("{broken", encoding="utf-8")
    request["parameters"]["library_root"] = str(library.root)
    found = list_reviewed_layouts(request)
    assert len(found["choices"]) == 1
    assert "Paper Calibration Take" in found["choices"][0]["label"]
    assert "Archived" in found["choices"][0]["label"]
    assert found["choices"][0]["path"] == request["parameters"]["source_path"]
    assert found["problems"]
