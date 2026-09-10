"""Copy reviewed layout evidence into another capture after fresh confirmation.

This is a file-backed assignment, not another camera solve. Optical settings and
scene stability are declared by the player, never inferred from device identity.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid4

from src.motion_capture.rig.capture_notes import MAX_NOTES_BYTES, NOTES_FILE, read_notes
from src.motion_capture.rig.documents import write_document

from ..calibration_profiles import validate_profile_set
from .frames import MAX_FRAME_BYTES, MAX_METADATA_BYTES
from .reuse_evidence import (
    MAX_DOCUMENT_BYTES,
    read_bounded,
    reviewed_source,
    source_result_path,
)

MAX_COPY_BYTES = 512 * 1024 * 1024


def _inspect(request: dict[str, Any]) -> tuple[Path, str, bytes, dict[str, Any]]:
    parameters = request["parameters"]
    selected = Path(parameters["source_path"]).resolve()
    root = selected.parents[2]
    relative = selected.relative_to(root).as_posix()
    source_result_path(root, relative)
    data = read_bounded(selected)
    payload = json.loads(data)
    if payload.get("schema_version") != "capture-reference-solve/1":
        raise ValueError("Choose an original reviewed camera estimate")
    expected = {
        view: (item[0], tuple(item[1]))
        for view, item in parameters["expected_cameras"].items()
    }
    validate_profile_set(
        data, expected, capture_root=root, verify_reference_frames=True
    )
    target = Path(request["workspace"])
    if read_notes(target).capture_id != request["capture_id"]:
        raise ValueError("Selected capture identity changed; reopen calibration")
    if target.resolve() == root or request["capture_id"] == payload["capture_id"]:
        raise ValueError("Choose a reviewed layout from a different capture")
    return root, relative, data, payload


def inspect_reuse(request: dict[str, Any]) -> dict[str, Any]:
    """Preview only; this action never publishes a calibration assignment."""
    root, _, data, payload = _inspect(request)
    return {
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_title": read_notes(root).title,
        "source_capture_id": payload["capture_id"],
        "result": payload,
    }


def _copy_evidence(
    source: Path, target: Path, relative: str, payload: dict[str, Any]
) -> None:
    revision = f"reference_calibration/{UUID(payload['reference_revision_id'])}.json"
    revision_data = read_bounded(source / revision)
    if hashlib.sha256(revision_data).hexdigest() != payload["revision_file_sha256"]:
        raise ValueError("Reference observations changed while copying; review again")
    document = json.loads(revision_data)
    files = {
        NOTES_FILE: MAX_NOTES_BYTES,
        relative: MAX_DOCUMENT_BYTES,
        revision: MAX_DOCUMENT_BYTES,
    }
    for sample in document["samples"]:
        frame = sample["source_frame"]
        files[frame] = MAX_FRAME_BYTES
        files[Path(frame).with_suffix(".json").as_posix()] = MAX_METADATA_BYTES
    total = 0
    for name, limit in files.items():
        data = read_bounded(source / name, limit)
        total += len(data)
        if total > MAX_COPY_BYTES:
            raise ValueError("Calibration evidence exceeds the 512 MiB copy limit")
        out = target / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("xb") as stream:
            stream.write(data)
    reviewed_source(target, payload)


def adopt_layout(request: dict[str, Any]) -> dict[str, Any]:
    """Publish an immutable assignment after copying and rechecking source bytes."""
    parameters = request["parameters"]
    if (
        parameters.get("settings_confirmed") is not True
        or parameters.get("scene_confirmed") is not True
    ):
        raise ValueError("Confirm camera settings and the unchanged camera/ball scene")
    source, relative, data, payload = _inspect(request)
    digest = hashlib.sha256(data).hexdigest()
    if digest != parameters.get("source_sha256"):
        raise ValueError("Camera layout changed since review; open it again")
    root = Path(request["workspace"])
    folder = root / "reference_calibration"
    sources = folder / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    destination = sources / str(uuid4())
    with TemporaryDirectory(prefix=".reuse-", dir=sources) as temporary:
        staged = Path(temporary) / "capture"
        staged.mkdir()
        _copy_evidence(source, staged, relative, payload)
        if hashlib.sha256(read_bounded(staged / relative)).hexdigest() != digest:
            raise ValueError("Camera layout changed while copying; review it again")
        document = {
            key: payload[key]
            for key in (
                "cameras",
                "intrinsic_records",
                "profile_selections",
                "scene_id",
                "world_frame",
                "anchor",
                "reference_revision_id",
            )
        }
        document.update(
            schema_version="capture-reference-assignment/1",
            capture_id=request["capture_id"],
            source_capture_id=payload["capture_id"],
            source_evidence={
                "root": destination.relative_to(root).as_posix(),
                "result_path": relative,
                "sha256": digest,
            },
            operator_reviewed=True,
            settings_confirmed=True,
            scene_confirmed=True,
            reviewed_utc=datetime.now(UTC).isoformat(),
        )
        staged.rename(destination)
    assignments = folder / "assignments"
    assignments.mkdir(exist_ok=True)
    path = assignments / f"{uuid4()}.json"
    write_document(path, document)
    return {"result_path": path.relative_to(root).as_posix(), "result": document}
