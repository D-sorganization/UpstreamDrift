"""Portable source evidence for an explicitly reused camera layout."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from src.motion_capture.rig.capture_notes import read_notes

from .evidence import reviewed_intrinsics
from .frames import verify_frame

MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


def read_bounded(path: Path, limit: int = MAX_DOCUMENT_BYTES) -> bytes:
    """Read one bounded evidence file without trusting its declared size."""
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Calibration evidence exceeds its file limit")
    return data


def reviewed_source(root: Path, payload: dict[str, Any]) -> bytes:
    """Verify the original revision and its archived frames without Tools imports."""
    if payload.get("schema_version") != "capture-reference-solve/1":
        raise ValueError("Choose an original reviewed camera estimate")
    intrinsics = reviewed_intrinsics(payload, root)
    revision = UUID(payload["reference_revision_id"])
    record = json.loads(
        read_bounded(root / "reference_calibration" / f"{revision}.json")
    )
    samples = record.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 256:
        raise ValueError("Reference evidence needs its archived observations")
    if record.get("capture_id") != payload["capture_id"]:
        raise ValueError("Reference observations belong to another capture")
    for sample in samples:
        verify_frame(
            root,
            sample["source_frame"],
            sample["source_sha256"],
            capture_id=payload["capture_id"],
        )
    return intrinsics


def _source_root(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.parent != PurePosixPath("reference_calibration/sources"):
        raise ValueError("Use capture-owned calibration evidence")
    UUID(path.name)
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Calibration evidence leaves its capture workspace")
    return target


def source_result_path(root: Path, relative: str) -> Path:
    """Resolve only the immutable original reviewed-result filename contract."""
    path = PurePosixPath(relative)
    if (
        path.parent != PurePosixPath("reference_calibration/results")
        or not path.name.startswith("reviewed-")
        or path.suffix != ".json"
    ):
        raise ValueError("Choose an original reviewed camera estimate")
    UUID(path.stem.removeprefix("reviewed-"))
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Reviewed result leaves its capture workspace")
    return target


def assigned_intrinsics(
    payload: dict[str, Any], root: Path | None, *, verify_frames: bool = False
) -> bytes:
    """Reject stale or rebound copies before the new swing enters processing."""
    if root is None or payload.get("capture_id") != read_notes(root).capture_id:
        raise ValueError("Review this camera layout for the selected capture")
    reviewed = payload.get("reviewed_utc")
    if (
        payload.get("operator_reviewed") is not True
        or payload.get("scene_confirmed") is not True
        or payload.get("settings_confirmed") is not True
        or not isinstance(reviewed, str)
        or datetime.fromisoformat(reviewed).utcoffset() is None
    ):
        raise ValueError("Confirm camera settings and the unchanged scene before reuse")
    evidence = payload["source_evidence"]
    source = _source_root(root, evidence["root"])
    data = read_bounded(source_result_path(source, evidence["result_path"]))
    if hashlib.sha256(data).hexdigest() != evidence["sha256"]:
        raise ValueError("Original camera review changed; review the layout again")
    original = json.loads(data)
    if original.get("capture_id") != payload.get("source_capture_id"):
        raise ValueError("Original capture identity changed")
    for key in (
        "cameras",
        "intrinsic_records",
        "profile_selections",
        "scene_id",
        "world_frame",
        "anchor",
        "reference_revision_id",
    ):
        if payload.get(key) != original.get(key):
            raise ValueError("Reused camera layout differs from its original review")
    return (
        reviewed_source(source, original)
        if verify_frames
        else reviewed_intrinsics(original, source)
    )


def validate_reference_layout(payload: dict[str, Any], root: Path | None) -> None:
    """Gate capture-owned layouts at the camera reader, including wizard bypasses."""
    from src.motion_capture.rig.bundle import load_bundle
    from ..calibration_profiles import validate_profile_set

    if root is None:
        raise ValueError("Open the selected capture before reading this camera layout")
    data = json.dumps(payload).encode()
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Calibration layout exceeds its document limit")
    plan, index, _ = load_bundle(root)
    identities = {camera.view: camera.identity for camera in plan.cameras}
    expected = {
        entry.view: (
            identities[entry.view],
            (int(entry.width or 0), int(entry.height or 0)),
        )
        for entry in index.recordings
    }
    validate_profile_set(
        data, expected, capture_root=root, verify_reference_frames=True
    )
