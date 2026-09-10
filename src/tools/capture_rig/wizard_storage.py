"""Capture-owned wizard bookmarks; saving progress never modifies analysis files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.motion_capture.rig.bundle import MANIFEST_FILE, PLAN_FILE, RECORDINGS_FILE
from src.motion_capture.rig.capture_notes import read_notes
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import EDITS_FILE
from src.motion_capture.rig.equipment import CAPTURE_CLUB_FILE

from .goal_planner import CaptureProgress
from .session import SessionMedia

PROGRESS_FILE = "capture_workflow.json"
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


def read_document(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Workflow metadata is too large: {path.name}")
    return data


def input_revision(media: SessionMedia) -> str:
    """Hash bounded metadata and file stats, without decoding or hashing videos.

    This is a UI invalidation token, not a media-integrity certificate. Replaced
    media retaining identical size and modification time needs manual review.
    """
    digest = hashlib.sha256()
    root = media.root
    for name in (
        PLAN_FILE,
        RECORDINGS_FILE,
        MANIFEST_FILE,
        EDITS_FILE,
        CAPTURE_CLUB_FILE,
        "intrinsics.json",
    ):
        path = root / name
        digest.update(name.encode())
        digest.update(read_document(path) if path.is_file() else b"missing")
    for view in media.views:
        for media_path in (view.recording, view.observations):
            if media_path is None:
                digest.update(b"missing")
                continue
            stat = media_path.stat()
            digest.update(
                json.dumps([str(media_path), stat.st_size, stat.st_mtime_ns]).encode()
            )
        if view.observations is not None:
            path = view.observations.parent / "observations.json"
            digest.update(read_document(path) if path.is_file() else b"missing-index")
    digest.update(json.dumps(media.timing, sort_keys=True).encode())
    return digest.hexdigest()


def save_progress(root: Path, progress: CaptureProgress) -> None:
    """Atomically save a bookmark only for this capture; preserve malformed history."""
    if read_notes(root).capture_id != progress.capture_id:
        raise ValueError("Cannot save workflow progress to another capture")
    path = root / PROGRESS_FILE
    if path.exists():
        load_progress(root)
    write_document(path, progress.model_dump(mode="json"))


def load_progress(root: Path) -> CaptureProgress:
    progress = CaptureProgress.model_validate_json(read_document(root / PROGRESS_FILE))
    if read_notes(root).capture_id != progress.capture_id:
        raise ValueError("Saved workflow belongs to another capture")
    return progress
