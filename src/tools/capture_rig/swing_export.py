"""Publish a separate swing video through the existing coaching exporter (#9860)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import ViewEdit, load_edits
from src.motion_capture.coaching import DrawingLayer

from .clips import ClipRange, ClipRendering, export_clip
from .player import VideoReader
from .session import load_session


def _digest(path: Path, cancelled: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            if cancelled():
                raise InterruptedError("Swing export cancelled")
            digest.update(data)
    return digest.hexdigest()


def publish_export(staged: Path, out: Path) -> None:
    """Publish staged media/JSON without overwriting either destination.

    Roll back the media link when publishing metadata fails. Both staged files
    must be on the destination filesystem; callers retain staging ownership.
    """
    os.link(staged, out)
    try:
        os.link(staged.with_suffix(".json"), out.with_suffix(".json"))
    except OSError:
        out.unlink()
        raise


def export_swing(
    root: Path,
    view: str,
    out: Path,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda done, total: None,
    drawings: DrawingLayer | None = None,
) -> dict[str, Any]:
    """Export saved trim/crop and provenance, refusing existing output files.

    Work is staged beside the destination. Hard links publish without replacing
    an existing path, including a path created concurrently by another exporter.
    Cancellation/decode errors publish nothing. Publication of video and sidecar
    is recoverable on ordinary errors, not a cross-file power-loss transaction.
    """
    out = out.resolve()
    sidecar = out.with_suffix(".json")
    if out.suffix.lower() not in (".avi", ".mp4"):
        raise ValueError("Choose an AVI or MP4 output file")
    if out.exists() or sidecar.exists():
        raise FileExistsError(
            "Choose a new filename; video or provenance already exists"
        )
    media = load_session(root)
    original = media.view(view)
    if original.recording is None:
        raise ValueError("The original recording is unavailable")
    # No resized proxy and no implicit detector overlay in a clean swing export.
    source = replace(original, proxy=None, observations=None, observation_sets=None)
    edit = load_edits(root).views.get(view, ViewEdit())
    source_hash = _digest(original.recording, cancelled)
    with VideoReader(original.recording) as reader:
        if drawings is not None and (
            drawings.view != view
            or drawings.frames != reader.frame_count
            or (drawings.width, drawings.height) != (reader.width, reader.height)
        ):
            raise ValueError("Drawing layer does not match the original recording")
        clip = ClipRange(
            edit.first, edit.last if edit.last is not None else reader.frame_count - 1
        )
    rendering = ClipRendering(
        crop=edit.crop,
        clock=False,
        strict=True,
        cancelled=cancelled,
        progress=progress,
        drawings=drawings,
    )
    with TemporaryDirectory(prefix=".swing-export-", dir=out.parent) as temporary:
        staged = Path(temporary) / out.name
        result = export_clip(source, clip, staged, rendering=rendering)
        result.update(
            schema_version="swing-video/1.0.0",
            file=str(out),
            source=str(original.recording),
            source_sha256=source_hash,
            edit=edit.model_dump(mode="json"),
            drawings=drawings.model_dump(mode="json") if drawings else None,
            padding={
                "right": edit.crop.width % 2 if edit.crop else 0,
                "bottom": edit.crop.height % 2 if edit.crop else 0,
            },
        )
        staged_notes = staged.with_suffix(".json")
        write_document(staged_notes, result)
        if cancelled():
            raise InterruptedError("Swing export cancelled")
        publish_export(staged, out)
    return result
