"""Lossless coaching still export with the same source-coordinate renderer."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from src.motion_capture.coaching import DrawingLayer, render_layer
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import ViewEdit, load_edits

from .player import VideoReader
from .session import load_session
from .swing_export import publish_export


def publish_still(image: np.ndarray, out: Path, metadata: dict[str, Any]) -> None:
    """Publish a lossless analysis frame and sidecar without replacing files."""
    import cv2

    if out.suffix.lower() != ".png":
        raise ValueError("Choose a PNG filename")
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError("Choose a new filename for the image and sidecar")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Still image must be BGR uint8")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the reference image")
    with TemporaryDirectory(prefix=".coaching-still-", dir=out.parent) as temporary:
        staged = Path(temporary) / out.name
        staged.write_bytes(encoded.tobytes())
        write_document(staged.with_suffix(".json"), metadata)
        publish_export(staged, out)


def export_still(root: Path, layer: DrawingLayer, frame: int, out: Path) -> None:
    """Publish a new PNG plus portable references; retain original frame numbers."""
    if out.suffix.lower() != ".png":
        raise ValueError("Choose a PNG filename")
    sidecar = out.with_suffix(".json")
    if out.exists() or sidecar.exists():
        raise FileExistsError("Choose a new filename for the image and sidecar")
    source = load_session(root).view(layer.view).recording
    if source is None:
        raise ValueError("Original recording is unavailable")
    with VideoReader(source) as reader:
        if layer.frames != reader.frame_count or not 0 <= frame < layer.frames:
            raise ValueError("Drawing timeline does not match the source")
        image = reader.read(frame)
        if image is None:
            raise ValueError("Could not decode the selected frame")
        seconds = frame / reader.fps if reader.fps else None
    image = render_layer(image, layer, frame)
    edit = load_edits(root).views.get(layer.view, ViewEdit())
    if edit.crop:
        crop = edit.crop
        image = image[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
    publish_still(
        image,
        out,
        {
            "schema_version": "coaching-still/1.0.0",
            "source": str(source),
            "frame": frame,
            "source_seconds": seconds,
            "edit": edit.model_dump(mode="json"),
            "drawings": layer.model_dump(mode="json"),
        },
    )
