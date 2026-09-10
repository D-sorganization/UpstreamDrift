"""Original calibration frames with capture identity and verified PNG bytes."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import UUID, uuid4

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from src.motion_capture.rig.documents import write_document

from ..swing_export import publish_export

FRAME_DIRECTORY = PurePosixPath("reference_calibration/frames")
MAX_PIXELS = 40_000_000
MAX_FRAME_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024


class ArchivedFrame(BaseModel):
    """A frame snapshot, not a mutable link to the current video or camera."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["capture-reference-frame/1"] = "capture-reference-frame/1"
    capture_id: str = Field(min_length=1, max_length=200)
    view: str = Field(min_length=1, max_length=200)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_index: int = Field(ge=0, strict=True)
    timestamp_s: float = Field(ge=0)
    source_label: str = Field(min_length=1, max_length=500)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


def _owned_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.parent != FRAME_DIRECTORY or path.suffix != ".png":
        raise ValueError("Use an archived reference frame path")
    try:
        UUID(path.stem)
    except ValueError as exc:
        raise ValueError("Use an archived reference frame path") from exc
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Reference frame path leaves its capture workspace")
    return target


def archive_frame(
    root: Path,
    frame: npt.NDArray[np.uint8],
    *,
    capture_id: str,
    view: str,
    frame_index: int,
    timestamp_s: float,
    source_label: str,
) -> ArchivedFrame:
    """Publish a separate lossless frame and sidecar using the existing exporter.

    Call off the GUI thread for large camera images. Publication never overwrites
    prior snapshots; markings are stored separately and never burn into this image.
    """
    if (
        frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or not 0 < frame.shape[0] * frame.shape[1] <= MAX_PIXELS
    ):
        raise ValueError("Use an original 8-bit BGR frame within the image limit")
    success, encoded = cv2.imencode(".png", frame)
    if not success or encoded.nbytes > MAX_FRAME_BYTES:
        raise ValueError("Cannot archive this reference image within the file limit")
    data = encoded.tobytes()
    relative = (FRAME_DIRECTORY / f"{uuid4()}.png").as_posix()
    record = ArchivedFrame(
        capture_id=capture_id,
        view=view,
        path=relative,
        sha256=hashlib.sha256(data).hexdigest(),
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        source_label=source_label,
        width=frame.shape[1],
        height=frame.shape[0],
    )
    out = _owned_path(root, relative)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".reference-frame-", dir=out.parent) as temporary:
        staged = Path(temporary) / out.name
        staged.write_bytes(data)
        write_document(staged.with_suffix(".json"), record.model_dump(mode="json"))
        publish_export(staged, out)
    return record


def read_frame_record(root: Path, relative: str, *, capture_id: str) -> ArchivedFrame:
    """Read bounded capture-owned frame metadata; pixels are verified by load_frame."""
    path = _owned_path(root, relative)
    with path.with_suffix(".json").open("rb") as stream:
        metadata = stream.read(MAX_METADATA_BYTES + 1)
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError("Reference frame metadata exceeds its limit")
    record = ArchivedFrame.model_validate_json(metadata)
    if record.capture_id != capture_id:
        raise ValueError("Reference frame belongs to another capture")
    if record.path != relative:
        raise ValueError("Reference frame metadata changed; select the original frame")
    return record


def _verified_frame_bytes(
    root: Path, relative: str, expected_sha256: str, *, capture_id: str
) -> tuple[bytes, ArchivedFrame]:
    """Verify the capture binding, exact archived bytes and bounded PNG dimensions."""
    path = _owned_path(root, relative)
    record = read_frame_record(root, relative, capture_id=capture_id)
    if record.sha256 != expected_sha256:
        raise ValueError("Reference frame metadata changed; select the original frame")
    with path.open("rb") as stream:
        data = stream.read(MAX_FRAME_BYTES + 1)
    if (
        len(data) > MAX_FRAME_BYTES
        or hashlib.sha256(data).hexdigest() != expected_sha256
    ):
        raise ValueError("Reference frame changed; select the original frame")
    if len(data) < 24 or data[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise ValueError("Reference frame is not an archived PNG")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (
        record.width,
        record.height,
    ) or not 0 < width * height <= MAX_PIXELS:
        raise ValueError("Reference frame dimensions changed or exceed the limit")
    return data, record


def verify_frame(
    root: Path, relative: str, expected_sha256: str, *, capture_id: str
) -> ArchivedFrame:
    """Check already-archived evidence without decompressing it during every save.

    Archive creation and point inspection validate image pixels. This check binds
    subsequent operations to those same bytes; it does not inspect image content.
    """
    _, record = _verified_frame_bytes(
        root, relative, expected_sha256, capture_id=capture_id
    )
    return record


def load_frame(
    root: Path, relative: str, expected_sha256: str, *, capture_id: str
) -> npt.NDArray[np.uint8]:
    """Verify identity and the exact saved bytes before allocating a decoded image."""
    data, record = _verified_frame_bytes(
        root, relative, expected_sha256, capture_id=capture_id
    )
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (record.height, record.width, 3):
        raise ValueError("Cannot decode the archived reference frame")
    return np.asarray(frame, dtype=np.uint8)
