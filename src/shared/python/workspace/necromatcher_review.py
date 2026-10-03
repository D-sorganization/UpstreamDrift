"""Read-only source-frame review shared by Necromatcher web and desktop."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from src.shared.python.core.contracts.exceptions import StateError
from .necromatcher import NecromatcherLibrary


class CaptureReview:
    """Keep one verified capture archive open; never substitute missing frames."""

    def __init__(self, library: NecromatcherLibrary, capture_id: str) -> None:
        asset = library.load_asset(capture_id)
        if asset.kind != "image_capture":
            raise ValueError("Source review requires an image capture")
        self.capture_id = capture_id
        self._path = Path(asset.path)
        self._signature = self._file_signature()
        self._archive = ZipFile(self._path, "r")
        try:
            self._receipt = json.loads(self._archive.read("receipt.json"))
            with self._archive.open("observations.jsonl") as observations:
                self._rows = [json.loads(line) for line in observations]
            if len(self._rows) != self._receipt["frame_count"] or not self._rows:
                raise ValueError("Capture frame count disagrees with receipt")
        except (KeyError, ValueError, OSError):
            self.close()
            raise

    def _file_signature(self) -> tuple[int, int]:
        stat = self._path.stat()
        return stat.st_size, stat.st_mtime_ns

    def _check_index(self, index: int) -> None:
        if self._file_signature() != self._signature:
            raise StateError("Capture archive changed; reopen and verify this version")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self._rows)
        ):
            raise IndexError("Source frame index is outside the capture")

    @property
    def frame_count(self) -> int:
        return len(self._rows)

    def frame(self, index: int) -> dict[str, Any]:
        """Return a detached source-clock observation for the selected frame."""
        self._check_index(index)
        row = deepcopy(self._rows[index])
        return {
            "capture_id": self.capture_id,
            "frame_index": index,
            "frame_count": self.frame_count,
            "image_width": self._receipt["source"]["width_px"],
            "image_height": self._receipt["source"]["height_px"],
            "frame": row["frame"],
            "observation": row["observation"],
        }

    def image(self, index: int) -> bytes:
        """Read original PNG bytes with ZIP CRC verification and no extraction."""
        self._check_index(index)
        return self._archive.read(self._rows[index]["image"])

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> CaptureReview:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
