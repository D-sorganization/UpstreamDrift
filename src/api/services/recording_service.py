"""Recording persistence and export service (issue #7451).

Persists finalized ``GenericPhysicsRecorder`` sessions to disk under
``output/recordings/<id>/`` and exports them via the *same* serializers the
desktop app uses (``src.shared.python.data_io.export``). No parallel
serialization code paths exist here — exports call
``export_recording_all_formats`` / ``export_to_c3d`` directly, so web and
desktop downloads are byte-identical for the same recording.

Storage layout per recording::

    output/recordings/<id>/
        metadata.json   # engine, model, duration, frames, created, key order
        data.npz        # compressed numpy arrays (nested keys joined by "::")
        export.<ext>    # lazily created export artifacts

Design by Contract:
    - Recording ids must match ``RECORDING_ID_PATTERN`` (path-traversal safety).
    - ``persist`` postcondition: directory with metadata.json + data.npz exists.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.data_io.export import (
    export_recording_all_formats,
    export_to_c3d,
    get_available_export_formats,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

#: Safe recording-id pattern. Anything else is rejected before touching disk.
RECORDING_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

#: Separator used to flatten nested dict-of-array entries into npz keys.
_NESTED_KEY_SEP = "::"

_METADATA_FILE = "metadata.json"
_DATA_FILE = "data.npz"


def default_recordings_dir() -> Path:
    """Return the canonical recordings directory (``<repo>/output/recordings``).

    Mirrors the project ``output/`` convention used by the dataset and
    data-explorer routes.
    """
    return Path(__file__).parent.parent.parent.parent / "output" / "recordings"


def exportable_formats() -> dict[str, dict[str, Any]]:
    """Formats currently exportable, derived from the desktop registry.

    Single source of truth: ``get_available_export_formats`` from
    ``src.shared.python.data_io.export`` — the exact function the desktop
    dashboard window uses to populate its Export tab. Availability flags in
    that registry already probe optional dependency importability
    (scipy / h5py / ezc3d), so this enumeration is honest by construction.
    """
    return get_available_export_formats()


class RecordingNotFoundError(KeyError):
    """Raised when a recording id does not exist on disk."""


class InvalidRecordingIdError(ValueError):
    """Raised when a recording id fails the safety pattern."""


def validate_recording_id(recording_id: str) -> str:
    """Validate a recording id against the safe pattern.

    Raises:
        InvalidRecordingIdError: if the id is empty or contains characters
            outside ``[A-Za-z0-9_-]`` (path-traversal safety).
    """
    if not recording_id or not RECORDING_ID_PATTERN.match(recording_id):
        raise InvalidRecordingIdError(
            f"Invalid recording id {recording_id!r}: must match "
            f"{RECORDING_ID_PATTERN.pattern}"
        )
    return recording_id


def _is_json_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float))  # noqa: UP038


class RecordingStore:
    """Disk-backed store for persisted simulation recordings."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize the store.

        Args:
            base_dir: Root directory for recordings. Defaults to
                ``<repo>/output/recordings``.
        """
        self.base_dir = base_dir if base_dir is not None else default_recordings_dir()

    def _recording_dir(self, recording_id: str, *, must_exist: bool) -> Path:
        validate_recording_id(recording_id)
        path = self.base_dir / recording_id
        if must_exist and not (path / _METADATA_FILE).is_file():
            raise RecordingNotFoundError(recording_id)
        return path

    # ── Persistence ────────────────────────────────────────────────

    def persist(
        self,
        data_dict: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a recorder data dictionary to disk.

        Args:
            data_dict: Output of ``GenericPhysicsRecorder.get_data_dict()``.
            metadata: Session context (engine, model, duration).

        Returns:
            The generated recording id.
        """
        if not isinstance(data_dict, dict) or not data_dict:
            raise ValueError("data_dict must be a non-empty dict")

        recording_id = (
            f"rec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )
        rec_dir = self._recording_dir(recording_id, must_exist=False)
        rec_dir.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        entries: list[dict[str, Any]] = []
        scalars: dict[str, Any] = {}

        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
                entries.append({"key": key, "kind": "array"})
            elif isinstance(value, dict):
                sub_entries = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        arrays[f"{key}{_NESTED_KEY_SEP}{sub_key}"] = sub_value
                        sub_entries[str(sub_key)] = (
                            "int" if isinstance(sub_key, int) else "str"
                        )
                entries.append({"key": key, "kind": "nested", "subkeys": sub_entries})
            elif _is_json_scalar(value):
                scalars[key] = value
                entries.append({"key": key, "kind": "scalar"})
            else:
                logger.debug(
                    "Skipping non-persistable recording entry %r (%s)",
                    key,
                    type(value).__name__,
                )

        np.savez_compressed(rec_dir / _DATA_FILE, **arrays)

        frames = int(scalars.get("num_frames", 0)) or self._frames_from(data_dict)
        duration = self._duration_from(data_dict, metadata)
        meta = {
            "id": recording_id,
            "engine": (metadata or {}).get("engine"),
            "model": (metadata or {}).get("model") or scalars.get("model_name"),
            "duration": duration,
            "frames": frames,
            "created": datetime.now(timezone.utc).isoformat(),
            "entries": entries,
            "scalars": scalars,
        }
        (rec_dir / _METADATA_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("Persisted recording %s (%d frames)", recording_id, frames)
        return recording_id

    @staticmethod
    def _frames_from(data_dict: dict[str, Any]) -> int:
        times = data_dict.get("times")
        if isinstance(times, np.ndarray):
            return int(times.shape[0])
        return 0

    @staticmethod
    def _duration_from(
        data_dict: dict[str, Any], metadata: dict[str, Any] | None
    ) -> float | None:
        if metadata and metadata.get("duration") is not None:
            return float(metadata["duration"])
        times = data_dict.get("times")
        if isinstance(times, np.ndarray) and times.size > 1:
            return float(times[-1] - times[0])
        return None

    # ── Querying ──────────────────────────────────────────────────

    def get_metadata(self, recording_id: str) -> dict[str, Any]:
        """Return public metadata for a recording (without internal entries)."""
        rec_dir = self._recording_dir(recording_id, must_exist=True)
        raw = json.loads((rec_dir / _METADATA_FILE).read_text(encoding="utf-8"))
        return {
            key: raw.get(key)
            for key in ("id", "engine", "model", "duration", "frames", "created")
        }

    def list_recordings(self) -> list[dict[str, Any]]:
        """List all persisted recordings, newest first."""
        if not self.base_dir.is_dir():
            return []
        results = []
        for child in sorted(self.base_dir.iterdir(), reverse=True):
            if not child.is_dir() or not RECORDING_ID_PATTERN.match(child.name):
                continue
            try:
                results.append(self.get_metadata(child.name))
            except (RecordingNotFoundError, json.JSONDecodeError, OSError):
                logger.warning("Skipping unreadable recording dir %s", child)
        return results

    def delete(self, recording_id: str) -> None:
        """Delete a recording directory and all export artifacts."""
        rec_dir = self._recording_dir(recording_id, must_exist=True)
        shutil.rmtree(rec_dir)
        logger.info("Deleted recording %s", recording_id)

    # ── Data reconstruction ───────────────────────────────────────

    def load_data(self, recording_id: str) -> dict[str, Any]:
        """Reconstruct the recorder data dictionary from disk.

        Restores the exact key order and key types recorded at persist time so
        that exports of the reloaded dict are byte-identical to exports of the
        original (golden-test guarantee).
        """
        rec_dir = self._recording_dir(recording_id, must_exist=True)
        raw = json.loads((rec_dir / _METADATA_FILE).read_text(encoding="utf-8"))
        with np.load(rec_dir / _DATA_FILE, allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}

        data: dict[str, Any] = {}
        for entry in raw.get("entries", []):
            key = entry["key"]
            kind = entry["kind"]
            if kind == "array":
                if key in arrays:
                    data[key] = arrays[key]
            elif kind == "nested":
                nested: dict[Any, np.ndarray] = {}
                for sub_key_str, sub_type in entry.get("subkeys", {}).items():
                    flat = f"{key}{_NESTED_KEY_SEP}{sub_key_str}"
                    if flat in arrays:
                        sub_key: Any = (
                            int(sub_key_str) if sub_type == "int" else sub_key_str
                        )
                        nested[sub_key] = arrays[flat]
                data[key] = nested
            elif kind == "scalar":
                data[key] = raw.get("scalars", {}).get(key)
        return data

    # ── Export (shared desktop serializers) ───────────────────────

    def export(self, recording_id: str, export_format: str) -> Path:
        """Export a recording via the desktop serializers.

        Args:
            recording_id: Persisted recording id.
            export_format: One of the keys of ``exportable_formats()``.

        Returns:
            Path to the exported artifact inside the recording directory.

        Raises:
            ValueError: if the format is unknown or its dependency is missing.
            RecordingNotFoundError: if the recording does not exist.
            RuntimeError: if the underlying exporter reports failure.
        """
        formats = exportable_formats()
        info = formats.get(export_format)
        if info is None:
            raise ValueError(
                f"Unknown export format {export_format!r}. "
                f"Supported: {', '.join(sorted(formats))}"
            )
        if not info["available"]:
            raise ValueError(
                f"Export format {export_format!r} is unavailable: "
                "optional dependency not installed"
            )

        rec_dir = self._recording_dir(recording_id, must_exist=True)
        output_path = rec_dir / f"export{info['extension']}"
        if output_path.is_file():
            return output_path

        data = self.load_data(recording_id)
        if export_format == "c3d":
            self._export_c3d(output_path, data)
            return output_path

        # Same call path as the desktop Export tab (window.export_data →
        # export_recording_all_formats) — guarantees byte parity.
        results = export_recording_all_formats(
            str(rec_dir / "export"), data, formats=[export_format]
        )
        if not results.get(export_format):
            raise RuntimeError(f"Export to {export_format!r} failed")
        if not output_path.is_file():
            raise RuntimeError(
                f"Exporter reported success but {output_path.name} is missing"
            )
        return output_path

    @staticmethod
    def _export_c3d(output_path: Path, data: dict[str, Any]) -> None:
        times = data.get("times")
        positions = data.get("joint_positions")
        if not isinstance(times, np.ndarray) or not isinstance(positions, np.ndarray):
            raise ValueError("Recording lacks times/joint_positions for C3D export")
        if times.size > 1:
            dt = float(np.mean(np.diff(times)))
            frame_rate = 1.0 / dt if dt > 0 else 60.0
        else:
            frame_rate = 60.0
        joint_names = [f"joint_{i}" for i in range(positions.shape[1])]
        success = export_to_c3d(
            str(output_path),
            times,
            positions,
            joint_names,
            frame_rate=frame_rate,
        )
        if not success:
            raise RuntimeError("Export to 'c3d' failed")


__all__ = [
    "RECORDING_ID_PATTERN",
    "InvalidRecordingIdError",
    "RecordingNotFoundError",
    "RecordingStore",
    "default_recordings_dir",
    "exportable_formats",
    "validate_recording_id",
]
