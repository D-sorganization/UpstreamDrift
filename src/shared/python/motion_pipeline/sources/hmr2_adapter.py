"""4D-Humans / HMR 2.0 sidecar output adapter (monocular 3D SMPL joints).

Reads the ``joints3d.csv`` artifact written by the HMR2 sidecar
(:mod:`src.tools.hmr2_sidecar.run_hmr2`): columns ``frame,time``
followed by ``<joint>_x,<joint>_y,<joint>_z`` triplets for the 22 SMPL
body joints, positions in meters and ``time`` in seconds.

Sniffing is conservative by design so generic trajectory CSVs are left
to the generic ``CSVAdapter``: a file is claimed only when its header
matches the sidecar column contract exactly, or when it has the
``frame,time`` + ``*_x/_y/_z`` triplet shape *and* a sibling
``metadata.json`` declares the 4D-Humans tool.

Emits a 3D :class:`KeypointSequence` with ``schema_name="custom"``
(SMPL is not a member of the ``SchemaName`` literal), joint names
preserved verbatim, and timestamps taken from the ``time`` column.
The ``4D-Humans`` package itself is never imported.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter

#: The 22 SMPL body joints, in canonical SMPL kinematic-tree order.
#: Duplicated from ``src.tools.hmr2_sidecar.run_hmr2.SMPL_BODY_JOINTS``
#: because adapters may not import ``src.tools`` (Law-of-Demeter gate in
#: ``tests/unit/motion_pipeline/sources/test_lod.py``); the two copies
#: are kept in sync by
#: ``tests/unit/motion_pipeline/sources/test_hmr2_adapter.py``.
SMPL_BODY_JOINTS: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

_AXES = ("x", "y", "z")
_EXPECTED_HEADER = [
    "frame",
    "time",
    *(f"{joint}_{axis}" for joint in SMPL_BODY_JOINTS for axis in _AXES),
]


def _read_header(path: Path) -> list[str] | None:
    """Return the lower-cased, stripped header cells of *path*, or None."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            header = f.readline()
    except (OSError, UnicodeDecodeError):
        return None
    if not header.strip():
        return None
    return [c.strip().lower() for c in header.split(",")]


def _sidecar_metadata(path: Path) -> dict | None:
    """Return the sibling ``metadata.json`` payload if present and valid."""
    meta_path = Path(path).parent / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


@register_adapter
class HMR2Adapter(MocapSourceAdapter):
    """Adapter for the HMR2 sidecar's ``joints3d.csv`` artifact.

    Postconditions (verified via :meth:`load_checked`): at least one
    frame, monotonically non-decreasing ``time`` timestamps, finite
    coordinates (keypoints with non-finite values are dropped; frames
    with no finite keypoints are skipped), and every keypoint carries a
    ``z`` coordinate (the sequence is fully 3D, in meters).
    """

    format_name = "hmr2"
    file_extensions = (".csv",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Return True iff *path* looks like HMR2 sidecar joints3d output.

        Conservative: either the header equals the sidecar column
        contract exactly, or the header has the ``frame,time`` +
        ``*_x/_y/_z`` triplet shape and a sibling ``metadata.json``
        names the 4D-Humans tool. Plain trajectory CSVs (which use
        ``timestamp`` and ``x_<joint>`` columns) are never claimed.
        """
        p = Path(path)
        if p.suffix.lower() not in cls.file_extensions or not p.is_file():
            return False
        cols = _read_header(p)
        if cols is None:
            return False
        if cols == _EXPECTED_HEADER:
            return True
        if not cls._has_joint_triplet_shape(cols):
            return False
        meta = _sidecar_metadata(p)
        if meta is None:
            return False
        tool = str(meta.get("tool", "")).lower()
        return "4d-humans" in tool or "hmr2" in tool

    @staticmethod
    def _has_joint_triplet_shape(cols: list[str]) -> bool:
        """Check for ``frame,time`` then complete ``<joint>_x/_y/_z`` triplets."""
        if len(cols) < 5 or cols[0] != "frame" or cols[1] != "time":
            return False
        joint_cols = cols[2:]
        if len(joint_cols) % 3 != 0:
            return False
        for i in range(0, len(joint_cols), 3):
            triplet = joint_cols[i : i + 3]
            stems = {c[:-2] for c in triplet}
            suffixes = [c[-2:] for c in triplet]
            if len(stems) != 1 or suffixes != ["_x", "_y", "_z"]:
                return False
        return True

    # ------------------------------------------------------------------
    # Parsing

    @staticmethod
    def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
        """Return (joint_names, data_rows) for *path*.

        Raises :class:`ValueError` when the header does not carry the
        expected triplet shape.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"HMR2 joints3d CSV not found: {p}")
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = [c.strip() for c in reader.fieldnames or []]
            rows = list(reader)
        lower = [c.lower() for c in fields]
        if not HMR2Adapter._has_joint_triplet_shape(lower):
            raise ValueError(
                f"HMR2 joints3d CSV {p} header does not match the sidecar "
                "contract (frame,time then <joint>_x/_y/_z triplets)"
            )
        joints = [fields[i][:-2] for i in range(2, len(fields), 3)]
        return joints, rows

    def metadata(self, path: Path) -> SourceMetadata:
        """Return metadata for *path* (fps inferred from the time column)."""
        p = Path(path)
        joints, rows = self._read_rows(p)
        fps = 30.0
        sidecar_meta = _sidecar_metadata(p)
        if sidecar_meta is not None and isinstance(
            sidecar_meta.get("fps"), int | float
        ):
            reported = float(sidecar_meta["fps"])
            if math.isfinite(reported) and reported > 0:
                fps = reported
        elif len(rows) >= 2:
            try:
                dt = float(rows[1]["time"]) - float(rows[0]["time"])
                if dt > 0:
                    fps = 1.0 / dt
            except (KeyError, TypeError, ValueError):
                pass
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=len(rows),
            unit_system="meters",
            keypoint_schema="custom",
            notes=f"SMPL body joints ({len(joints)}); 4D-Humans/HMR2 sidecar",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> KeypointSequence:
        """Load *path* into a 3D custom-schema :class:`KeypointSequence`.

        Keypoints with non-finite coordinates are dropped; frames with
        no finite keypoints are skipped. Raises :class:`ValueError`
        with file context for malformed inputs.
        """
        p = Path(path)
        joints, rows = self._read_rows(p)
        if not rows:
            raise ValueError(f"HMR2 joints3d CSV {p} has no data rows")

        frames: list[KeypointFrame] = []
        for idx, row in enumerate(rows):
            try:
                t = float(row["time"])
                fi = int(float(row["frame"]))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    f"HMR2 joints3d CSV {p} row {idx} has a missing or "
                    f"invalid frame/time cell: {e}"
                ) from e
            keypoints: list[Keypoint] = []
            for joint in joints:
                try:
                    coords = [float(row[f"{joint}_{axis}"]) for axis in _AXES]
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(
                        f"HMR2 joints3d CSV {p} row {idx} has a missing or "
                        f"non-numeric coordinate for joint {joint!r}: {e}"
                    ) from e
                if not all(math.isfinite(c) for c in coords):
                    continue
                keypoints.append(
                    Keypoint(
                        x=coords[0],
                        y=coords[1],
                        z=coords[2],
                        confidence=1.0,
                        name=joint,
                    )
                )
            if not keypoints:
                continue
            frames.append(
                KeypointFrame(
                    timestamp=t,
                    keypoints=keypoints,
                    schema_name="custom",
                    frame_index=fi,
                )
            )
        if not frames:
            raise ValueError(
                f"HMR2 joints3d CSV {p} produced no usable frames "
                "(all keypoints missing or non-finite)"
            )
        return KeypointSequence(
            id=f"hmr2-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={
                "source_file": str(p),
                "joints": joints,
                "unit_system": "meters",
                "smpl_body_joints": list(SMPL_BODY_JOINTS),
            },
        )
