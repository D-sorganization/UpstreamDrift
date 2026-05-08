"""Generic CSV motion adapter.

Expected columns:

- ``frame`` (int) and ``timestamp`` (float, seconds)
- triplets ``x_<joint>``, ``y_<joint>``, ``z_<joint>`` for each joint

Outputs a :class:`KeypointSequence` with schema ``custom``. A simple
heuristic guesses delimiter (``,`` or ``;``) from the header line.
"""

from __future__ import annotations

import csv
import io
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


def _sniff_delimiter(header: str) -> str:
    if header.count(";") > header.count(","):
        return ";"
    return ","


@register_adapter
class CSVAdapter(MocapSourceAdapter):
    """Generic CSV keypoint trajectory adapter."""

    format_name = "csv"
    file_extensions = (".csv",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.file_extensions:
            return False
        try:
            with open(p, encoding="utf-8", newline="") as f:
                header = f.readline()
        except (OSError, UnicodeDecodeError):
            return False
        if not header:
            return False
        delim = _sniff_delimiter(header)
        cols = [c.strip().lower() for c in header.split(delim)]
        return (
            "frame" in cols
            and "timestamp" in cols
            and any(c.startswith("x_") for c in cols)
        )

    def _read_rows(self, path: Path) -> tuple[list[str], list[dict]]:
        text = Path(path).read_text(encoding="utf-8")
        first_line = text.splitlines()[0] if text else ""
        delim = _sniff_delimiter(first_line)
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        rows = list(reader)
        return list(reader.fieldnames or []), rows

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        fields, rows = self._read_rows(p)
        fps = 30.0
        if len(rows) >= 2:
            try:
                t0 = float(rows[0]["timestamp"])
                t1 = float(rows[1]["timestamp"])
                if t1 > t0:
                    fps = 1.0 / (t1 - t0)
            except (KeyError, ValueError):
                pass
        joints = sorted({f[2:] for f in fields if f.lower().startswith("x_")})
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=len(rows),
            unit_system="meters",
            keypoint_schema="custom",
            notes=f"joints={len(joints)}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> KeypointSequence:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {p}")
        fields, rows = self._read_rows(p)
        if not rows:
            raise ValueError(f"CSV {p} has no data rows")

        joints = sorted({f[2:] for f in fields if f.lower().startswith("x_")})
        if not joints:
            raise ValueError(
                f"CSV {p} has no x_<joint> columns; cannot infer joint set"
            )

        frames: list[KeypointFrame] = []
        for idx, row in enumerate(rows):
            try:
                t = float(row["timestamp"])
                fi = int(float(row["frame"]))
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"CSV row {idx} missing/invalid frame or timestamp"
                ) from e
            kps: list[Keypoint] = []
            for joint in joints:
                xk, yk, zk = f"x_{joint}", f"y_{joint}", f"z_{joint}"
                try:
                    x = float(row[xk])
                    y = float(row[yk])
                except (KeyError, ValueError):
                    continue
                z = None
                if zk in row and row[zk] not in ("", None):
                    try:
                        z = float(row[zk])
                    except ValueError:
                        z = None
                kps.append(Keypoint(x=x, y=y, z=z, confidence=1.0, name=joint))
            if not kps:
                continue
            frames.append(
                KeypointFrame(
                    timestamp=t,
                    keypoints=kps,
                    schema_name="custom",
                    frame_index=fi,
                )
            )
        if not frames:
            raise ValueError(f"CSV {p} produced no usable frames")
        return KeypointSequence(
            id=f"csv-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={"source_file": str(p), "joints": joints},
        )
