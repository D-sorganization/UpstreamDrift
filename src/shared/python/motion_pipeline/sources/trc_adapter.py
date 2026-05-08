"""OpenSim TRC marker-trajectory adapter.

The Track Row Column (TRC) text format ships with OpenSim and is the most
common export from Theia3D and OpenCap. Layout::

    PathFileType  4  (X/Y/Z)  <filename>
    DataRate  CameraRate  NumFrames  NumMarkers  Units  OrigDataRate  OrigDataStartFrame  OrigNumFrames
    <values...>
    Frame#  Time  Marker1                Marker2  ...
            X1  Y1  Z1                  X2  Y2  Z2
            <data rows>
"""

from __future__ import annotations

from pathlib import Path

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter

_MM_TO_M = 0.001


@register_adapter
class TRCAdapter(MocapSourceAdapter):
    """OpenSim TRC marker file adapter."""

    format_name = "trc"
    file_extensions = (".trc",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.file_extensions:
            return False
        try:
            with open(p, encoding="utf-8") as f:
                first = f.readline().strip().lower()
            return first.startswith("pathfiletype")
        except (OSError, UnicodeDecodeError):
            return False

    def _parse_header(self, lines: list[str]) -> dict[str, object]:
        # Line 0: PathFileType 4 (X/Y/Z) <filename>
        # Line 1: header keys
        # Line 2: header values
        # Line 3: Frame#  Time  Marker1  Marker2 ...
        # Line 4: tab tab X1 Y1 Z1 X2 Y2 Z2 ...
        if len(lines) < 5:
            raise ValueError("TRC file truncated; need at least 5 header rows")
        keys = lines[1].split()
        values = lines[2].split()
        if len(values) < len(keys):
            values.extend([""] * (len(keys) - len(values)))
        header = dict(zip(keys, values, strict=False))
        marker_tokens = lines[3].split("\t")
        # Strip leading "Frame#" and "Time" cells, then drop empty cells
        marker_names = [t.strip() for t in marker_tokens[2:] if t.strip()]
        return {"header": header, "marker_names": marker_names}

    def metadata(self, path: Path) -> SourceMetadata:
        p = Path(path)
        lines = p.read_text(encoding="utf-8").splitlines()
        info = self._parse_header(lines)
        header: dict = info["header"]  # type: ignore[assignment]
        try:
            fps = float(header.get("DataRate", 0.0))
        except (TypeError, ValueError):
            fps = 0.0
        if fps <= 0:
            try:
                fps = float(header.get("CameraRate", 30.0)) or 30.0
            except (TypeError, ValueError):
                fps = 30.0
        try:
            num_frames = int(header.get("NumFrames", 0))
        except (TypeError, ValueError):
            num_frames = 0
        units_str = str(header.get("Units", "mm")).strip().lower()
        unit_system = "millimeters" if units_str.startswith("mm") else "meters"
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=num_frames,
            unit_system=unit_system,  # type: ignore[arg-type]
            marker_set_name=None,
            notes=f"markers={len(info['marker_names'])}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> MarkerTrajectory:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"TRC file not found: {p}")
        lines = p.read_text(encoding="utf-8").splitlines()
        info = self._parse_header(lines)
        marker_names: list[str] = info["marker_names"]  # type: ignore[assignment]
        header: dict = info["header"]  # type: ignore[assignment]
        units_str = str(header.get("Units", "mm")).strip().lower()
        scale = _MM_TO_M if units_str.startswith("mm") else 1.0

        frames: list[MarkerFrame] = []
        # Data starts at line 5 (after 2 header lines, marker name line, axis line)
        for line_idx, raw in enumerate(lines[5:], start=5):
            stripped = raw.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if len(tokens) < 2:
                continue
            try:
                frame_idx = int(float(tokens[0]))
                t = float(tokens[1])
            except ValueError as e:
                raise ValueError(
                    f"TRC line {line_idx} has invalid frame/time: {stripped!r}"
                ) from e
            coord_tokens = tokens[2:]
            markers: dict[str, Marker] = {}
            for m_i, name in enumerate(marker_names):
                base = m_i * 3
                if base + 2 >= len(coord_tokens):
                    break
                try:
                    x = float(coord_tokens[base]) * scale
                    y = float(coord_tokens[base + 1]) * scale
                    z = float(coord_tokens[base + 2]) * scale
                except ValueError:
                    # Treat unparseable as occluded
                    continue
                markers[name] = Marker(name=name, x=x, y=y, z=z)
            frames.append(
                MarkerFrame(timestamp=t, markers=markers, frame_index=frame_idx)
            )
        if not frames:
            raise ValueError(f"TRC file {p} has no data rows")
        return MarkerTrajectory(
            id=f"trc-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={"source_file": str(p), "units": units_str},
        )
