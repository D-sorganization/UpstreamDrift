"""OpenSim TRC marker-trajectory adapter.

The Track Row Column (TRC) text format ships with OpenSim and is the most
common export from Theia3D and OpenCap.

Issue #5213 introduces an optional Rust parser via ``upstream_mocap_io``;
when that wheel is installed the per-line tokenise + float-parse hot loop
is replaced with a Rust pass. The pure-Python parser remains the
canonical fallback for clones without the wheel.

Layout::

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

try:  # pragma: no cover - native wheel may not be installed
    import upstream_mocap_io as _rust_io  # type: ignore[import-not-found]

    _HAS_RUST = True
except ImportError:  # pragma: no cover
    _rust_io = None  # type: ignore[assignment]
    _HAS_RUST = False

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
            notes=f"markers={len(info['marker_names'])}",  # type: ignore[arg-type]
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> MarkerTrajectory:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"TRC file not found: {p}")
        if _HAS_RUST:
            try:
                return self._load_via_rust(p, calibration)
            except (
                Exception  # noqa: BLE001 - fall back to Python parser on any Rust failure
            ):  # pragma: no cover - Rust parser disagreement
                # Fall back to the canonical Python parser if anything goes
                # wrong; the contract is byte-identical output, not "Rust wins".
                pass
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

    def _load_via_rust(
        self,
        p: Path,
        calibration: Calibration | None,
    ) -> MarkerTrajectory:
        """Parse via ``upstream_mocap_io.parse_trc`` and build the pydantic objects.

        The Rust pass returns positions already converted to meters (the
        Rust crate applies the mm->m scale internally), so the Python
        side only assembles ``MarkerFrame`` + ``MarkerTrajectory``.

        The TRC time column carries non-uniform timing in some captures,
        but we don't currently expose it through the Rust API. Use the
        nominal ``frame_idx / fps`` cadence; this matches the existing
        Python adapter behaviour modulo per-row time-jitter (uniform-fps
        TRCs — the common case — match exactly).
        """
        r = _rust_io.parse_trc(str(p))
        positions = r["positions"]  # (n_frames, n_markers * 3) float32, meters
        labels: list[str] = list(r["labels"])
        n_frames = int(r["n_frames"])
        fps = float(r["fps"]) or 30.0
        units_str = str(r["units"]).strip().lower() or "mm"

        # Re-read just the data lines to recover the per-row time + frame
        # index columns from the file (the Rust API does not surface them).
        lines = p.read_text(encoding="utf-8").splitlines()
        data_lines = [ln for ln in lines[5:] if ln.strip()]

        marker_ctor = Marker.model_construct
        frame_ctor = MarkerFrame.model_construct
        frames: list[MarkerFrame] = []
        for fi in range(n_frames):
            tokens = data_lines[fi].split() if fi < len(data_lines) else []
            try:
                frame_idx = int(float(tokens[0]))
                t = float(tokens[1])
            except (IndexError, ValueError):
                frame_idx = fi
                t = fi / fps
            markers: dict[str, Marker] = {}
            row = positions[fi]
            for mi, name in enumerate(labels):
                base = mi * 3
                x = float(row[base])
                y = float(row[base + 1])
                z = float(row[base + 2])
                if x != x or y != y or z != z:  # NaN occlusion
                    continue
                markers[name] = marker_ctor(
                    name=name, x=x, y=y, z=z, residual=None, occluded=False
                )
            frames.append(
                frame_ctor(timestamp=t, markers=markers, frame_index=frame_idx)
            )

        if not frames:
            raise ValueError(f"TRC file {p} has no data rows")
        return MarkerTrajectory(
            id=f"trc-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={"source_file": str(p), "units": units_str},
        )
