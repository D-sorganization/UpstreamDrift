"""C3D marker-trajectory adapter (optional dependency: ``ezc3d``).

If ``ezc3d`` is not importable, this module skips registration entirely so
the rest of the framework keeps working. Hard ImportError at module load
is *not* raised.
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
    AdapterContractError,
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter

try:  # pragma: no cover - exercised only when ezc3d is installed
    import ezc3d as _ezc3d  # type: ignore[import-not-found]

    _HAS_EZC3D = True
except ImportError:  # pragma: no cover
    _ezc3d = None  # type: ignore[assignment]
    _HAS_EZC3D = False


class C3DAdapter(MocapSourceAdapter):
    """C3D file adapter using ezc3d."""

    format_name = "c3d"
    file_extensions = (".c3d",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        if not _HAS_EZC3D:
            return False
        p = Path(path)
        return p.suffix.lower() in cls.file_extensions and p.exists()

    def metadata(self, path: Path) -> SourceMetadata:  # pragma: no cover
        if not _HAS_EZC3D:
            raise RuntimeError("ezc3d is not installed; cannot read C3D metadata")
        try:
            c = _ezc3d.c3d(str(path))
        except OSError as e:
            raise AdapterContractError(f"Failed to read C3D metadata from {path}: {e}") from e
        params = c["parameters"]
        point = c["data"]["points"]
        fps = float(params["POINT"]["RATE"]["value"][0])
        frame_count = int(point.shape[2])
        units = (
            str(params["POINT"]["UNITS"]["value"][0]).strip().lower()
            if params["POINT"]["UNITS"]["value"]
            else "mm"
        )
        unit_system = "millimeters" if units.startswith("mm") else "meters"
        return SourceMetadata(
            format_name=self.format_name,
            fps=fps,
            frame_count=frame_count,
            unit_system=unit_system,  # type: ignore[arg-type]
            marker_set_name=None,
        )

    def load(  # pragma: no cover
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> MarkerTrajectory:
        if not _HAS_EZC3D:
            raise RuntimeError("ezc3d is not installed; cannot load C3D files")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"C3D file not found: {p}")
        try:
            c = _ezc3d.c3d(str(p))
        except OSError as e:
            raise AdapterContractError(f"Failed to load C3D file {p}: {e}") from e
        params = c["parameters"]
        points = c["data"]["points"]  # shape (4, N_markers, N_frames)
        labels_raw = params["POINT"]["LABELS"]["value"]
        labels = [str(label).strip() for label in labels_raw]
        fps = float(params["POINT"]["RATE"]["value"][0]) or 30.0
        units = (
            str(params["POINT"]["UNITS"]["value"][0]).strip().lower()
            if params["POINT"]["UNITS"]["value"]
            else "mm"
        )
        scale = 0.001 if units.startswith("mm") else 1.0
        n_frames = int(points.shape[2])
        frames: list[MarkerFrame] = []
        for fi in range(n_frames):
            markers: dict[str, Marker] = {}
            for mi, name in enumerate(labels):
                if mi >= points.shape[1]:
                    break
                x = float(points[0, mi, fi]) * scale
                y = float(points[1, mi, fi]) * scale
                z = float(points[2, mi, fi]) * scale
                # Skip occluded markers (NaN per ezc3d conventions)
                if any(val != val for val in (x, y, z)):  # NaN check
                    continue
                markers[name] = Marker(name=name, x=x, y=y, z=z)
            frames.append(
                MarkerFrame(timestamp=fi / fps, markers=markers, frame_index=fi)
            )
        if not frames:
            raise ValueError(f"C3D {p} produced no frames")
        return MarkerTrajectory(
            id=f"c3d-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={"source_file": str(p), "units": units},
        )


if _HAS_EZC3D:  # pragma: no cover
    register_adapter(C3DAdapter)