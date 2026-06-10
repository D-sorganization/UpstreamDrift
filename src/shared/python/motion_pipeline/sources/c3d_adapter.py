"""C3D marker-trajectory adapter.

Primary parser is the Rust ``upstream_mocap_io`` wheel (see
``rust_core/upstream-mocap-io/``); if that wheel is not available we fall
back to ``ezc3d`` (an optional Python dependency). If neither is installed
the adapter still imports cleanly but reports ``supports() == False``.

Issue #5213 — native C3D / BVH / TRC adapters via Rust.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
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
from src.shared.python.motion_pipeline.sources._marker_coordinates import (
    has_nan_coordinate,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter

try:  # pragma: no cover - native wheel may not be installed in dev clones
    import upstream_mocap_io as _rust_io  # type: ignore[import-not-found]

    _HAS_RUST = True
except ImportError:  # pragma: no cover
    _rust_io = None  # type: ignore[assignment]
    _HAS_RUST = False

try:  # pragma: no cover - exercised only when ezc3d is installed
    import ezc3d as _ezc3d  # type: ignore[import-not-found]

    _HAS_EZC3D = True
except ImportError:  # pragma: no cover
    _ezc3d = None  # type: ignore[assignment]
    _HAS_EZC3D = False

_HAS_C3D_BACKEND = _HAS_RUST or _HAS_EZC3D


def _normalize_rust_events(raw_events: object) -> list[dict[str, object]]:
    """Return JSON-serializable C3D events from the Rust parser payload."""
    if not isinstance(raw_events, Iterable):
        return []
    events: list[dict[str, object]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        label = str(raw_event.get("label", "")).strip()
        if not label:
            continue
        time_s = float(raw_event.get("time_s", 0.0))
        if not math.isfinite(time_s):
            continue
        events.append(
            {
                "label": label,
                "context": str(raw_event.get("context", "")).strip(),
                "time_s": time_s,
            }
        )
    return events


def _normalize_rust_analog(raw_analog: object) -> dict[str, object] | None:
    """Return JSON-serializable analog metadata from the Rust parser payload."""
    if not isinstance(raw_analog, Mapping):
        return None
    return {
        "labels": [str(label).strip() for label in raw_analog.get("labels", [])],
        "units": [str(unit).strip() for unit in raw_analog.get("units", [])],
        "n_frames": int(raw_analog.get("n_frames", 0)),
        "samples_per_frame": int(raw_analog.get("samples_per_frame", 0)),
        "n_channels": int(raw_analog.get("n_channels", 0)),
        "rate": float(raw_analog.get("rate", 0.0)),
    }


def _normalize_rust_force_platforms(
    raw_platforms: object,
) -> list[dict[str, object]]:
    """Return JSON-serializable force-platform metadata from Rust payloads."""
    if not isinstance(raw_platforms, Iterable):
        return []
    platforms: list[dict[str, object]] = []
    for raw_platform in raw_platforms:
        if not isinstance(raw_platform, Mapping):
            continue
        platforms.append(
            {
                "type": int(raw_platform.get("type", 0)),
                "channels": [
                    int(channel) for channel in raw_platform.get("channels", [])
                ],
                "corners": [
                    [float(coord) for coord in corner]
                    for corner in raw_platform.get("corners", [])
                ],
                "origin": [
                    float(coord) for coord in raw_platform.get("origin", [0, 0, 0])
                ],
            }
        )
    return platforms


class C3DAdapter(MocapSourceAdapter):
    """C3D file adapter using ezc3d."""

    format_name = "c3d"
    file_extensions = (".c3d",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        if not _HAS_C3D_BACKEND:
            return False
        p = Path(path)
        return p.suffix.lower() in cls.file_extensions and p.exists()

    def metadata(self, path: Path) -> SourceMetadata:  # pragma: no cover
        if _HAS_RUST:
            try:
                r = _rust_io.parse_c3d(str(path))
            except (OSError, ValueError) as e:
                raise AdapterContractError(
                    f"Failed to read C3D metadata from {path}: {e}"
                ) from e
            units = str(r["units"]).strip().lower() or "mm"
            unit_system = "millimeters" if units.startswith("mm") else "meters"
            return SourceMetadata(
                format_name=self.format_name,
                fps=float(r["fps"]),
                frame_count=int(r["n_frames"]),
                unit_system=unit_system,  # type: ignore[arg-type]
                marker_set_name=None,
            )
        if not _HAS_EZC3D:
            raise RuntimeError(
                "No C3D backend available (install upstream-mocap-io or ezc3d)"
            )
        try:
            c = _ezc3d.c3d(str(path))
        except OSError as e:
            raise AdapterContractError(
                f"Failed to read C3D metadata from {path}: {e}"
            ) from e
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
        if not _HAS_C3D_BACKEND:
            raise RuntimeError(
                "No C3D backend available (install upstream-mocap-io or ezc3d)"
            )
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"C3D file not found: {p}")
        if _HAS_RUST:
            return self._load_via_rust(p, calibration)
        return self._load_via_ezc3d(p, calibration)

    def _load_via_rust(  # pragma: no cover
        self,
        p: Path,
        calibration: Calibration | None,
    ) -> MarkerTrajectory:
        try:
            r = _rust_io.parse_c3d(str(p))
        except (OSError, ValueError) as e:
            raise AdapterContractError(f"Failed to load C3D file {p}: {e}") from e
        labels: list[str] = list(r["labels"])
        positions = r["positions"]  # (n_frames, n_markers * 3) float32, meters
        n_frames = int(r["n_frames"])
        fps = float(r["fps"]) or 30.0
        units = (str(r["units"]).strip().lower()) or "mm"
        events = _normalize_rust_events(r.get("events", []))
        analog = _normalize_rust_analog(r.get("analog"))
        force_platforms = _normalize_rust_force_platforms(r.get("force_platforms", []))
        # Bypass pydantic validation on the inner Marker / MarkerFrame
        # objects via model_construct: the Rust pass already guaranteed
        # finiteness + occlusion handling. The outer MarkerTrajectory is
        # still validated by the regular constructor below. This is what
        # buys us the ≥10× speedup vs the pure-Python adapter loop.
        marker_ctor = Marker.model_construct
        frame_ctor = MarkerFrame.model_construct
        frames: list[MarkerFrame] = []
        for fi in range(n_frames):
            markers: dict[str, Marker] = {}
            row = positions[fi]
            for mi, name in enumerate(labels):
                base = mi * 3
                x = float(row[base])
                y = float(row[base + 1])
                z = float(row[base + 2])
                if has_nan_coordinate(x, y, z):
                    continue
                markers[name] = marker_ctor(
                    name=name, x=x, y=y, z=z, residual=None, occluded=False
                )
            frames.append(
                frame_ctor(timestamp=fi / fps, markers=markers, frame_index=fi)
            )
        if not frames:
            raise ValueError(f"C3D {p} produced no frames")
        return MarkerTrajectory(
            id=f"c3d-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={
                "source_file": str(p),
                "units": units,
                "events": events,
                "analog": analog,
                "force_platforms": force_platforms,
            },
        )

    def _load_via_ezc3d(  # pragma: no cover
        self,
        p: Path,
        calibration: Calibration | None,
    ) -> MarkerTrajectory:
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
                if has_nan_coordinate(x, y, z):
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


if _HAS_C3D_BACKEND:  # pragma: no cover
    register_adapter(C3DAdapter)
