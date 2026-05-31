"""OpenCap session adapter.

OpenCap exports OpenSim-native session data, commonly including augmented
marker TRC files under ``OpenSimData/MarkerData``. This adapter resolves the
session output file, reuses the canonical TRC parser, and lifts the result into
``CanonicalObservations`` with marker names normalized to OpenSim marker sites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CanonicalObservationFrame,
    CanonicalObservations,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter
from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter


_SESSION_METADATA_FILENAMES = (
    "sessionMetadata.json",
    "session_metadata.json",
    "metadata.json",
)

_MARKER_FILE_TOKENS = (
    "augmented",
    "marker_trajectories",
    "markers",
)

_OPENSIM_MARKER_ALIASES = {
    "r_asis": "R.ASIS",
    "rasis": "R.ASIS",
    "r.asis": "R.ASIS",
    "l_asis": "L.ASIS",
    "lasis": "L.ASIS",
    "l.asis": "L.ASIS",
    "r_psis": "R.PSIS",
    "rpsis": "R.PSIS",
    "r.psis": "R.PSIS",
    "l_psis": "L.PSIS",
    "lpsis": "L.PSIS",
    "l.psis": "L.PSIS",
    "r_shoulder": "R.Acromium",
    "rshoulder": "R.Acromium",
    "r_acromion": "R.Acromium",
    "r.acromium": "R.Acromium",
    "l_shoulder": "L.Acromium",
    "lshoulder": "L.Acromium",
    "l_acromion": "L.Acromium",
    "l.acromium": "L.Acromium",
}


def normalize_opencap_marker_name(name: str) -> str:
    """Return the OpenSim marker-site name for an OpenCap marker label."""
    key = name.strip().lower()
    return _OPENSIM_MARKER_ALIASES.get(key, name.strip())


def _marker_with_name(marker: Marker, name: str) -> Marker:
    return marker.model_copy(update={"name": name})


def _normalize_marker_frame(frame: MarkerFrame) -> CanonicalObservationFrame:
    markers: dict[str, Marker] = {}
    source_names: dict[str, str] = {}
    for source_name, marker in frame.markers.items():
        opensim_name = normalize_opencap_marker_name(source_name)
        markers[opensim_name] = _marker_with_name(marker, opensim_name)
        if opensim_name != source_name:
            source_names[opensim_name] = source_name
    return CanonicalObservationFrame(
        timestamp=frame.timestamp,
        markers=markers,
        frame_index=frame.frame_index,
        metadata={"source_marker_names": source_names},
    )


def _read_json_metadata(session_dir: Path) -> dict[str, Any]:
    for filename in _SESSION_METADATA_FILENAMES:
        path = session_dir / filename
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        raise ValueError(f"OpenCap metadata {path} must contain a JSON object")
    return {}


@register_adapter
class OpenCapSessionAdapter(MocapSourceAdapter):
    """Adapter for OpenCap session directories and augmented-marker TRC files."""

    format_name = "opencap_session"
    file_extensions = (".trc",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.is_dir():
            return cls._find_marker_file(p) is not None
        return cls._is_opencap_marker_file(p)

    @classmethod
    def _is_opencap_marker_file(cls, path: Path) -> bool:
        if path.suffix.lower() != ".trc" or not TRCAdapter.supports(path):
            return False
        filename = path.stem.lower()
        return any(token in filename for token in _MARKER_FILE_TOKENS)

    @classmethod
    def _find_marker_file(cls, session_dir: Path) -> Path | None:
        candidates = sorted(session_dir.rglob("*.trc"))
        preferred = [p for p in candidates if cls._is_opencap_marker_file(p)]
        if preferred:
            return preferred[0]
        supported = [p for p in candidates if TRCAdapter.supports(p)]
        return supported[0] if supported else None

    def metadata(self, path: Path) -> SourceMetadata:
        marker_file = self._resolve_marker_file(Path(path))
        marker_metadata = TRCAdapter().metadata(marker_file)
        return SourceMetadata(
            format_name=self.format_name,
            fps=marker_metadata.fps,
            frame_count=marker_metadata.frame_count,
            unit_system=marker_metadata.unit_system,
            marker_set_name="OpenCap-OpenSim",
            notes=f"marker_file={marker_file.name}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> CanonicalObservations:
        p = Path(path)
        marker_file = self._resolve_marker_file(p)
        trajectory = TRCAdapter().load_checked(marker_file, calibration=calibration)
        if not isinstance(trajectory, MarkerTrajectory):
            raise TypeError("OpenCap marker import expected a MarkerTrajectory")
        subject = _read_json_metadata(p) if p.is_dir() else {}
        frames = [_normalize_marker_frame(frame) for frame in trajectory.frames]
        session_id = p.name if p.is_dir() else marker_file.stem
        return CanonicalObservations(
            id=f"opencap-{session_id}",
            frames=frames,
            calibration=calibration,
            marker_set_name="OpenCap-OpenSim",
            subject=subject or None,
            source_provenance={
                "format": self.format_name,
                "source_path": str(p),
                "marker_file": str(marker_file),
            },
            metadata={"source_marker_set": "OpenCap augmented markers"},
        )

    def _resolve_marker_file(self, path: Path) -> Path:
        if path.is_dir():
            marker_file = self._find_marker_file(path)
            if marker_file is not None:
                return marker_file
            raise FileNotFoundError(f"OpenCap session has no TRC marker file: {path}")
        if self._is_opencap_marker_file(path):
            return path
        raise ValueError(f"Not an OpenCap marker file or session directory: {path}")
