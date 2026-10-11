"""OpenCap session adapter.

OpenCap writes augmented markers to ``MarkerData/<trial>.trc``. The augmented
markers are named after the sites of OpenCap's LaiUhlrich2022 OpenSim model
(``r.ASIS_study``, ``RHJC_study``, ...), and the same TRC also carries the raw
2-D detector keypoints the augmenter consumed (``Neck``, ``RShoulder``, ...).

This adapter resolves the trial's TRC, reuses the canonical TRC parser, and
lifts the result into ``CanonicalObservations``. Real OpenCap names are kept
verbatim so IK against the session's scaled model needs no renaming; legacy
labels (``R_ASIS``, ``R.Acromium``) are mapped onto that vocabulary (#11402).
Detector keypoints are never aliased onto augmented markers: OpenCap stores
both ``RShoulder`` and ``r_shoulder_study`` and they are different points.

For the scaled model, IK kinematics and subject metadata of a whole session,
use :func:`opencap_session.load_opencap_session`.
"""

from __future__ import annotations

from pathlib import Path

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CanonicalObservationFrame,
    CanonicalObservations,
    MarkerFrame,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.opencap_layout import (
    OpenCapSessionLayout,
)
from src.shared.python.motion_pipeline.sources.opencap_markers import (
    OPENCAP_AUGMENTED_MARKERS,
    OPENCAP_MARKER_SET_NAME,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter
from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter

_STUDY_SUFFIX = "_study"

_MARKER_FILE_TOKENS = (
    "augmented",
    "marker_trajectories",
    "markers",
)

# Labels used by hand-assembled sessions before #11402. Keys are lower-case.
# Detector keypoint names (``rshoulder``) are deliberately absent.
_LEGACY_ALIASES = {
    "r_asis": "r.ASIS_study",
    "rasis": "r.ASIS_study",
    "l_asis": "L.ASIS_study",
    "lasis": "L.ASIS_study",
    "r_psis": "r.PSIS_study",
    "rpsis": "r.PSIS_study",
    "l_psis": "L.PSIS_study",
    "lpsis": "L.PSIS_study",
    "r.acromium": "r_shoulder_study",
    "r_acromion": "r_shoulder_study",
    "l.acromium": "L_shoulder_study",
    "l_acromion": "L_shoulder_study",
}


def _build_alias_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for name in OPENCAP_AUGMENTED_MARKERS:
        key = name.lower()
        table[key] = name
        table[key.removesuffix(_STUDY_SUFFIX)] = name
    table.update(_LEGACY_ALIASES)
    return table


_ALIASES = _build_alias_table()
_AUGMENTED_MARKERS = frozenset(OPENCAP_AUGMENTED_MARKERS)


def normalize_opencap_marker_name(name: str) -> str:
    """Return the LaiUhlrich2022 site name for an OpenCap marker label.

    Real OpenCap names and unknown labels (including detector keypoints) are
    returned stripped but otherwise unchanged.
    """
    stripped = name.strip()
    return _ALIASES.get(stripped.lower(), stripped)


def _name_mapping(source_names: list[str]) -> dict[str, str]:
    """Map each source label to its canonical name, rejecting collisions."""
    mapping: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for source in source_names:
        canonical = normalize_opencap_marker_name(source)
        if canonical in claimed:
            raise ValueError(
                f"OpenCap markers {claimed[canonical]!r} and {source!r} both "
                f"map to {canonical!r}; refusing to drop one of them"
            )
        claimed[canonical] = source
        mapping[source] = canonical
    return mapping


def _normalize_marker_frame(
    frame: MarkerFrame, mapping: dict[str, str]
) -> CanonicalObservationFrame:
    markers = {
        mapping[source]: marker.model_copy(update={"name": mapping[source]})
        for source, marker in frame.markers.items()
    }
    renamed = {
        canonical: source
        for source, canonical in mapping.items()
        if canonical != source and source in frame.markers
    }
    return CanonicalObservationFrame(
        timestamp=frame.timestamp,
        markers=markers,
        frame_index=frame.frame_index,
        metadata={"source_marker_names": renamed},
    )


@register_adapter
class OpenCapSessionAdapter(MocapSourceAdapter):
    """Adapter for OpenCap session directories and augmented-marker TRC files.

    A session directory with several motion trials is ambiguous here; load a
    specific ``MarkerData/<trial>.trc`` or use ``load_opencap_session``.
    """

    format_name = "opencap_session"
    file_extensions = (".trc",)

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        if p.is_dir():
            return bool(OpenCapSessionLayout.discover(p).marker_files)
        return cls._is_opencap_marker_file(p)

    @classmethod
    def _is_opencap_marker_file(cls, path: Path) -> bool:
        if path.suffix.lower() != ".trc" or not TRCAdapter.supports(path):
            return False
        if path.parent.name == "MarkerData":
            return True
        filename = path.stem.lower()
        return any(token in filename for token in _MARKER_FILE_TOKENS)

    def metadata(self, path: Path) -> SourceMetadata:
        marker_file = self._resolve_marker_file(Path(path))
        marker_metadata = TRCAdapter().metadata(marker_file)
        return SourceMetadata(
            format_name=self.format_name,
            fps=marker_metadata.fps,
            frame_count=marker_metadata.frame_count,
            unit_system=marker_metadata.unit_system,
            marker_set_name=OPENCAP_MARKER_SET_NAME,
            notes=f"marker_file={marker_file.name}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> CanonicalObservations:
        p = Path(path)
        marker_file = self._resolve_marker_file(p)
        session_dir = p if p.is_dir() else None
        return self.load_trial(marker_file, session_dir, calibration=calibration)

    def load_trial(
        self,
        marker_file: Path,
        session_dir: Path | None,
        calibration: Calibration | None = None,
    ) -> CanonicalObservations:
        """Load one trial's TRC; ``session_dir`` supplies subject metadata."""
        trajectory = TRCAdapter().load_checked(marker_file, calibration=calibration)
        if not isinstance(trajectory, MarkerTrajectory):
            raise TypeError("OpenCap marker import expected a MarkerTrajectory")
        mapping = _name_mapping(trajectory.frames[0].marker_names)
        frames = [_normalize_marker_frame(f, mapping) for f in trajectory.frames]
        canonical_names = list(mapping.values())
        subject = (
            OpenCapSessionLayout.discover(session_dir).metadata
            if session_dir is not None
            else {}
        )
        session_id = session_dir.name if session_dir is not None else "file"
        return CanonicalObservations(
            id=f"opencap-{session_id}-{marker_file.stem}",
            frames=frames,
            calibration=calibration,
            marker_set_name=OPENCAP_MARKER_SET_NAME,
            subject=subject or None,
            source_provenance={
                "format": self.format_name,
                "source_path": str(session_dir or marker_file),
                "marker_file": str(marker_file),
                "trial": marker_file.stem,
            },
            metadata={
                "source_marker_set": "OpenCap augmented markers",
                "augmented_markers": [
                    n for n in canonical_names if n in _AUGMENTED_MARKERS
                ],
                "detector_keypoints": [
                    n for n in canonical_names if n not in _AUGMENTED_MARKERS
                ],
            },
        )

    def _resolve_marker_file(self, path: Path) -> Path:
        if path.is_dir():
            layout = OpenCapSessionLayout.discover(path)
            return layout.marker_files[layout.resolve_trial()]
        if self._is_opencap_marker_file(path):
            return path
        raise ValueError(f"Not an OpenCap marker file or session directory: {path}")
