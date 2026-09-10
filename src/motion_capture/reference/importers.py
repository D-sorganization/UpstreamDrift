"""Reference imports through the existing C3D, CIR and body-target contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from src.motion_capture.provenance import sha256_of
from src.shared.python.motion_pipeline.contracts import MarkerTrajectory
from src.shared.python.motion_pipeline.sources import C3DAdapter
from src.shared.python.motion_pipeline.sources._unit_contracts import (
    normalize_spatial_units,
)
from src.shared.python.motion_matching.loaders import load_body_target_json

from .model import (
    MAX_REFERENCE_BYTES,
    MAX_SAMPLES,
    Axis,
    ReferenceMotion,
    ReferenceSource,
)


class MotionImportOptions(TypedDict):
    title: str
    units: Literal["m", "cm", "mm"]
    axes: tuple[Axis, Axis, Axis]
    joint_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    model_identity: str | None


@dataclass(frozen=True)
class MotionDraft:
    """Decoded metres in source axes, awaiting explicit mapping confirmation."""

    source: ReferenceSource
    names: tuple[str, ...]
    time_s: tuple[float, ...]
    points: np.ndarray
    source_units: str
    units_declared: bool
    canonical: bool = False
    model_identity: str | None = None


def _from_markers(source: ReferenceSource, trajectory: MarkerTrajectory) -> MotionDraft:
    names = tuple(trajectory.metadata.get("source_labels", ())) or tuple(
        dict.fromkeys(name for frame in trajectory.frames for name in frame.markers)
    )
    count = len(trajectory.frames)
    if not names or len(names) > 256 or count * len(names) > MAX_SAMPLES:
        raise ValueError(
            "Reference marker/frame count exceeds supported limits; trim the source"
        )
    points = np.full((count, len(names), 3), np.nan)
    for fi, frame in enumerate(trajectory.frames):
        for mi, name in enumerate(names):
            marker = frame.markers.get(name)
            if marker is not None and not marker.occluded:
                points[fi, mi] = (marker.x, marker.y, marker.z)
    return MotionDraft(
        source,
        names,
        tuple(f.timestamp for f in trajectory.frames),
        points,
        str(trajectory.metadata.get("units", "m")),
        bool(trajectory.metadata.get("units_declared", False)),
    )


def load_motion_draft(path: Path) -> MotionDraft:
    """Load without resampling, filling gaps or guessing a camera registration."""
    path = path.expanduser().resolve()
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError(
            "Reference source exceeds 64 MB; trim or export a shorter motion"
        )
    digest = sha256_of(path)
    if path.suffix.lower() == ".c3d":
        source = ReferenceSource(path=str(path), sha256=digest, format="c3d")
        draft = _from_markers(source, C3DAdapter().load(path))
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        from .trace_import import load_trace_draft

        draft = load_trace_draft(path, digest)
    else:
        draft = _load_json(path, digest)
    if sha256_of(path) != digest:
        raise ValueError(
            "Reference source changed during import; retry after saving it"
        )
    return draft


def _load_json(path: Path, digest: str) -> MotionDraft:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reference JSON must contain a versioned object")
    schema = payload.get("schema")
    if schema == "body_target_json_v1":
        body = load_body_target_json(path)
        if payload.get("coordinate_frame") != "z_up_right_handed":
            raise ValueError(
                "Body-target reference must explicitly declare z_up_right_handed"
            )
        source = ReferenceSource(path=str(path), sha256=digest, format=schema)
        return MotionDraft(
            source,
            body.marker_names,
            tuple(body.time),
            body.marker_xyz,
            "m",
            True,
            True,
        )
    if schema == "marker-trajectory/1.0.0":
        if set(payload) != {"schema", "trajectory"}:
            raise ValueError(
                "Marker trajectory JSON requires only schema and trajectory"
            )
        trajectory = MarkerTrajectory.model_validate(payload["trajectory"])
        source = ReferenceSource(path=str(path), sha256=digest, format=schema)
        # CIR marker coordinates are always metres, regardless of original source metadata.
        draft = _from_markers(source, trajectory)
        return MotionDraft(source, draft.names, draft.time_s, draft.points, "m", True)
    raise ValueError(
        "Unsupported reference schema; use body_target_json_v1 or marker-trajectory/1.0.0"
    )


def finish_motion_import(
    draft: MotionDraft,
    *,
    title: str,
    units: Literal["m", "cm", "mm"],
    axes: tuple[Axis, Axis, Axis],
    joint_names: tuple[str, ...],
    edges: tuple[tuple[int, int], ...] = (),
    model_identity: str | None = None,
) -> ReferenceMotion:
    """Apply the user-confirmed units, signed axis permutation and joint mapping."""
    if len({axis[-1] for axis in axes}) != 3:
        raise ValueError("Assign each source axis exactly once")
    if draft.source.format == "simulation-trace/2" and units != "m":
        raise ValueError("Simulation trace marker coordinates already use metres")
    if draft.canonical and (units != "m" or axes != ("+X", "+Y", "+Z")):
        raise ValueError(
            "Canonical body targets already use metres and right-handed Z-up axes"
        )
    source_path = Path(draft.source.path)
    actual = normalize_spatial_units(
        units, format_name="Reference", path=source_path
    ).scale_to_meters
    decoded = normalize_spatial_units(
        draft.source_units, format_name="Reference", path=source_path
    ).scale_to_meters
    indices = ["XYZ".index(axis[-1]) for axis in axes]
    signs = np.array([1 if axis[0] == "+" else -1 for axis in axes])
    points = draft.points[:, :, indices] * signs * (actual / decoded)
    if np.isinf(points).any():
        raise ValueError("Reference contains infinite coordinates")
    return ReferenceMotion(
        title=title,
        source=draft.source,
        source_units=units,
        source_axes=axes,
        source_names=draft.names,
        joint_names=joint_names,
        edges=edges,
        model_identity=model_identity,
        time_s=draft.time_s,
        points_m=tuple(
            tuple(None if np.isnan(p).any() else tuple(p) for p in row)
            for row in points
        ),
    )
