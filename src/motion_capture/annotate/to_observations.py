"""Manual annotations as an observation set the pipeline already reads (#9801, #9803).

``to_view_observations`` turns one view's clicks into a
``view-observations/1.0.0`` payload in the reconstruct layout (estimator
``manual``): confidence 1.0 where the user clicked, 0.0 elsewhere.
``merge`` lays a corrections layer over a detector set: a click replaces
the detector's point, a skip rejects it (confidence 0), every other
detector point is kept; the provenance counts what changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import require

from ..provenance import write_stamped
from ..reconstruct.skeleton import JOINT_NAMES
from ..rig.ingest import (
    INGEST_INDEX_FILE,
    VIEW_OBSERVATIONS_SCHEMA_VERSION,
    IngestIndex,
    ViewIngestStatus,
    ViewObservations,
)
from .store import AnnotationSet

MANUAL_LAYOUT = {"name": "manual_15", "keypoint_names": list(JOINT_NAMES)}
DEFAULT_OUT = "observations_manual"


def _row(store: AnnotationSet, frame: int, camera_id: str) -> dict[str, Any] | None:
    points = store.points_at(frame)
    if not points:
        return None
    keypoints = [[0.0, 0.0] for _ in JOINT_NAMES]
    confidence = [0.0 for _ in JOINT_NAMES]
    for joint, point in points.items():
        k = JOINT_NAMES.index(joint)
        keypoints[k] = [point.x_px, point.y_px]
        confidence[k] = 1.0
    return {
        "camera_id": camera_id,
        "time_s": frame / store.fps,
        "keypoints_px": keypoints,
        "confidence": confidence,
    }


def to_view_observations(
    store: AnnotationSet,
    *,
    identity: str = "manual",
    camera_id: str | None = None,
    frames_total: int | None = None,
    annotation_file: Path | None = None,
) -> ViewObservations:
    """One view's clicks as a ``view-observations`` payload.

    Preconditions: the store's joints are the reconstruct joints. Frames with
    only skips produce no row. Postcondition: every row has one confidence
    per joint and confidence 1.0 exactly where a point was clicked.
    """
    require(store.joints == JOINT_NAMES, "store must use the reconstruct joints")
    cid = camera_id or store.view
    rows = [
        row
        for frame in store.annotated_frames()
        if (row := _row(store, frame, cid)) is not None
    ]
    last = max(store.annotated_frames(), default=-1)
    return ViewObservations(
        view=store.view,
        identity=identity,
        camera_id=cid,
        fps=store.fps,
        width=store.width,
        height=store.height,
        frames_total=frames_total if frames_total is not None else last + 1,
        frames_with_pose=len(rows),
        detector_layout=dict(MANUAL_LAYOUT),
        frames=tuple(rows),
        provenance={
            "estimator": "manual",
            "annotator": store.annotator,
            "annotation_file": str(annotation_file) if annotation_file else None,
            "points": store.count(),
        },
    )


def merge(
    store: AnnotationSet, detector: Mapping[str, Any], *, camera_id: str | None = None
) -> tuple[ViewObservations, dict[str, int]]:
    """The detector view with the corrections layer applied.

    Returns the merged payload and ``{"replaced", "rejected", "added"}``:
    clicks on frames the detector detected replace the joint; skips zero its
    confidence; clicks on frames without a detection add a row.
    Precondition: the detector payload is in the reconstruct layout.
    """
    names = tuple(detector["detector_layout"]["keypoint_names"])
    require(names == JOINT_NAMES, "detector view must be in the reconstruct layout")
    fps = float(detector["fps"])
    by_frame: dict[int, dict[str, Any]] = {}
    for row in detector["frames"]:
        by_frame[int(round(float(row["time_s"]) * fps))] = {
            **row,
            "keypoints_px": [list(map(float, p)) for p in row["keypoints_px"]],
            "confidence": [float(c) for c in row["confidence"]],
        }
    counts = {"replaced": 0, "rejected": 0, "added": 0}
    cid = camera_id or str(detector.get("camera_id", store.view))
    frames = sorted(set(by_frame) | set(store.annotated_frames()))
    for frame in frames:
        row = by_frame.get(frame)
        if row is None:
            row = _row(store, frame, cid)
            if row is None:
                continue
            by_frame[frame] = row
            counts["added"] += 1
            continue
        for joint, point in store.points_at(frame).items():
            k = JOINT_NAMES.index(joint)
            row["keypoints_px"][k] = [point.x_px, point.y_px]
            row["confidence"][k] = 1.0
            counts["replaced"] += 1
        for joint in store.skipped_at(frame):
            k = JOINT_NAMES.index(joint)
            if row["confidence"][k] > 0:
                counts["rejected"] += 1
            row["confidence"][k] = 0.0
    rows = tuple(by_frame[f] for f in sorted(by_frame))
    merged = ViewObservations(
        view=str(detector["view"]),
        identity=str(detector.get("identity", "edited")),
        camera_id=cid,
        fps=fps,
        width=detector.get("width"),
        height=detector.get("height"),
        frames_total=int(detector.get("frames_total", len(rows))),
        frames_with_pose=len(rows),
        detector_layout=dict(detector["detector_layout"]),
        frames=rows,
        provenance={
            **dict(detector.get("provenance") or {}),
            "estimator": f"{(detector.get('provenance') or {}).get('estimator', 'unknown')}+manual",
            "annotator": store.annotator,
            "corrections": dict(counts),
        },
    )
    return merged, counts


def write_observation_set(
    session: Path,
    out_set: str,
    views: Mapping[str, ViewObservations],
    *,
    plan_name: str,
    inputs: list[Path],
    parameters: Mapping[str, Any],
) -> Path:
    """``<session>/<out_set>/<view>.json`` + index, both stamped; returns the dir."""
    require(bool(views), "at least one view to write")
    out_dir = session / out_set
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    written: list[Path] = []
    for view, payload in views.items():
        path = out_dir / f"{view}.json"
        write_stamped(
            path,
            payload.model_dump(mode="json"),
            schema_version=VIEW_OBSERVATIONS_SCHEMA_VERSION,
            module=__name__,
            inputs=inputs,
            parameters=dict(parameters),
            derived_from=inputs,
            base=session,
        )
        written.append(path)
        statuses.append(
            ViewIngestStatus(
                view=view,
                identity=payload.identity,
                status="available",
                file=path.name,
                frames_total=payload.frames_total,
                frames_with_pose=payload.frames_with_pose,
            )
        )
    index = IngestIndex(
        plan_name=plan_name, views=tuple(statuses), provenance=dict(parameters)
    )
    write_stamped(
        out_dir / INGEST_INDEX_FILE,
        index.model_dump(mode="json"),
        schema_version=VIEW_OBSERVATIONS_SCHEMA_VERSION,
        module=__name__,
        inputs=written,
        derived_from=written,
        base=session,
    )
    return out_dir
