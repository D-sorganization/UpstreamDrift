"""C3D loader for cluster-marker mocap files.

Reuses the canonical ``C3DDataReader`` from
``src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py``.
Marker-name discovery is heuristic because the cluster-marker set is not
documented in this repo (issue #013 is the verification pass).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader
from src.shared.python.upstream_drift_tools.lab.bio._c3d_marker_set import (
    MarkerSet,
    MarkerSetMismatchError,
)
from src.shared.python.upstream_drift_tools.lab.bio._c3d_models import C3DEvent

from ..club_target import AlignOptions, ClubTarget, SourceProvenance
from ._align import detect_impact_index, resample_target
from ._marker_clusters import extract_cluster_club_pose, has_marker_clusters
from ._quaternion import rotmat_to_quat

logger = logging.getLogger(__name__)

BUTT_CANDIDATES: tuple[str, ...] = (
    "BUTT",
    "GRIP",
    "Grip",
    "GripButt",
    "ClubButt",
    "BUTT_END",
    "CLUB_BUTT",
)
HEAD_CANDIDATES: tuple[str, ...] = (
    "CH",
    "ClubHead",
    "CLUBHEAD",
    "Clubhead",
    "HEAD",
    "ClubFace",
    "CLUB_HEAD",
)


def _sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_marker(labels: list[str], candidates: tuple[str, ...]) -> str | None:
    """First label whose upper-cased form contains any candidate substring."""
    upper = [lbl.upper() for lbl in labels]
    for cand in candidates:
        u = cand.upper()
        for i, lbl in enumerate(upper):
            if u in lbl:
                return labels[i]
    return None


def _marker_xyz(df: pd.DataFrame, marker: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time, xyz)`` for a single marker from a tidy points dataframe."""
    sub = df[df["marker"] == marker].sort_values("frame")
    time = sub["time"].to_numpy(dtype=np.float64)
    xyz = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return time, xyz


@precondition(
    lambda path, opts, **_: Path(path).exists(),
    "C3D file must exist",
)
@precondition(
    lambda path, opts, **_: opts.sample_rate_hz > 0,
    "sample_rate_hz must be > 0",
)
@postcondition(
    lambda result: isinstance(result, ClubTarget),
    "load_club_target_c3d must return a ClubTarget",
)
def load_club_target_c3d(
    path: Path | str,
    opts: AlignOptions,
    *,
    event_label_for_alignment: str | None = None,
    marker_set_override: MarkerSet | None = None,
) -> ClubTarget:
    """Load a cluster-marker C3D file into a canonical ``ClubTarget``.

    Orientation is reconstructed from the (butt, clubhead) shaft direction
    alone — the C3D file does not carry the 3x3 rotation matrices that the
    Excel sheets do, so the quaternion encodes the swing of an axis-aligned
    shaft (no roll information). Issue #013 will refine this once the
    cluster-marker convention is fully documented.

    Args:
        path: Path to a ``.c3d`` file.
        opts: Resampling and impact-alignment options.
        event_label_for_alignment: Optional label of a C3D ``EVENT`` group
            entry (e.g. ``"Impact"``) to use as the alignment frame. When
            provided, the matching event's time replaces the kinematic-peak
            heuristic; when ``None``, the loader falls back to
            :func:`detect_impact_index` (logged at INFO when the file has no
            events). When the label is provided but absent from the file's
            events, ``ValueError`` is raised listing the available labels.
    """
    path = Path(path)
    reader = C3DDataReader(path)
    metadata = reader.get_metadata()
    df = reader.points_dataframe(include_time=True, target_units="m")
    labels = list(metadata.marker_labels)

    detected = getattr(metadata, "marker_set", MarkerSet.UNKNOWN)
    # Per issue #4710: don't return a target with NaN club poses for files
    # whose marker set we can't classify. Only raise when no fallback
    # cluster/butt/head label is available either, to preserve behaviour
    # for legacy files predating the marker-set registry.
    if (
        detected is MarkerSet.UNKNOWN
        and marker_set_override is None
        and not has_marker_clusters(path.name, labels)
        and (
            _pick_marker(labels, BUTT_CANDIDATES) is None
            or _pick_marker(labels, HEAD_CANDIDATES) is None
        )
    ):
        raise MarkerSetMismatchError(
            (
                f"C3D file {path.name} has an unrecognised marker set "
                "and no club butt/head markers; pass "
                "marker_set_override=MarkerSet.GOLF_CLUSTER to force "
                "cluster-marker handling, or supply a file with a "
                "registered marker set."
            ),
            detected=detected,
            labels=labels,
        )

    if has_marker_clusters(path.name, labels):
        logger.info(
            "Detected cluster-marker C3D schema in %s; using cluster pose",
            path.name,
        )
        raw_time, butt_raw, head_raw, raw_quat = _cluster_pose_from_dataframe(
            df, labels
        )
    else:
        butt_label = _pick_marker(labels, BUTT_CANDIDATES)
        head_label = _pick_marker(labels, HEAD_CANDIDATES)
        if butt_label is None or head_label is None:
            raise ValueError(
                "Could not identify club butt or clubhead markers in C3D file. "
                f"Available labels: {labels}"
            )
        t_butt, butt_raw = _marker_xyz(df, butt_label)
        t_head, head_raw = _marker_xyz(df, head_label)
        if t_butt.shape != t_head.shape:
            raise ValueError("Butt and clubhead marker time vectors disagree on length")
        raw_time = t_butt - float(t_butt[0])
        raw_quat = _shaft_quaternions(butt_raw, head_raw)

    impact_raw = _resolve_alignment_index(
        metadata_events=list(getattr(metadata, "events", []) or []),
        raw_time=raw_time,
        head_xyz=head_raw,
        event_label=event_label_for_alignment,
        file_label=path.name,
    )
    sim_time, butt, clubhead, quat, impact_idx = resample_target(
        raw_time, butt_raw, head_raw, raw_quat, impact_raw, opts
    )

    source = SourceProvenance(
        filename=path.name,
        format="c3d",
        subject_id=path.stem,
        trial_id=path.stem,
        sha256=_sha256_of(path),
    )
    logger.info(
        "Loaded ClubTarget from %s: %d output samples (impact=%d)",
        path.name,
        sim_time.shape[0],
        impact_idx,
    )
    return ClubTarget(
        time=sim_time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=int(impact_idx),
        source=source,
    )


def _cluster_pose_from_dataframe(
    df: pd.DataFrame, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a tidy points DataFrame into per-marker arrays then run cluster pose.

    Returns ``(time, butt_xyz, clubhead_xyz, club_quat)`` with the
    Y-up -> Z-up swap applied and rigid-body rotation derived from the
    clubhead cluster.
    """
    by_marker = {m: g.sort_values("frame") for m, g in df.groupby("marker")}
    if "time" in df.columns:
        any_marker = next(iter(by_marker.values()))
        time = any_marker["time"].to_numpy(dtype=np.float64)
    else:
        n = len(next(iter(by_marker.values())))
        time = np.arange(n, dtype=np.float64)
    time = time - float(time[0])

    points: dict[str, np.ndarray] = {}
    n = time.shape[0]
    for label in labels:
        sub = by_marker.get(label)
        if sub is None:
            continue
        xyz = np.full((n, 3), np.nan, dtype=np.float64)
        frames = sub["frame"].to_numpy(dtype=np.int64)
        # Frames may be 0- or 1-indexed depending on the source.
        if frames.min() == 1:
            frames = frames - 1
        valid = (frames >= 0) & (frames < n)
        xyz[frames[valid], 0] = sub["x"].to_numpy(dtype=np.float64)[valid]
        xyz[frames[valid], 1] = sub["y"].to_numpy(dtype=np.float64)[valid]
        xyz[frames[valid], 2] = sub["z"].to_numpy(dtype=np.float64)[valid]
        points[label] = xyz

    # Pick the first all-finite frame across both clusters as the address ref.
    address = _first_clean_frame(points)
    pose = extract_cluster_club_pose(
        points, address_frame=address, convert_to_z_up=True
    )
    keep = (
        np.all(np.isfinite(pose.clubhead), axis=1)
        & np.all(np.isfinite(pose.butt), axis=1)
        & np.all(np.isfinite(pose.rotation.reshape(pose.rotation.shape[0], -1)), axis=1)
    )
    if keep.sum() < 5:
        raise ValueError(
            f"Only {int(keep.sum())} valid frames after cluster pose extraction"
        )
    time = time[keep]
    time = time - float(time[0])
    quat = rotmat_to_quat(pose.rotation[keep])
    return time, pose.butt[keep], pose.clubhead[keep], quat


def _first_clean_frame(points: dict[str, np.ndarray]) -> int:
    """Return the lowest frame index where every required cluster marker is finite."""
    from ._marker_clusters import CLUBHEAD_CLUSTER, GRIP_CLUSTER

    required = [points[m] for m in (*CLUBHEAD_CLUSTER, *GRIP_CLUSTER)]
    n = required[0].shape[0]
    for i in range(n):
        if all(np.all(np.isfinite(arr[i])) for arr in required):
            return i
    raise ValueError(
        "No frame where all required cluster markers are simultaneously finite"
    )


def _fill_rotation_nans(rot: np.ndarray) -> np.ndarray:
    """Replace NaN-rows in a (N,3,3) rotation stack with the previous valid one."""
    out = rot.copy()
    last_good = np.eye(3, dtype=np.float64)
    seen = False
    for i in range(out.shape[0]):
        if np.all(np.isfinite(out[i])):
            last_good = out[i]
            seen = True
        elif seen:
            out[i] = last_good
        else:
            out[i] = np.eye(3, dtype=np.float64)
    return out


def _resolve_alignment_index(
    *,
    metadata_events: list[C3DEvent],
    raw_time: np.ndarray,
    head_xyz: np.ndarray,
    event_label: str | None,
    file_label: str,
) -> int:
    """Return the raw-frame index used for impact alignment.

    When ``event_label`` is supplied, the matching ``C3DEvent`` time is mapped
    to the nearest raw frame. When the label is supplied but not present in
    ``metadata_events``, a :class:`ValueError` enumerates the available
    labels. When ``event_label`` is ``None`` the kinematic-peak heuristic is
    used; this is logged at INFO level so callers can see when manual event
    annotations would have been preferred.
    """
    if event_label is not None:
        if not metadata_events:
            raise ValueError(
                f"event_label_for_alignment={event_label!r} requested but "
                f"{file_label} has no EVENT annotations"
            )
        for ev in metadata_events:
            if ev.label == event_label:
                # ``raw_time`` is referenced to its own zero (the first frame
                # in the trace). C3D ``EVENT.TIMES`` are referenced to the
                # capture's time origin, so subtract the same origin we
                # stripped from ``raw_time``.
                target_t = float(ev.time)
                idx = int(np.argmin(np.abs(raw_time - target_t)))
                logger.info(
                    "Using EVENT %r at t=%.4fs (frame %d) for impact alignment in %s",
                    event_label,
                    target_t,
                    idx,
                    file_label,
                )
                return idx
        available = [ev.label for ev in metadata_events]
        raise ValueError(
            f"event_label_for_alignment={event_label!r} not found in "
            f"{file_label}; available event labels: {available}"
        )
    if not metadata_events:
        logger.info(
            "No EVENT annotations in %s; falling back to kinematic-peak "
            "impact heuristic",
            file_label,
        )
    return int(detect_impact_index(raw_time, head_xyz))


def _shaft_quaternions(butt: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Quaternion that rotates ``+z`` onto each shaft direction.

    This is a stand-in until issue #013 documents the cluster-marker
    convention for full 3-DOF club orientation.
    """
    n = butt.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        v = head[i] - butt[i]
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            out[i] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        v = v / norm
        dot = float(np.dot(z_axis, v))
        if dot >= 1.0 - 1.0e-12:
            out[i] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        if dot <= -1.0 + 1.0e-12:
            out[i] = np.array([0.0, 1.0, 0.0, 0.0])
            continue
        axis = np.cross(z_axis, v)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(dot)
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        q = np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])
        if q[0] < 0:
            q = -q
        out[i] = q / np.linalg.norm(q)
    return out
