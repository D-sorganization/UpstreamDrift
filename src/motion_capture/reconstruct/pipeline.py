"""Session bundle to 3-D: ingest output -> layout -> clean -> joint fit.

This is the UpstreamDrift orchestration seam (C7, #9627): it takes the
``observations/<view>.json`` files ``rig ingest`` wrote for a real take, maps
them onto the reconstruct skeleton, cleans each view with the dynamics prior
(rejections listed), writes the cleaned views under ``reconstruct/``, and runs
the joint fit from a starting camera placement — the previous take's
``reconstruction.json`` cameras, or explicit records. Everything the stages
produce is written beside the bundle so a take can be audited file by file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .analytics import summarize_swing
from .cameras import PinholeCamera
from .clean import CleanReport, clean_view
from .bundle import observations_from_views
from .fit import (
    RECONSTRUCTION_FILE,
    Reconstruction,
    cameras_from_records,
    fit_bundle,
    load_views,
)
from .initialize import initialize_cameras, subject_frame
from .layouts import to_reconstruct_layout
from .skeleton import JOINT_NAMES
from .temporal import SmootherOptions

logger = get_logger(__name__)

RECONSTRUCT_DIR = "reconstruct"
CLEAN_REPORT_FILE = "clean_report.json"
DEFAULT_ACCELERATION_SIGMA_PX = 20_000.0  # px/s^2 at 1920x1200 from ~4 m


class SessionReconstruction(BaseModel):
    """``reconstruct/session_reconstruction.json``: the chain's summary."""

    model_config = ConfigDict(frozen=True)

    session: str
    views: tuple[str, ...]
    cleaned_rejections: dict[str, int]
    reconstruction_file: str
    rms_px: float
    unobservable_points: int
    swing_summary_file: str | None = None
    fps: float | None = None
    excluded_joints: tuple[str, ...] = ()


def start_cameras_from(path: Path) -> list[PinholeCamera]:
    """Camera records from a JSON list, or the ``cameras`` of a reconstruction."""
    require(path.is_file(), "camera start file must exist", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["cameras"] if isinstance(payload, dict) else payload
    require(bool(isinstance(records, list) and records), "no camera records in file")
    return cameras_from_records(records)


def intrinsics_from(path: Path) -> list[tuple[str, Any, tuple[int, int]]]:
    """``(camera_id, K, image_size)`` per entry of a JSON list (extrinsics ignored)."""
    require(path.is_file(), "intrinsics file must exist", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["cameras"] if isinstance(payload, dict) else payload
    out = []
    for r in records:
        matrix = r["intrinsics"]["matrix"] if "intrinsics" in r else r["matrix"]
        size = r["image_size_px"]
        out.append(
            (
                str(r["camera_id"]),
                np.asarray(matrix, dtype=float),
                (int(size[0]), int(size[1])),
            )
        )
    require(len(out) >= 2, "need intrinsics for at least two cameras")
    return out


def reconstruct_session(
    session_dir: Path,
    *,
    start_cameras: Sequence[PinholeCamera] | None = None,
    intrinsics: Sequence[tuple[str, Any, tuple[int, int]]] | None = None,
    scale_anchor: tuple[str, float],
    acceleration_sigma_px: float = DEFAULT_ACCELERATION_SIGMA_PX,
    min_confidence: float = 0.05,
    exclude_joints: Sequence[str] = (),
) -> SessionReconstruction:
    """Run layout -> clean -> fit on ``<session>/observations`` and write results.

    Start from ``start_cameras`` (a previous take) or, for the first take of a
    new setup, from ``intrinsics`` alone: placement is then initialised from
    the golfer's joints and the anchor length (see :mod:`.initialize`).
    Preconditions: at least two views whose ids match; exactly one of
    ``start_cameras``/``intrinsics``; a positive acceleration prior.
    Postcondition: ``reconstruct/`` holds the cleaned views, their clean
    reports, ``reconstruction.json`` and the summary.
    """
    require(acceleration_sigma_px > 0, "acceleration_sigma_px must be positive")
    unknown = [j for j in exclude_joints if j not in JOINT_NAMES]
    require(not unknown, "exclude_joints must name fit joints", unknown)
    require(
        (start_cameras is None) != (intrinsics is None),
        "give start cameras or intrinsics, not both",
    )
    views = load_views(session_dir)
    ids = (
        [c.camera_id for c in start_cameras]
        if start_cameras
        else [i[0] for i in intrinsics or []]
    )
    missing = [i for i in ids if i not in views]
    require(not missing, "start cameras without observations", missing)
    out_dir = session_dir / RECONSTRUCT_DIR
    obs_dir = out_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    options = SmootherOptions(acceleration_sigma=acceleration_sigma_px)
    rejections: dict[str, int] = {}
    reports: dict[str, Any] = {}
    for view_id in ids:
        payload: dict[str, Any] = dict(views[view_id])
        if tuple(payload["detector_layout"]["keypoint_names"]) != JOINT_NAMES:
            payload = to_reconstruct_layout(payload)
        cleaned, report = clean_view(payload, options, min_confidence=min_confidence)
        if exclude_joints:
            cleaned = _exclude(cleaned, exclude_joints)
        (obs_dir / f"{view_id}.json").write_text(
            json.dumps(cleaned, indent=1), encoding="utf-8"
        )
        rejections[view_id] = len(report.rejected)
        reports[view_id] = _report_dict(report)
        logger.info("cleaned %s: %d rejections", view_id, len(report.rejected))
    (out_dir / CLEAN_REPORT_FILE).write_text(
        json.dumps(reports, indent=1), encoding="utf-8"
    )
    if start_cameras is None:
        assert intrinsics is not None
        cleaned_views = load_views(out_dir)
        obs = observations_from_views(cleaned_views, ids)
        init = initialize_cameras(
            obs,
            [i[1] for i in intrinsics],
            [i[2] for i in intrinsics],
            anchor=scale_anchor,
        )
        require(
            init.ok,
            "camera initialisation lacks inliers",
            [p.inliers for p in init.pairs],
        )
        start_cameras = subject_frame(init.cameras, obs)
        logger.info(
            "initialised placement from joints: %s",
            [(p.camera_id, p.inliers) for p in init.pairs],
        )
    record: Reconstruction = fit_bundle(
        out_dir, scale_anchor=scale_anchor, start_cameras=start_cameras
    )
    fps = float(views[ids[0]]["fps"])
    joints = np.load(out_dir / "joints_3d_m.npy")
    swing, _series = summarize_swing(joints, fps)
    swing_file = out_dir / "swing_summary.json"
    swing_file.write_text(swing.model_dump_json(indent=2), encoding="utf-8")
    summary = SessionReconstruction(
        session=str(session_dir),
        views=tuple(ids),
        cleaned_rejections=rejections,
        reconstruction_file=str(out_dir / RECONSTRUCTION_FILE),
        rms_px=record.rms_px,
        unobservable_points=record.unobservable_points,
        swing_summary_file=str(swing_file),
        fps=fps,
        excluded_joints=tuple(exclude_joints),
    )
    (out_dir / "session_reconstruction.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    return summary


def _exclude(payload: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """The cleaned view with the named joints' confidence set to 0 (#9662).

    The fit treats confidence 0 as unobserved; the rigid-segment and
    symmetry priors still place the joint, so nothing downstream changes shape.
    """
    idx = [JOINT_NAMES.index(n) for n in names]
    rows = []
    for row in payload["frames"]:
        conf = list(row["confidence"])
        for i in idx:
            conf[i] = 0.0
        rows.append({**row, "confidence": conf})
    return {**payload, "frames": rows}


def _report_dict(report: CleanReport) -> dict[str, Any]:
    return {
        "frames": report.frames,
        "rejected": [r.model_dump() for r in report.rejected],
        "bound_violations": report.bound_violations,
        "measurement_sigma_px": report.measurement_sigma_px,
    }
