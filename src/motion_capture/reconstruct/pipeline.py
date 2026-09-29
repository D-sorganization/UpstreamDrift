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
from collections.abc import Sequence, Mapping

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .analytics import SwingDataUnavailable, summarize_swing
from .cameras import PinholeCamera
from .camera_source import CameraSourceEvidence
from .clean import CleanReport, clean_view
from .bundle import compact_frames, observations_from_views
from .measurements import expand_measurements, gauge
from .fit import (
    RECONSTRUCTION_FILE,
    Reconstruction,
    cameras_from_records,
    fit_bundle,
    load_views,
    load_views_from,
)
from .initialize import initialize_cameras, subject_frame
from .layouts import to_reconstruct_layout
from .lens import LensCorrection, correct_camera_views, retain_camera_lenses
from ..provenance import write_stamped
from ..variants import ensure_variant, register_variant
from .skeleton import JOINT_NAMES
from .temporal import SmootherOptions

logger = get_logger(__name__)

RECONSTRUCT_DIR = "reconstruct"
CLEAN_REPORT_FILE = "clean_report.json"
DEFAULT_ACCELERATION_SIGMA_PX = 20_000.0  # px/s^2 at 1920x1200 from ~4 m


SESSION_RECONSTRUCTION_SCHEMA = "session-reconstruction/1.0.0"
CLEAN_REPORT_SCHEMA = "clean-report/1.0.0"


class SessionReconstruction(BaseModel):
    """``reconstruct/session_reconstruction.json``: the chain's summary."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SESSION_RECONSTRUCTION_SCHEMA
    session: str
    views: tuple[str, ...]
    cleaned_rejections: dict[str, int]
    reconstruction_file: str
    rms_px: float
    unobservable_points: int
    swing_summary_file: str | None = None
    swing_summary_unavailable_reason: str | None = None
    measured_lengths_m: dict[str, float] = {}
    fps: float | None = None
    excluded_joints: tuple[str, ...] = ()
    observation_set: str = "observations"
    variant: str = ""
    camera_source: str | None = None
    camera_source_sha256: str | None = None


@dataclass(frozen=True)
class MatchSpec:
    """Which observations and cameras a match uses, and where it lives (#9793)."""

    observation_set: str = "observations"
    views: tuple[str, ...] | None = None
    variant: str = ""
    camera_source: str | None = None
    exclude_joints: tuple[str, ...] = ()
    lens_corrections: Mapping[str, LensCorrection] | None = None
    camera_evidence: CameraSourceEvidence | None = None


def start_cameras_from(
    path: Path, *, capture_root: Path | None = None
) -> list[PinholeCamera]:
    """Camera records from a JSON list, or the ``cameras`` of a reconstruction."""
    require(path.is_file(), "camera start file must exist", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("schema_version") in {
        "capture-reference-solve/1",
        "capture-reference-assignment/1",
    }:
        from src.tools.capture_rig.reference_calibration.reuse_evidence import (
            validate_reference_layout,
        )

        validate_reference_layout(payload, capture_root)
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
    scale_anchor: tuple[str, float] | None = None,
    measurements: Sequence[str] = (),
    acceleration_sigma_px: float = DEFAULT_ACCELERATION_SIGMA_PX,
    min_confidence: float = 0.05,
    match: MatchSpec | None = None,
) -> SessionReconstruction:
    """Run layout -> clean -> fit on ``<session>/<observation_set>``; write results.

    ``match`` selects views, observation set, output variant and exclusions;
    its camera source is recorded in provenance (#9793).

    Start from ``start_cameras`` (a previous take) or, for the first take of a
    new setup, from ``intrinsics`` alone: placement is then initialised from
    the golfer's joints and the anchor length (see :mod:`.initialize`).
    Preconditions: at least two views whose ids match; exactly one of
    ``start_cameras``/``intrinsics``; a positive acceleration prior.
    Postcondition: ``reconstruct/`` holds the cleaned views, their clean
    reports, ``reconstruction.json`` and the summary.
    """
    require(acceleration_sigma_px > 0, "acceleration_sigma_px must be positive")
    measured, scale_anchor = _measurements_with_gauge(measurements, scale_anchor)
    match = match or MatchSpec()
    if match.camera_evidence is not None:
        match.camera_evidence.verify()
    exclude_joints = match.exclude_joints
    unknown = [j for j in exclude_joints if j not in JOINT_NAMES]
    require(not unknown, "exclude_joints must name fit joints", unknown)
    require(
        (start_cameras is None) != (intrinsics is None),
        "give start cameras or intrinsics, not both",
    )
    obs_set_dir = session_dir / match.observation_set
    all_views = load_views_from(obs_set_dir)
    ids, start_cameras, intrinsics = _select_views(
        match.views, start_cameras, intrinsics
    )
    missing = [i for i in ids if i not in all_views]
    require(not missing, "start cameras without observations", missing)
    views_used = {v: all_views[v] for v in ids}
    views_used, corrections = correct_camera_views(
        views_used, start_cameras, match.lens_corrections
    )
    root = ensure_variant(session_dir, match.variant)
    out_dir = root / RECONSTRUCT_DIR
    obs_dir = out_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    options = SmootherOptions(acceleration_sigma=acceleration_sigma_px)
    rejections = _clean_all(
        views_used, ids, out_dir, options, min_confidence, tuple(exclude_joints)
    )
    if start_cameras is None:
        assert intrinsics is not None
        start_cameras = _initial_cameras(out_dir, ids, intrinsics, scale_anchor)
    start_cameras = retain_camera_lenses(start_cameras, corrections)
    record: Reconstruction = fit_bundle(
        out_dir,
        base=session_dir,
        scale_anchor=scale_anchor,
        start_cameras=start_cameras,
        measured_lengths_m=measured,
    )
    fps = float(views_used[ids[0]]["fps"])
    joints = np.load(out_dir / "joints_3d_m.npy")
    swing_file, swing_reason = _write_swing_summary(out_dir, joints, fps)
    summary = SessionReconstruction(
        session=str(session_dir),
        views=tuple(ids),
        cleaned_rejections=rejections,
        reconstruction_file=str(out_dir / RECONSTRUCTION_FILE),
        rms_px=record.rms_px,
        unobservable_points=record.unobservable_points,
        swing_summary_file=str(swing_file) if swing_reason is None else None,
        swing_summary_unavailable_reason=swing_reason,
        measured_lengths_m=dict(measured),
        fps=fps,
        excluded_joints=tuple(exclude_joints),
        observation_set=match.observation_set,
        variant=match.variant,
        camera_source=match.camera_source,
        camera_source_sha256=match.camera_evidence.sha256
        if match.camera_evidence is not None
        else None,
    )
    parameters = {
        "anchors": list(measurements),
        "scale_anchor": list(scale_anchor),
        "acceleration_sigma_px": acceleration_sigma_px,
        "lens_corrections": _lens_signatures(corrections, ids),
    }
    if match.camera_evidence is not None:
        match.camera_evidence.verify()
    _write_summary(session_dir, out_dir, obs_set_dir, summary, parameters)
    return summary


def _measurements_with_gauge(
    measurements: Sequence[str], scale_anchor: tuple[str, float] | None
) -> tuple[dict[str, float], tuple[str, float]]:
    """Expand measured dimensions while preserving the caller's scale anchor."""
    measured = expand_measurements(measurements)
    if scale_anchor is None:
        scale_anchor = gauge(measured)
    elif scale_anchor[0] not in measured:
        measured = {scale_anchor[0]: scale_anchor[1], **measured}
    return measured, scale_anchor


def _lens_signatures(
    corrections: Mapping[str, LensCorrection], ids: Sequence[str]
) -> dict[str, str]:
    """Record only the lens profiles contributing to this reconstruction."""
    return {
        view: correction.signature
        for view, correction in corrections.items()
        if view in ids
    }


def _write_swing_summary(
    out_dir: Path,
    joints: np.ndarray,
    fps: float,
) -> tuple[Path, str | None]:
    """Write swing_summary.json or remove stale summary if unavailable."""
    swing_file = out_dir / "swing_summary.json"
    swing_reason = None
    try:
        swing, _series = summarize_swing(joints, fps)
    except SwingDataUnavailable as exc:
        swing_reason = str(exc)
        swing_file.unlink(missing_ok=True)
        logger.warning("Swing summary unavailable: %s", swing_reason)
    else:
        swing_file.write_text(swing.model_dump_json(indent=2), encoding="utf-8")
    return swing_file, swing_reason


def _select_views(
    views: Sequence[str] | None,
    start_cameras: Sequence[PinholeCamera] | None,
    intrinsics: Sequence[tuple[str, Any, tuple[int, int]]] | None,
) -> tuple[list[str], Sequence[PinholeCamera] | None, Any]:
    """The camera ids to use (in ``views`` order when given) and the matching
    start cameras / intrinsics. Precondition: ``views`` (if given) names
    two or more cameras that have records."""
    ids = (
        [c.camera_id for c in start_cameras]
        if start_cameras
        else [i[0] for i in intrinsics or []]
    )
    if views is None:
        return ids, start_cameras, intrinsics
    unknown = [v for v in views if v not in ids]
    require(not unknown, "requested views have no camera record", unknown)
    require(len(views) >= 2, "triangulation needs at least two views", views)
    ids = list(views)
    if start_cameras:
        start_cameras = [c for v in ids for c in start_cameras if c.camera_id == v]
    elif intrinsics is not None:
        intrinsics = [i for v in ids for i in intrinsics if i[0] == v]
    return ids, start_cameras, intrinsics


def _write_summary(
    session_dir: Path,
    out_dir: Path,
    obs_set_dir: Path,
    summary: SessionReconstruction,
    parameters: Mapping[str, Any],
) -> None:
    """``session_reconstruction.json`` with provenance; register the variant."""
    ids = list(summary.views)
    source_views = [obs_set_dir / f"{v}.json" for v in ids]
    write_stamped(
        out_dir / "session_reconstruction.json",
        summary.model_dump(mode="json"),
        schema_version=SESSION_RECONSTRUCTION_SCHEMA,
        module=__name__,
        inputs=[*source_views, out_dir / RECONSTRUCTION_FILE],
        parameters={
            "views": ids,
            "observation_set": summary.observation_set,
            "variant": summary.variant,
            "excluded_joints": list(summary.excluded_joints),
            "camera_source": summary.camera_source,
            "source": {"kind": "triangulate"},
            **dict(parameters),
        },
        derived_from=[out_dir / RECONSTRUCTION_FILE, *source_views],
        base=session_dir,
    )
    register_variant(
        session_dir,
        summary.variant,
        views=ids,
        observation_set=summary.observation_set,
        source={"kind": "triangulate", "camera_source": summary.camera_source},
        module=__name__,
    )


def _clean_all(
    views: Mapping[str, dict[str, Any]],
    ids: Sequence[str],
    out_dir: Path,
    options: SmootherOptions,
    min_confidence: float,
    exclude_joints: Sequence[str],
) -> dict[str, int]:
    """Clean every view into ``out_dir/observations``; write the clean report."""
    obs_dir = out_dir / "observations"
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
    write_stamped(
        out_dir / CLEAN_REPORT_FILE,
        {"views": reports},
        schema_version=CLEAN_REPORT_SCHEMA,
        module=__name__,
        parameters={
            "acceleration_sigma": options.acceleration_sigma,
            "min_confidence": min_confidence,
            "excluded_joints": list(exclude_joints),
        },
    )
    return rejections


def _initial_cameras(
    out_dir: Path,
    ids: Sequence[str],
    intrinsics: Sequence[tuple[str, Any, tuple[int, int]]],
    scale_anchor: tuple[str, float],
) -> tuple[PinholeCamera, ...]:
    """First-take placement from the cleaned joints and the intrinsics alone."""
    obs, _ = compact_frames(observations_from_views(load_views(out_dir), ids))
    init = initialize_cameras(
        obs,
        [i[1] for i in intrinsics],
        [i[2] for i in intrinsics],
        anchor=scale_anchor,
    )
    require(
        init.ok, "camera initialisation lacks inliers", [p.inliers for p in init.pairs]
    )
    logger.info(
        "initialised placement from joints: %s",
        [(p.camera_id, p.inliers) for p in init.pairs],
    )
    return subject_frame(init.cameras, obs)


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
