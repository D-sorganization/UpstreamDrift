"""Run the joint fit on a bundle of per-view observations and write the evidence.

Input is a directory with ``observations/<view>.json`` (what ``rig ingest``
writes, or the synthetic renderer) plus the cameras to start from: either the
``cameras`` list of a ``truth.json`` (synthetic, or the previous take's
solution) or explicit :class:`CameraCalibration` records. Output is
``reconstruction.json``: refined camera records, the learned bone lengths,
per-view residual statistics, every rejected observation with its residual,
and — when ``truth.json`` is present — the metrics against it. Nothing is
reported as measured that was not; a view or point without evidence says so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.core.contracts import require

from ..provenance import write_stamped
from src.shared.python.pose_estimation.observations import CameraCalibration

from .bundle import (
    BundleOptions,
    BundleResult,
    bundle_adjust,
    compact_frames,
    expand_result,
    observations_from_views,
)
from .cameras import PinholeCamera
from .layouts import to_reconstruct_layout
from .metrics import bone_length_errors, camera_pose_error, joint_position_errors
from .skeleton import DEFAULT_LENGTHS_M, JOINT_NAMES
from .synthetic import SyntheticTruth, load_truth

RECONSTRUCTION_SCHEMA_VERSION = "reconstruction/1.0.0"
RECONSTRUCTION_FILE = "reconstruction.json"


class ViewResidualStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    view: str
    observations: int
    rejected: int
    rms_px: float | None
    p95_px: float | None


class Reconstruction(BaseModel):
    """``reconstruction.json``."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = RECONSTRUCTION_SCHEMA_VERSION
    joint_names: tuple[str, ...]
    frames: int
    cameras: list[dict[str, Any]]  # CameraCalibration.to_dict()
    bone_lengths_m: dict[str, float]
    scale_anchor: tuple[str, float]
    measured_lengths_m: dict[str, float] = {}
    rms_px: float
    initial_rms_px: float
    unobservable_points: int
    views: tuple[ViewResidualStats, ...]
    rejected: list[dict[str, Any]]
    metrics: dict[str, Any] = Field(default_factory=dict)


def load_views(bundle_dir: Path) -> dict[str, dict[str, Any]]:
    """Every ``observations/<view>.json`` keyed by its ``view``."""
    return load_views_from(bundle_dir / "observations")


def load_views_from(obs_dir: Path, minimum: int = 2) -> dict[str, dict[str, Any]]:
    """Every ``<view>.json`` of an observation-set directory keyed by ``view``.

    Precondition: the directory exists and holds at least ``minimum`` views
    (index/report files are skipped).
    """
    require(obs_dir.is_dir(), "observation set directory missing", str(obs_dir))
    views: dict[str, dict[str, Any]] = {}
    for path in sorted(obs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "view" in payload and "frames" in payload:
            views[str(payload["view"])] = payload
    require(len(views) >= minimum, "not enough views in set", sorted(views))
    return views


def cameras_from_records(records: Sequence[Mapping[str, Any]]) -> list[PinholeCamera]:
    return [
        PinholeCamera.from_calibration(CameraCalibration.from_dict(r)) for r in records
    ]


def _view_stats(
    result: BundleResult, camera_ids: Sequence[str]
) -> tuple[ViewResidualStats, ...]:
    out = []
    for c, cid in enumerate(camera_ids):
        res = result.residuals_px[c]
        finite = res[np.isfinite(res)]
        rejected = sum(1 for r in result.rejected if r.camera_id == cid)
        out.append(
            ViewResidualStats(
                view=cid,
                observations=int(finite.size),
                rejected=rejected,
                rms_px=float(np.sqrt(np.mean(finite**2))) if finite.size else None,
                p95_px=float(np.percentile(finite, 95)) if finite.size else None,
            )
        )
    return tuple(out)


def _metrics_against(truth: SyntheticTruth, result: BundleResult) -> dict[str, Any]:
    truth_cams = cameras_from_records(truth.cameras)
    by_id = {c.camera_id: c for c in truth_cams}
    cams: dict[str, Any] = {}
    for est in result.cameras:
        true = by_id.get(est.camera_id)
        if true is None:
            continue
        err = camera_pose_error(
            est.rotation_world_from_camera,
            est.position_m,
            true.rotation_world_from_camera,
            true.position_m,
        )
        cams[est.camera_id] = err.model_dump()
    joints = joint_position_errors(result.joints_3d_m, np.array(truth.joints_3d_m))
    lengths = bone_length_errors(result.bone_lengths_m, truth.bone_lengths_m)
    return {
        "cameras": cams,
        "joints": joints.model_dump(),
        "bone_length_relative_error": {
            k: (None if np.isnan(v) else v) for k, v in lengths.items()
        },
    }


def fit_bundle(
    bundle_dir: Path,
    *,
    base: Path | None = None,
    scale_anchor: tuple[str, float],
    start_cameras: Sequence[PinholeCamera] | None = None,
    length_prior_m: Mapping[str, float] = DEFAULT_LENGTHS_M,
    options: BundleOptions | None = None,
    measured_lengths_m: Mapping[str, float] | None = None,
) -> Reconstruction:
    """Fit the bundle and write ``reconstruction.json``; returns the record.

    Preconditions: at least two views whose detector layouts agree; a start
    placement from ``start_cameras`` or from ``truth.json``'s cameras.
    Postcondition: the file lists every rejected observation with its residual.
    """
    views = load_views(bundle_dir)
    truth_path = bundle_dir / "truth.json"
    truth = load_truth(truth_path) if truth_path.is_file() else None
    if start_cameras is None:
        require(truth is not None, "no start cameras and no truth.json to start from")
        assert truth is not None
        start_cameras = cameras_from_records(truth.cameras)
    ids = [c.camera_id for c in start_cameras]
    require(all(i in views for i in ids), "start cameras must match the views", ids)
    from .lens import correct_camera_views

    views, _ = correct_camera_views(views, start_cameras)
    # Detector layouts (MediaPipe 33, BODY_25) are mapped onto the 15-joint
    # reconstruct skeleton; midpoints carry the minimum parent confidence.
    views = {
        k: (
            v
            if tuple(v["detector_layout"]["keypoint_names"]) == JOINT_NAMES
            else to_reconstruct_layout(v)
        )
        for k, v in views.items()
    }
    obs = observations_from_views(views, ids)
    names = tuple(views[ids[0]]["detector_layout"]["keypoint_names"])
    require(names == JOINT_NAMES, "fit expects the 15-joint reconstruct layout", names)
    opts = options or BundleOptions()
    opts = BundleOptions(
        **{
            **vars(opts),
            "scale_anchor": scale_anchor,
            "measured_lengths_m": dict(measured_lengths_m or {}),
        }
    )
    compact, keep = compact_frames(obs)
    result = expand_result(
        bundle_adjust(
            start_cameras, compact, length_prior_m=length_prior_m, options=opts
        ),
        keep,
        obs.pixels.shape[1],
    )
    record = Reconstruction(
        joint_names=names,
        frames=int(result.joints_3d_m.shape[0]),
        cameras=[c.to_calibration().to_dict() for c in result.cameras],
        bone_lengths_m=result.bone_lengths_m,
        scale_anchor=scale_anchor,
        measured_lengths_m=dict(measured_lengths_m or {}),
        rms_px=result.rms_px,
        initial_rms_px=result.initial_rms_px,
        unobservable_points=result.unobservable_points,
        views=_view_stats(result, ids),
        rejected=[
            {
                "view": r.camera_id,
                "frame": r.frame,
                "joint": r.joint,
                "residual_px": r.residual_px,
            }
            for r in result.rejected
        ],
        metrics=_metrics_against(truth, result) if truth is not None else {},
    )
    cleaned = sorted((bundle_dir / "observations").glob("*.json"))
    write_stamped(
        bundle_dir / RECONSTRUCTION_FILE,
        record.model_dump(mode="json"),
        schema_version=record.schema_version,
        module=__name__,
        inputs=cleaned,
        parameters={
            "scale_anchor": list(scale_anchor),
            "measured_lengths_m": dict(measured_lengths_m or {}),
            "start_cameras": [c.camera_id for c in start_cameras],
        },
        derived_from=cleaned,
        base=base or bundle_dir.parent,
    )
    np.save(bundle_dir / "joints_3d_m.npy", result.joints_3d_m)
    return record
