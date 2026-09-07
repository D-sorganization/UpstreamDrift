"""Clean one view's 2-D observations with the dynamics prior (C5 + C6).

Applies :func:`.temporal.smooth` joint by joint to a ``view-observations``
payload (the file ``rig ingest`` writes or the synthetic renderer produces):
each joint's pixel track is fitted with the acceleration prior, gross
detections are rejected and *listed*, and the cleaned payload keeps the same
schema so every later stage reads cleaned and raw files identically.

Honesty rules: a rejected observation gets confidence 0 in the output (and
its original coordinates are left in place, not replaced by the fit); the
fitted track is written to a separate ``fit_px`` field with its uncertainty;
and every rejection carries the residual that condemned it. Nothing is
interpolated into ``keypoints_px``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require

from .temporal import SmootherOptions, smooth

CLEAN_SCHEMA_VERSION = "view-observations-clean/1.0.0"


class JointRejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame: int
    joint: str
    residual_px: float
    threshold_px: float


class CleanReport(BaseModel):
    """What cleaning did to one view."""

    model_config = ConfigDict(frozen=True)

    view: str
    frames: int
    joint_names: tuple[str, ...]
    rejected: tuple[JointRejection, ...]
    bound_violations: int
    measurement_sigma_px: dict[str, float]

    @property
    def flagged(self) -> set[tuple[int, int]]:
        """``(frame, joint index)`` pairs for :func:`.metrics.outlier_flag_scores`."""
        index = {n: i for i, n in enumerate(self.joint_names)}
        return {(r.frame, index[r.joint]) for r in self.rejected}


def _tracks(
    payload: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    names = list(payload["detector_layout"]["keypoint_names"])
    fps = float(payload["fps"])
    total = int(payload["frames_total"])
    k = len(names)
    px = np.full((total, k, 2), np.nan)
    conf = np.zeros((total, k))
    for row in payload["frames"]:
        t = int(round(float(row["time_s"]) * fps))
        if 0 <= t < total:
            px[t] = np.asarray(row["keypoints_px"], dtype=float)
            conf[t] = np.asarray(row["confidence"], dtype=float)
    return names, px, conf, fps


def clean_view(
    payload: Mapping[str, Any],
    options: SmootherOptions,
    *,
    min_confidence: float = 0.05,
) -> tuple[dict[str, Any], CleanReport]:
    """Return a cleaned copy of ``payload`` and the report of what was rejected.

    ``options.acceleration_sigma`` is in pixels per second squared. Detections
    with confidence below ``min_confidence`` are treated as unobserved.
    Postcondition: the cleaned payload has the same frames and keypoint order;
    rejected observations have confidence 0 and untouched coordinates.
    """
    require(0.0 <= min_confidence <= 1.0, "min_confidence in [0, 1]", min_confidence)
    names, px, conf, fps = _tracks(payload)
    total, k, _ = px.shape
    require(total >= 3, "need at least 3 frames to clean", total)
    fit = np.full_like(px, np.nan)
    unc = np.full_like(px, np.nan)
    clean_conf = conf.copy()
    rejected: list[JointRejection] = []
    sigmas: dict[str, float] = {}
    violations = 0
    for j, name in enumerate(names):
        observed = conf[:, j] >= min_confidence
        z = np.where(observed[:, None], px[:, j, :], np.nan)
        if observed.sum() < 3:
            continue
        result = smooth(z, np.clip(conf[:, j], 0.0, 1.0), fps, options)
        fit[:, j, :] = result.values
        unc[:, j, :] = result.uncertainty
        sigmas[name] = float(np.mean(result.measurement_sigma))
        violations += len(result.violations)
        bad = sorted({r.frame for r in result.rejected})
        for t in bad:
            residual = float(np.linalg.norm(px[t, j] - result.values[t]))
            gate = float(max(r.threshold for r in result.rejected if r.frame == t))
            rejected.append(
                JointRejection(
                    frame=t, joint=name, residual_px=residual, threshold_px=gate
                )
            )
            clean_conf[t, j] = 0.0
    out = dict(payload)
    out["schema_version"] = CLEAN_SCHEMA_VERSION
    frames: list[dict[str, Any]] = []
    for row in payload["frames"]:
        t = int(round(float(row["time_s"]) * fps))
        new = dict(row)
        new["confidence"] = clean_conf[t].tolist()
        new["fit_px"] = np.nan_to_num(fit[t], nan=0.0).tolist()
        new["fit_uncertainty_px"] = np.nan_to_num(unc[t], nan=0.0).tolist()
        frames.append(new)
    out["frames"] = frames
    out["provenance"] = {
        **dict(payload.get("provenance", {})),
        "cleaned_with": "reconstruct.clean",
        "acceleration_sigma_px_s2": options.acceleration_sigma,
        "rejections": len(rejected),
    }
    report = CleanReport(
        view=str(payload["view"]),
        frames=total,
        joint_names=tuple(names),
        rejected=tuple(rejected),
        bound_violations=violations,
        measurement_sigma_px=sigmas,
    )
    return out, report
