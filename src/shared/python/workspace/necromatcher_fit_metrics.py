"""Canonical confidence-weighted body image metrics shared by native workers."""

from __future__ import annotations
from typing import Any
import numpy as np
from src.shared.python.motion_matching.pipeline.plant import MatchingPlant
from src.shared.python.motion_matching.historical_fit import (
    CameraProjection,
    CaptureImageEvidence,
)


def dense_reprojection_metrics(
    native: MatchingPlant,
    camera: CameraProjection,
    attachments: dict[str, Any],
    evidence: CaptureImageEvidence,
    q: np.ndarray,
    fit_indices: tuple[int, ...],
) -> dict[str, float | None]:
    residuals = np.array(
        [
            camera.residual(
                native.marker_positions(pose, attachments), observed, weights
            )
            for pose, observed, weights in zip(
                q, evidence.observed_pixels, evidence.confidence, strict=True
            )
        ]
    )
    held_out = np.array([index not in fit_indices for index in evidence.frame_indices])
    result: dict[str, float | None] = {}
    for name, selected in (
        ("dense_rms_pixels", np.ones(len(q), dtype=bool)),
        ("held_out_rms_pixels", held_out),
    ):
        weight = float(np.sum(evidence.confidence[selected]))
        result[name] = (
            float(np.sqrt(np.sum(residuals[selected] ** 2) / weight))
            if weight > 0
            else None
        )
    return result
