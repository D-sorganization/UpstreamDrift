"""Cleaning a synthetic view: injected outliers are flagged, nothing is imputed."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct import (
    PinholeCamera,
    RenderOptions,
    SyntheticScene,
    look_at,
    outlier_flag_scores,
)
from src.motion_capture.reconstruct.cameras import intrinsics_from_fov
from src.motion_capture.reconstruct.clean import CLEAN_SCHEMA_VERSION, clean_view
from src.motion_capture.reconstruct.temporal import SmootherOptions

pytestmark = pytest.mark.unit

# Joint accelerations in the synthetic swing reach a few thousand px/s^2 at
# 1920x1200 from 4 m; the prior is set well above that.
OPTS = SmootherOptions(acceleration_sigma=20_000.0)


def _view(options: RenderOptions):
    k = intrinsics_from_fov(1920, 1200, 70.0)
    pos = np.array([0.0, 1.2, 4.0])
    cam = PinholeCamera(
        "face_on", k, look_at(pos, np.array([0.0, 1.0, 0.0])), pos, (1920, 1200)
    )
    scene = SyntheticScene([cam], fps=60.0, n_frames=180)
    views, truth = scene.render(options)
    return views["face_on"], truth


def test_injected_outliers_are_flagged_with_high_precision_and_recall() -> None:
    payload, truth = _view(
        RenderOptions(noise_px=1.0, occlusion_rate=0.03, outlier_rate=0.03, seed=7)
    )
    cleaned, report = clean_view(payload, OPTS)
    injected = {(f, j) for f, j in truth.outliers["face_on"]}
    assert len(injected) >= 30
    scores = outlier_flag_scores(report.flagged, injected)
    # Measured 0.94 / 0.88 on this seed; misses are end-of-take points and
    # sub-10 px offsets, false flags are neighbours of rejected points.
    assert scores.recall is not None and scores.recall >= 0.9
    assert scores.precision is not None and scores.precision >= 0.85
    assert cleaned["schema_version"] == CLEAN_SCHEMA_VERSION
    # a rejected observation keeps its coordinates and loses its confidence
    frame, joint = next(iter(injected & report.flagged))
    raw_row = payload["frames"][frame]
    clean_row = cleaned["frames"][frame]
    assert clean_row["keypoints_px"][joint] == raw_row["keypoints_px"][joint]
    assert clean_row["confidence"][joint] == 0.0 and raw_row["confidence"][joint] > 0.9
    # the fit is close to the truth where the raw point was wrong
    truth_px = np.array(cleaned["frames"][frame]["fit_px"][joint])
    assert np.linalg.norm(truth_px - np.array(raw_row["keypoints_px"][joint])) > 30.0
    assert all(r.residual_px > r.threshold_px > 0 for r in report.rejected)


def test_clean_signal_is_left_alone() -> None:
    payload, _ = _view(
        RenderOptions(noise_px=1.0, occlusion_rate=0.0, outlier_rate=0.0, seed=1)
    )
    cleaned, report = clean_view(payload, OPTS)
    assert report.rejected == () and report.bound_violations == 0
    assert [r["confidence"] for r in cleaned["frames"]] == [
        r["confidence"] for r in payload["frames"]
    ]
    assert all(0.5 < s < 2.0 for s in report.measurement_sigma_px.values())


def test_contracts() -> None:
    payload, _ = _view(
        RenderOptions(noise_px=0.5, occlusion_rate=0.0, outlier_rate=0.0)
    )
    with pytest.raises(Exception, match="min_confidence"):
        clean_view(payload, OPTS, min_confidence=2.0)
    short = dict(payload)
    short["frames_total"] = 2
    short["frames"] = payload["frames"][:2]
    with pytest.raises(Exception, match="at least 3"):
        clean_view(short, OPTS)
