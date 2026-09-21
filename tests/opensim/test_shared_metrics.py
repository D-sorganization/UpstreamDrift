"""Unit tests for the five shared metrics computation (OS-3b, OS-5, FB-6)."""

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.metrics import (
    SharedMetrics,
    compute_shared_metrics,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    tracked_labels,
)

pytestmark = pytest.mark.unit


def _synthetic_capture(frames: int = 250) -> tuple[TourCapture, np.ndarray]:
    labels = tracked_labels()
    time_s = np.arange(frames) / 360.0
    rng = np.random.default_rng(42)
    # Generate smooth baseline points
    points = rng.uniform(-0.5, 0.5, size=(frames, len(labels), 3))
    # Place waist markers specifically for pelvis yaw calculation
    idx_wl = labels.index("WaistLeft")
    idx_wr = labels.index("WaistRight")
    for f in range(frames):
        angle = 0.5 * np.sin(2.0 * np.pi * time_s[f])
        points[f, idx_wl] = np.array([0.15 * np.sin(angle), 0.95, 0.15 * np.cos(angle)])
        points[f, idx_wr] = np.array(
            [-0.15 * np.sin(angle), 0.95, -0.15 * np.cos(angle)]
        )

    valid = np.ones((frames, len(labels)), dtype=bool)
    capture = TourCapture(time_s=time_s, labels=labels, points_m=points, valid=valid)
    return capture, points


def test_compute_shared_metrics_zero_error() -> None:
    capture, points = _synthetic_capture(frames=250)
    metrics = compute_shared_metrics(capture, points)
    assert isinstance(metrics, SharedMetrics)
    assert metrics.whole_marker_rmse_m == pytest.approx(0.0, abs=1e-10)
    assert metrics.early_marker_rmse_m == pytest.approx(0.0, abs=1e-10)
    assert metrics.terminal_marker_rmse_m == pytest.approx(0.0, abs=1e-10)
    assert metrics.club_marker_rmse_m == pytest.approx(0.0, abs=1e-10)
    assert metrics.pelvis_yaw_rmse_rad == pytest.approx(0.0, abs=1e-10)


def test_compute_shared_metrics_known_constant_error() -> None:
    capture, points = _synthetic_capture(frames=250)
    err = 0.02  # 20 mm offset along X
    perturbed = np.array(points, copy=True)
    perturbed[:, :, 0] += err

    metrics = compute_shared_metrics(capture, perturbed)
    assert metrics.whole_marker_rmse_m == pytest.approx(err, rel=1e-5)
    assert metrics.early_marker_rmse_m == pytest.approx(err, rel=1e-5)
    assert metrics.terminal_marker_rmse_m == pytest.approx(err, rel=1e-5)
    assert metrics.club_marker_rmse_m == pytest.approx(err, rel=1e-5)
    # Uniform translation does not change pelvis yaw orientation
    assert metrics.pelvis_yaw_rmse_rad == pytest.approx(0.0, abs=1e-7)


def test_compute_shared_metrics_club_only_perturbation() -> None:
    capture, points = _synthetic_capture(frames=250)
    perturbed = np.array(points, copy=True)
    # Only perturb club markers
    club_indices = [
        i
        for i, label in enumerate(capture.labels)
        if label.startswith(("Marker_2", "Marker_3"))
    ]

    perturbed[:, club_indices, 1] += 0.05

    metrics = compute_shared_metrics(capture, perturbed)
    assert metrics.club_marker_rmse_m == pytest.approx(0.05, rel=1e-5)
    assert metrics.whole_marker_rmse_m < 0.05
    assert metrics.whole_marker_rmse_m > 0.0


def test_compute_shared_metrics_validates_inputs() -> None:
    capture, points = _synthetic_capture(frames=50)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_shared_metrics(capture, points[:30])
