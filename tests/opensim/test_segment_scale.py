"""Unit tests for pure-Python segment length and scale estimation (OS-3b)."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.scale import (
    DEFAULT_NOMINAL_LENGTHS_M,
    SegmentScaleResult,
    estimate_segment_scales,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TourCapture,
    load_tour_capture,
)

pytestmark = pytest.mark.unit

C3D_PATH = Path(__file__).resolve().parents[2] / "data/C3D_TA_Driver.c3d"


def _synthetic_capture(
    scale: float = 1.0, frames: int = 25, noise_std: float = 0.0
) -> TourCapture:
    """Construct a minimal valid TourCapture with canonical marker layout."""
    labels = (
        "WaistLeft",
        "WaistRight",
        "WaistLBack",
        "WaistRBack",
        "BackTop",
        "BackLeft",
        "BackRight",
        "LShoulderTop",
        "LShoulderBack",
        "LElbowOut",
        "LWristTop",
        "RShoulderTop",
        "RShoulderBack",
        "RElbowOut",
        "RWristTop",
        "LKneeOut",
        "LAnkleOut",
        "LToeIn",
        "LToeOut",
        "RKneeOut",
        "RAnkleOut",
        "RToeIn",
        "RToeOut",
    )
    # Canonical address pose positions (approximate meters)
    base_pos = {
        "WaistLeft": np.array([-0.05, 0.95, 0.15]),
        "WaistRight": np.array([-0.05, 0.95, -0.15]),
        "WaistLBack": np.array([-0.15, 0.95, 0.12]),
        "WaistRBack": np.array([-0.15, 0.95, -0.12]),
        "BackTop": np.array([-0.10, 1.35, 0.0]),
        "BackLeft": np.array([-0.12, 1.20, 0.10]),
        "BackRight": np.array([-0.12, 1.20, -0.10]),
        "LShoulderTop": np.array([0.0, 1.40, 0.20]),
        "LShoulderBack": np.array([-0.08, 1.38, 0.20]),
        "LElbowOut": np.array([0.05, 1.10, 0.25]),
        "LWristTop": np.array([0.10, 0.85, 0.15]),
        "RShoulderTop": np.array([0.0, 1.40, -0.20]),
        "RShoulderBack": np.array([-0.08, 1.38, -0.20]),
        "RElbowOut": np.array([0.05, 1.10, -0.25]),
        "RWristTop": np.array([0.10, 0.85, -0.15]),
        "LKneeOut": np.array([0.02, 0.50, 0.18]),
        "LAnkleOut": np.array([0.0, 0.10, 0.16]),
        "LToeIn": np.array([0.15, 0.05, 0.10]),
        "LToeOut": np.array([0.15, 0.05, 0.20]),
        "RKneeOut": np.array([0.02, 0.50, -0.18]),
        "RAnkleOut": np.array([0.0, 0.10, -0.16]),
        "RToeIn": np.array([0.15, 0.05, -0.10]),
        "RToeOut": np.array([0.15, 0.05, -0.20]),
    }
    points = np.zeros((frames, len(labels), 3))
    rng = np.random.default_rng(42)
    for i, label in enumerate(labels):
        p = base_pos[label] * scale
        noise = (
            rng.normal(0.0, noise_std, size=(frames, 3))
            if noise_std > 0.0
            else np.zeros((frames, 3))
        )
        points[:, i, :] = p + noise
    valid = np.ones((frames, len(labels)), dtype=bool)
    time_s = np.arange(frames) / 360.0
    return TourCapture(time_s=time_s, labels=labels, points_m=points, valid=valid)


def test_estimate_segment_scales_synthetic_scale() -> None:
    target_scale = 1.15
    cap = _synthetic_capture(scale=target_scale, frames=25, noise_std=0.0)

    # Compute unscaled nominal lengths from scale=1.0 capture
    unscaled_cap = _synthetic_capture(scale=1.0, frames=25)
    baseline = estimate_segment_scales(unscaled_cap)

    res = estimate_segment_scales(cap, nominal_lengths_m=baseline.measured_lengths_m)
    assert isinstance(res, SegmentScaleResult)

    for seg, s in res.scale_factors.items():
        assert s == pytest.approx(target_scale, rel=1e-5), f"Segment {seg} mismatch"
        assert res.rigid_residuals_m[seg] == pytest.approx(0.0, abs=1e-12)


def test_estimate_segment_scales_rigid_residual_detects_variation() -> None:
    noise = 0.005  # 5 mm noise
    cap = _synthetic_capture(scale=1.0, frames=25, noise_std=noise)
    res = estimate_segment_scales(cap)

    for seg, residual in res.rigid_residuals_m.items():
        assert residual > 0.0
        assert residual < 0.05, f"Residual {residual} unreasonably large for {seg}"


def test_estimate_segment_scales_real_tour_capture() -> None:
    if not C3D_PATH.exists():
        pytest.skip(f"Capture file not found at {C3D_PATH}")

    cap = load_tour_capture(C3D_PATH)
    res = estimate_segment_scales(cap)

    expected_segments = {
        "femur_r",
        "femur_l",
        "tibia_r",
        "tibia_l",
        "calcn_r",
        "calcn_l",
        "humerus_r",
        "humerus_l",
        "radius_r",
        "radius_l",
        "torso",
    }
    assert set(res.scale_factors.keys()) == expected_segments
    assert set(res.measured_lengths_m.keys()) == expected_segments
    assert set(res.nominal_lengths_m.keys()) == expected_segments
    assert set(res.rigid_residuals_m.keys()) == expected_segments
    assert set(res.provenance.keys()) == expected_segments

    # Physiologic sanity checks: scale factors between 0.7 and 1.5
    for seg, s in res.scale_factors.items():
        assert 0.7 <= s <= 1.5, f"Scale factor {s} out of physiologic range for {seg}"

    # Rigid residual over first 20 frames at address should be small (< 20 mm)
    for seg, r in res.rigid_residuals_m.items():
        assert r < 0.02, f"Rigid residual {r} m too high for {seg}"


def test_estimate_segment_scales_handles_early_rshouldertop_occlusion() -> None:
    cap = _synthetic_capture(frames=25)
    idx = cap.index("RShoulderTop")
    valid = np.array(cap.valid, copy=True)
    valid[:, idx] = False
    points = np.array(cap.points_m, copy=True)
    points[:, idx] = np.nan
    occluded_cap = TourCapture(
        time_s=cap.time_s, labels=cap.labels, points_m=points, valid=valid
    )

    res = estimate_segment_scales(occluded_cap)
    assert np.isfinite(res.scale_factors["humerus_r"])
    assert np.isfinite(res.scale_factors["torso"])


def test_estimate_segment_scales_rejects_insufficient_frames() -> None:
    cap = _synthetic_capture(frames=15)
    with pytest.raises(ValueError, match="at least 20 frames"):
        estimate_segment_scales(cap)


def test_estimate_segment_scales_validates_nominal_lengths() -> None:
    cap = _synthetic_capture(frames=25)
    with pytest.raises(ValueError, match="positive finite"):
        estimate_segment_scales(cap, nominal_lengths_m={"tibia_r": -0.4})
