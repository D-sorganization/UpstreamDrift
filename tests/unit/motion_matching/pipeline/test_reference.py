from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.constants import (
    CONSISTENCY_PRIOR,
    RATE_HZ,
    REFERENCE_CUTOFF_HZ,
)
from src.shared.python.motion_matching.pipeline.reference import (
    consistency_resolve,
    full_capture_ik,
    marker_errors,
    smooth_reference,
)


@pytest.mark.unit
def test_smooth_reference_attenuates_high_frequencies() -> None:
    # 360 Hz sampling, 1 second = 360 points
    t = np.linspace(0.0, 1.0, int(RATE_HZ), endpoint=False)
    # Low frequency signal (1 Hz) + high frequency noise (50 Hz)
    low_freq = np.sin(2 * np.pi * 1.0 * t)[:, None]
    high_freq = 0.5 * np.sin(2 * np.pi * 50.0 * t)[:, None]
    q = low_freq + high_freq

    smoothed = smooth_reference(q, rate_hz=RATE_HZ, cutoff_hz=REFERENCE_CUTOFF_HZ)
    assert smoothed.shape == q.shape
    # Noise should be significantly attenuated
    noise_before = np.std(q - low_freq)
    noise_after = np.std(smoothed - low_freq)
    assert noise_after < 0.2 * noise_before


@pytest.mark.unit
def test_smooth_reference_dbc_validation() -> None:
    q = np.zeros((100, 5))
    with pytest.raises(ValueError, match="rate_hz must be positive"):
        smooth_reference(q, rate_hz=0.0, cutoff_hz=12.0)

    with pytest.raises(ValueError, match="cutoff_hz must be positive"):
        smooth_reference(q, rate_hz=100.0, cutoff_hz=0.0)

    with pytest.raises(ValueError, match="Nyquist"):
        smooth_reference(q, rate_hz=100.0, cutoff_hz=60.0)

    with pytest.raises(ValueError, match="2D array"):
        smooth_reference(np.zeros(10), rate_hz=100.0, cutoff_hz=10.0)


@pytest.mark.unit
def test_marker_errors_computes_l2_norm() -> None:
    kin = MagicMock()
    # 2 frames, 3 markers, 3D coordinates
    kin.marker_positions.side_effect = [
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    q = np.zeros((2, 5))
    points = np.zeros((2, 3, 3))
    errors = marker_errors(kin, q, points)

    assert errors.shape == (2, 3)
    np.testing.assert_allclose(errors[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(errors[1], [0.0, 0.0, 0.0])


@pytest.mark.unit
def test_marker_errors_dbc_validation() -> None:
    kin = MagicMock()
    with pytest.raises(ValueError, match="q must be a 2D array"):
        marker_errors(kin, np.zeros(5), np.zeros((2, 3, 3)))

    with pytest.raises(ValueError, match="points must be a 3D array"):
        marker_errors(kin, np.zeros((2, 5)), np.zeros((2, 3)))

    with pytest.raises(ValueError, match="Frame count mismatch"):
        marker_errors(kin, np.zeros((3, 5)), np.zeros((2, 3, 3)))


@pytest.mark.unit
def test_full_capture_ik_delegates_and_unwraps() -> None:
    lane = MagicMock()
    kin = MagicMock()
    kin.nq = 10
    kin.coordinate_order = [f"coord_{i}" for i in range(10)]
    q_start = np.zeros(10)

    raw_trajectory = np.zeros((5, 10))
    raw_fits = [MagicMock() for _ in range(5)]
    lane.trajectory.return_value = (raw_trajectory, raw_fits)

    q_ik, fits = full_capture_ik(lane, kin, q_start)
    assert q_ik.shape == (5, 10)
    assert fits == raw_fits
    assert lane.trajectory.call_count == 1
    args, _ = lane.trajectory.call_args
    assert args[0] is kin
    np.testing.assert_array_equal(args[1], q_start)


@pytest.mark.unit
def test_consistency_resolve_calls_kin_solve_trajectory() -> None:
    lane = MagicMock()
    lane.frames = 5
    lane.points = np.zeros((5, 3, 3))
    lane.valid = np.ones((5, 3), dtype=bool)
    lane.ground = MagicMock()
    lane.stance = [("heel_r",)] * 5
    lane.bounds = {"joint": (-1.0, 1.0)}

    kin = MagicMock()
    kin.nq = 10
    q_smooth = np.zeros((5, 10))
    expected_ref = np.ones((5, 10))
    expected_fits = [MagicMock() for _ in range(5)]
    kin.solve_trajectory.return_value = (expected_ref, expected_fits)

    q_ref, ref_fits = consistency_resolve(
        lane, kin, q_smooth, prior_weight=CONSISTENCY_PRIOR, iterations=30
    )
    assert np.array_equal(q_ref, expected_ref)
    assert ref_fits == expected_fits
    assert kin.solve_trajectory.call_count == 1
    args, kwargs = kin.solve_trajectory.call_args
    assert np.array_equal(args[0], lane.points)
    assert np.array_equal(args[1], lane.valid)
    assert np.array_equal(args[2], q_smooth[0])
    assert kwargs["ground"] is lane.ground
    assert kwargs["prior_weight"] == CONSISTENCY_PRIOR
    assert kwargs["iterations"] == 30
    assert kwargs["flat_feet_per_frame"] == lane.stance
    assert kwargs["plant_stance"] is True
    assert np.array_equal(kwargs["prior_trajectory"], q_smooth)
    assert kwargs["bounds"] == lane.bounds
