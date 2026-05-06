"""Unit tests for impact detection and resampling."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.club_target import AlignOptions
from src.shared.python.motion_matching.loaders._align import (
    detect_impact_index,
    resample_target,
)


def test_align_synthetic_centers_impact() -> None:
    n_raw = 200
    raw_time = np.linspace(0.0, 0.5, n_raw)
    # Speed peaks somewhere mid-trajectory; build clubhead so impact is at idx 100.
    speed_profile = np.exp(-((np.arange(n_raw) - 100.0) ** 2) / (2 * 25.0**2))
    raw_clubhead = (
        np.cumsum(
            np.column_stack([speed_profile, np.zeros(n_raw), np.zeros(n_raw)]),
            axis=0,
        )
        * 0.001
    )
    raw_butt = raw_clubhead - np.array([0.0, 0.0, 1.1])
    raw_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_raw, 1))

    impact_raw = detect_impact_index(raw_time, raw_clubhead)
    assert abs(impact_raw - 100) <= 2

    opts = AlignOptions(
        sample_rate_hz=1000.0,
        simulation_time_s=0.3,
        time_alignment="impact",
        impact_target_t_s=0.25,
    )
    sim_time, butt, head, quat, impact_idx = resample_target(
        raw_time, raw_butt, raw_clubhead, raw_quat, impact_raw, opts
    )
    assert sim_time.shape[0] == 301
    assert sim_time[0] == pytest.approx(0.0)
    assert sim_time[-1] == pytest.approx(0.3)
    assert butt.shape == (301, 3)
    assert head.shape == (301, 3)
    assert quat.shape == (301, 4)
    # Impact index in 1-based output corresponds to t = 0.25 s.
    assert sim_time[impact_idx - 1] == pytest.approx(0.25, abs=1.5e-3)


def test_align_resample_preserves_endpoints() -> None:
    n_raw = 50
    raw_time = np.linspace(0.0, 0.3, n_raw)
    raw_butt = np.column_stack([raw_time, np.zeros(n_raw), np.zeros(n_raw)])
    raw_clubhead = raw_butt + np.array([0.0, 0.0, 1.1])
    raw_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_raw, 1))
    opts = AlignOptions(
        sample_rate_hz=500.0,
        simulation_time_s=0.3,
        time_alignment="none",
        impact_target_t_s=0.25,
    )
    sim_time, butt, head, quat, _ = resample_target(
        raw_time, raw_butt, raw_clubhead, raw_quat, 25, opts
    )
    assert butt[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert butt[-1, 0] == pytest.approx(0.3, abs=1e-3)
    assert np.all(np.linalg.norm(quat, axis=1) == pytest.approx(1.0, abs=1e-9))


def test_align_address_alignment_starts_at_zero() -> None:
    n_raw = 30
    raw_time = np.linspace(0.5, 1.0, n_raw)  # not starting at 0
    raw_clubhead = np.column_stack(
        [np.linspace(0.0, 0.1, n_raw), np.zeros(n_raw), np.zeros(n_raw)]
    )
    raw_butt = raw_clubhead - np.array([0.0, 0.0, 1.1])
    raw_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_raw, 1))
    opts = AlignOptions(time_alignment="address")
    sim_time, butt, _, _, _ = resample_target(
        raw_time, raw_butt, raw_clubhead, raw_quat, 10, opts
    )
    assert sim_time[0] == pytest.approx(0.0)


def test_detect_impact_requires_two_samples() -> None:
    with pytest.raises(ValueError):
        detect_impact_index(np.array([0.0]), np.zeros((1, 3)))
