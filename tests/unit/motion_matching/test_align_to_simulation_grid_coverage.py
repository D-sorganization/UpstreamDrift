"""Coverage tests for ``align_to_simulation_grid``."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.align_to_simulation_grid import (
    AlignedTrajectory,
    align_to_simulation_grid,
)


def _raw(n: int = 200, t_end: float = 0.5):
    raw_time = np.linspace(0.0, t_end, n)
    speed = np.sin(np.pi * raw_time / t_end)
    butt = np.zeros((n, 3))
    clubhead = np.column_stack([speed, np.zeros(n), np.zeros(n)])
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return raw_time, butt, clubhead, quat


def test_returns_aligned_trajectory() -> None:
    """Pin: success returns an :class:`AlignedTrajectory`."""
    raw_time, butt, clubhead, quat = _raw()
    out = align_to_simulation_grid(raw_time, butt, clubhead, quat)
    assert isinstance(out, AlignedTrajectory)
    assert out.time.shape == out.butt.shape[:1]
    assert out.club_quat.shape[1] == 4


def test_too_few_samples_rejected() -> None:
    """Pin: < 2 samples rejected."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        align_to_simulation_grid(
            np.array([0.0]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.tile([1.0, 0.0, 0.0, 0.0], (1, 1)),
        )


def test_non_monotonic_rejected() -> None:
    """Pin: non-strictly-increasing raw_time is rejected."""
    raw_time = np.array([0.0, 0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="strictly increasing"):
        align_to_simulation_grid(
            raw_time,
            np.zeros((4, 3)),
            np.zeros((4, 3)),
            np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        )


def test_explicit_impact_idx_used() -> None:
    """Pin: ``impact_idx_raw`` overrides the auto-detect path."""
    raw_time, butt, clubhead, quat = _raw()
    out = align_to_simulation_grid(raw_time, butt, clubhead, quat, impact_idx_raw=10)
    # impact_out comes back as 1-based index on the simulation grid.
    assert out.impact_idx >= 1
