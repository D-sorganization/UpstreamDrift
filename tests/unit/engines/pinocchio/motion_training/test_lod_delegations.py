"""Tests for the LOD-driven delegating accessors added under issue #4139.

These tests pin down the new `ClubTrajectory` and `TrajectoryIKResult`
properties so future changes can't silently regress the LOD refactor.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_club_trajectory_delegates_event_markers() -> None:
    """ClubTrajectory exposes flat accessors for SwingEventMarkers fields."""
    try:
        from src.engines.physics_engines.pinocchio.python.motion_training.club_trajectory_parser import (  # noqa: E501
            ClubTrajectory,
            SwingEventMarkers,
        )
    except ImportError as e:  # pragma: no cover - dependency-gated
        pytest.skip(f"motion_training package not importable: {e}")

    events = SwingEventMarkers(
        address=10,
        top=50,
        impact=80,
        finish=120,
        club_head_speed=95.5,
    )
    traj = ClubTrajectory(events=events)

    assert traj.address_frame == 10
    assert traj.top_frame == 50
    assert traj.impact_frame == 80
    assert traj.finish_frame == 120
    assert traj.club_head_speed_mph == pytest.approx(95.5)


@pytest.mark.unit
def test_trajectory_ik_result_q_dim() -> None:
    """TrajectoryIKResult.q_dim returns the per-frame configuration size."""
    try:
        from src.engines.physics_engines.pinocchio.python.motion_training.dual_hand_ik_solver import (  # noqa: E501
            TrajectoryIKResult,
        )
    except ImportError as e:  # pragma: no cover - dependency-gated
        pytest.skip(f"dual_hand_ik_solver not importable: {e}")

    cfg = [np.zeros(7), np.ones(7), np.zeros(7)]
    result = TrajectoryIKResult(configurations=cfg)

    assert result.q_dim == 7
    assert result.q_trajectory.shape == (3, 7)


@pytest.mark.unit
def test_trajectory_ik_result_q_dim_empty() -> None:
    """An empty result reports q_dim == 0 rather than raising."""
    try:
        from src.engines.physics_engines.pinocchio.python.motion_training.dual_hand_ik_solver import (  # noqa: E501
            TrajectoryIKResult,
        )
    except ImportError as e:  # pragma: no cover - dependency-gated
        pytest.skip(f"dual_hand_ik_solver not importable: {e}")

    result = TrajectoryIKResult()

    assert result.q_dim == 0
