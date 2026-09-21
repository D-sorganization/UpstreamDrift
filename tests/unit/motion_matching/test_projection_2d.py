"""Tests for 3D-to-2D projection and swing plane calibration integration (TB-03 #10588)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.projection_2d import (
    project_target_to_calibrated_plane,
    project_to_2d,
)
from src.shared.python.tour_baselines.plane import RigidSwingPlane

pytestmark = pytest.mark.unit


def _dummy_source() -> SourceProvenance:
    return SourceProvenance(
        filename="dummy.c3d",
        format="c3d",
        subject_id="test",
        trial_id="trial1",
        sha256="0" * 64,
    )


def test_project_to_2d_with_calibrated_plane() -> None:
    """Projecting a ClubTarget onto a calibrated swing plane preserves in-plane geometry."""
    n_frames = 20
    times = np.linspace(0.0, 0.2, n_frames)

    # 45 deg tilted plane around X in Y-up world
    origin = np.array([0.0, 1.0, 0.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
    n_axis = np.cross(u_axis, v_axis)

    plane = RigidSwingPlane(origin=origin, normal=n_axis, u_axis=u_axis, v_axis=v_axis)

    # Butt trajectory purely in plane
    butt_3d = np.zeros((n_frames, 3))
    for i in range(n_frames):
        butt_3d[i] = origin + 0.1 * i * u_axis + 0.05 * i * v_axis

    # Clubhead trajectory purely in plane
    head_3d = np.zeros((n_frames, 3))
    for i in range(n_frames):
        head_3d[i] = origin + (0.1 * i + 0.8) * u_axis + (0.05 * i - 0.6) * v_axis

    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1))

    target = ClubTarget(
        time=times,
        butt=butt_3d,
        clubhead=head_3d,
        club_quat=quats,
        impact_idx=10,
        source=_dummy_source(),
    )

    projected, res = project_target_to_calibrated_plane(target, plane)

    assert isinstance(projected, ClubTarget)
    assert res.rmse_m < 1e-12
    # Check that in-plane distances (e.g. shaft length) are preserved
    original_lengths = np.linalg.norm(head_3d - butt_3d, axis=-1)
    projected_lengths = np.linalg.norm(projected.clubhead - projected.butt, axis=-1)
    np.testing.assert_allclose(projected_lengths, original_lengths, atol=1e-12)


def test_project_to_2d_backward_compatibility() -> None:
    """Calling project_to_2d without explicit plane fits a rigid plane from club observations."""
    n_frames = 15
    times = np.linspace(0.0, 0.15, n_frames)
    butt = np.column_stack(
        [np.linspace(0, 1, n_frames), np.zeros(n_frames), np.ones(n_frames)]
    )
    head = np.column_stack(
        [np.linspace(0, 1, n_frames), np.ones(n_frames), np.ones(n_frames)]
    )
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1))

    target = ClubTarget(
        time=times,
        butt=butt,
        clubhead=head,
        club_quat=quats,
        impact_idx=5,
        source=_dummy_source(),
    )
    projected = project_to_2d(target)

    assert isinstance(projected, ClubTarget)
    assert projected.butt.shape == butt.shape
    assert projected.clubhead.shape == head.shape
