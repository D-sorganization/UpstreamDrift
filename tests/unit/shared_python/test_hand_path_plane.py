"""Unit tests for biomechanics.shallowing.hand_path_plane (Phase 1, epic #5422).

Tests are written TDD-first (red phase before implementation).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.shallowing.hand_path_plane import (
    Plane3D,
    compute_hand_path_plane,
    extract_lead_hand_trajectory,
)

# ---------------------------------------------------------------------------
# compute_hand_path_plane
# ---------------------------------------------------------------------------


def test_horizontal_plane_residuals_near_zero() -> None:
    """Points on z=0 -> residuals approx 0."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]],
        dtype=float,
    )
    plane = compute_hand_path_plane(pts)
    assert plane.residuals < 1e-10


def test_horizontal_plane_normal_vertical() -> None:
    """Horizontal plane -> normal approx (0, 0, 1)."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=float,
    )
    plane = compute_hand_path_plane(pts)
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-6  # normal approx (0, 0, 1)


def test_known_tilted_plane() -> None:
    """Points on plane z = x -> normal in xz-plane at 45 deg.

    The plane z = x has equation x - z = 0, so normal is (-1, 0, 1)/sqrt(2)
    (or its negation). Points must span the plane, not just a line on it.
    """
    # Build a 2-D grid of points on the plane z = x:
    # direction 1: (1, 0, 1)/sqrt(2), direction 2: (0, 1, 0)
    t = np.linspace(0, 1, 5)
    s = np.linspace(0, 1, 5)
    T, S = np.meshgrid(t, s)
    # x=T, y=S, z=T (so z=x on this grid, spanning both directions)
    pts = np.column_stack([T.ravel(), S.ravel(), T.ravel()])
    plane = compute_hand_path_plane(pts)
    expected_normal = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2)
    assert np.allclose(np.abs(plane.normal @ expected_normal), 1.0, atol=0.01)


def test_minimum_points_enforced() -> None:
    """Fewer than 3 points raises ValueError mentioning '3'."""
    with pytest.raises(ValueError, match="3"):
        compute_hand_path_plane(np.array([[0, 0, 0], [1, 0, 0]], dtype=float))


def test_upward_hemisphere_convention() -> None:
    """Returned normal always has non-negative z-component."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=float,
    )
    plane = compute_hand_path_plane(pts)
    assert plane.normal[2] >= 0


def test_centroid_is_point_on_plane() -> None:
    """point_on_plane is the centroid of the input points."""
    pts = np.array(
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=float,
    )
    plane = compute_hand_path_plane(pts)
    assert np.allclose(plane.point_on_plane, pts.mean(axis=0))


def test_plane3d_is_dataclass() -> None:
    """Plane3D can be constructed directly."""
    p = Plane3D(
        normal=np.array([0.0, 0.0, 1.0]),
        point_on_plane=np.zeros(3),
        residuals=0.0,
    )
    assert p.residuals == 0.0


def test_exactly_three_points_accepted() -> None:
    """Exactly 3 non-collinear points should not raise."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    plane = compute_hand_path_plane(pts)
    assert plane.normal is not None


def test_normal_is_unit_vector() -> None:
    """Returned normal has unit length."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=float,
    )
    plane = compute_hand_path_plane(pts)
    assert abs(np.linalg.norm(plane.normal) - 1.0) < 1e-10


def test_residuals_are_non_negative() -> None:
    """residuals field is always >= 0."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((20, 3))
    plane = compute_hand_path_plane(pts)
    assert plane.residuals >= 0.0


# ---------------------------------------------------------------------------
# extract_lead_hand_trajectory
# ---------------------------------------------------------------------------


def _make_sim_frames(
    key: str = "lead_hand",
    n: int = 5,
) -> list[object]:
    """Return a list of minimal stub objects with joint_positions dict."""

    class _Frame:
        def __init__(self, pos: np.ndarray) -> None:
            self.joint_positions: dict[str, np.ndarray] = {key: pos}

    return [_Frame(np.array([float(i), 0.0, 0.0])) for i in range(n)]


def test_extract_lead_hand_returns_correct_shape() -> None:
    """extract_lead_hand_trajectory returns (N, 3) array."""
    frames = _make_sim_frames("lead_hand", n=7)
    traj = extract_lead_hand_trajectory(frames)
    assert traj.shape == (7, 3)


def test_extract_lead_hand_fallback_left_wrist() -> None:
    """Falls back to 'left_wrist' when 'lead_hand' is absent."""
    frames = _make_sim_frames("left_wrist", n=4)
    traj = extract_lead_hand_trajectory(frames)
    assert traj.shape == (4, 3)


def test_extract_lead_hand_fallback_wrist_l() -> None:
    """Falls back to 'wrist_L' when neither 'lead_hand' nor 'left_wrist' present."""
    frames = _make_sim_frames("wrist_L", n=3)
    traj = extract_lead_hand_trajectory(frames)
    assert traj.shape == (3, 3)


def test_extract_lead_hand_empty_raises() -> None:
    """Empty frame list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        extract_lead_hand_trajectory([])


def test_extract_lead_hand_missing_key_raises() -> None:
    """Missing lead hand marker key raises ValueError listing available keys."""

    class _BadFrame:
        joint_positions: dict[str, np.ndarray] = {"right_wrist": np.zeros(3)}

    with pytest.raises(ValueError, match="Available"):
        extract_lead_hand_trajectory([_BadFrame()])


def test_extract_positions_are_correct() -> None:
    """Extracted positions match the joint_positions values."""
    frames = _make_sim_frames("lead_hand", n=3)
    traj = extract_lead_hand_trajectory(frames)
    for i, frame in enumerate(frames):
        assert np.allclose(traj[i], frame.joint_positions["lead_hand"])


# ---------------------------------------------------------------------------
# Shape validation tests (issue #5566)
# ---------------------------------------------------------------------------


class TestComputeHandPathPlaneShapeValidation:
    """Tests for the (N, 3) shape guard introduced in #5566."""

    def test_rejects_wrong_column_count(self) -> None:
        """(N, 2) input must raise ValueError mentioning 'shape'."""
        bad = np.random.rand(5, 2)
        with pytest.raises(ValueError, match="shape"):
            compute_hand_path_plane(bad)

    def test_rejects_1d_array(self) -> None:
        """1-D input must raise ValueError mentioning 'shape'."""
        bad = np.random.rand(9)
        with pytest.raises(ValueError, match="shape"):
            compute_hand_path_plane(bad)

    def test_rejects_3d_array(self) -> None:
        """3-D input must raise ValueError mentioning 'shape'."""
        bad = np.random.rand(5, 3, 2)
        with pytest.raises(ValueError, match="shape"):
            compute_hand_path_plane(bad)

    def test_rejects_too_few_points(self) -> None:
        """(2, 3) input must raise ValueError mentioning 'at least 3 points'."""
        bad = np.random.rand(2, 3)
        with pytest.raises(ValueError, match="at least 3 points"):
            compute_hand_path_plane(bad)

    def test_rejects_four_column_array(self) -> None:
        """(N, 4) input must raise ValueError mentioning 'shape'."""
        bad = np.random.rand(5, 4)
        with pytest.raises(ValueError, match="shape"):
            compute_hand_path_plane(bad)
