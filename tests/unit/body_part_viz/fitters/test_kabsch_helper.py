"""Unit tests for the pure-NumPy Kabsch helper."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz.fitters._kabsch import (
    kabsch_rotation,
    stack_cluster,
)


def _rotation_z(angle: float) -> np.ndarray:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_kabsch_round_trip_recovers_known_rotation() -> None:
    rng = np.random.default_rng(0)
    points = rng.standard_normal((6, 3))
    points -= points.mean(axis=0)
    angle = np.deg2rad(37.5)
    rot_known = _rotation_z(angle)
    rotated = points @ rot_known.T

    rot_recovered = kabsch_rotation(points, rotated)

    assert np.allclose(rot_recovered, rot_known, atol=1e-9)
    assert np.isclose(np.linalg.det(rot_recovered), 1.0, atol=1e-9)


def test_kabsch_reflection_guard_returns_proper_rotation() -> None:
    rng = np.random.default_rng(1)
    points = rng.standard_normal((5, 3))
    points -= points.mean(axis=0)
    reflection = np.diag([1.0, 1.0, -1.0])
    reflected = points @ reflection.T

    rot = kabsch_rotation(points, reflected)

    # The Kabsch reflection guard forces det == +1 even when the data
    # demands a reflection — the closest *proper* rotation is returned.
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)


def test_kabsch_collinear_cluster_returns_finite_rotation() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    centred = points - points.mean(axis=0)
    rot_known = _rotation_z(np.deg2rad(15.0))
    rotated = centred @ rot_known.T

    rot = kabsch_rotation(centred, rotated)

    assert np.all(np.isfinite(rot))
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)


def test_kabsch_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching"):
        kabsch_rotation(np.zeros((4, 3)), np.zeros((5, 3)))


def test_kabsch_rejects_non_3d_columns() -> None:
    with pytest.raises(ValueError, match="3 columns"):
        kabsch_rotation(np.zeros((4, 2)), np.zeros((4, 2)))


def test_kabsch_rejects_non_2d_arrays() -> None:
    with pytest.raises(ValueError, match="2-D"):
        kabsch_rotation(np.zeros(3), np.zeros(3))


def test_stack_cluster_happy_path() -> None:
    markers = {
        "a": np.zeros((5, 3)),
        "b": np.ones((5, 3)),
        "c": np.full((5, 3), 2.0),
    }
    cluster = stack_cluster(markers, ("a", "b", "c"))
    assert cluster.shape == (5, 3, 3)
    assert np.allclose(cluster[0, 1], 1.0)


def test_stack_cluster_missing_marker_raises() -> None:
    with pytest.raises(KeyError, match="missing marker"):
        stack_cluster({"a": np.zeros((2, 3))}, ("a", "b"))


def test_stack_cluster_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        stack_cluster({"a": np.zeros((2, 4))}, ("a",))


def test_stack_cluster_inconsistent_frame_count_raises() -> None:
    markers = {"a": np.zeros((4, 3)), "b": np.zeros((5, 3))}
    with pytest.raises(ValueError, match="frames"):
        stack_cluster(markers, ("a", "b"))
