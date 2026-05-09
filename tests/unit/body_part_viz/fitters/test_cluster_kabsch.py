"""Unit tests for :class:`ClusterKabschFitter`."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import (
    BindingKind,
    MarkerBinding,
    ShapeFitter,
)
from src.shared.python.body_part_viz.fitters import ClusterKabschFitter

from ._stubs import StubShape


def _binding(names: tuple[str, ...] = ("m0", "m1", "m2", "m3")) -> MarkerBinding:
    return MarkerBinding(kind=BindingKind.CLUSTER, marker_names=names)


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


def _build_cluster(rest_points: np.ndarray, rotations: list[np.ndarray]) -> dict:
    """Apply per-frame ``rotations`` to ``rest_points`` and split per marker."""
    n_frames = len(rotations)
    n_markers = rest_points.shape[0]
    cluster = np.empty((n_frames, n_markers, 3))
    for t, rot in enumerate(rotations):
        cluster[t] = rest_points @ rot.T
    markers: dict = {}
    for j in range(n_markers):
        markers[f"m{j}"] = cluster[:, j, :]
    return markers


def test_implements_shape_fitter_protocol() -> None:
    assert isinstance(ClusterKabschFitter(), ShapeFitter)


def test_recovers_known_z_rotation_within_tolerance() -> None:
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    angle = np.deg2rad(45.0)
    rot_known = _rotation_z(angle)
    markers = _build_cluster(rest, [np.eye(3), rot_known])

    fitted = ClusterKabschFitter().fit(StubShape(), _binding(), markers)

    assert np.allclose(fitted.rotation_matrix[0], np.eye(3), atol=1e-9)
    assert np.allclose(fitted.rotation_matrix[1], rot_known, atol=1e-9)
    # Pure rigid: scale unchanged.
    assert np.allclose(fitted.scale, 1.0)


def test_reflection_guard_returns_proper_rotation() -> None:
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    reflection = np.diag([1.0, 1.0, -1.0])
    markers = _build_cluster(rest, [np.eye(3), reflection])

    fitted = ClusterKabschFitter().fit(StubShape(), _binding(), markers)

    det = float(np.linalg.det(fitted.rotation_matrix[1]))
    assert np.isclose(det, 1.0, atol=1e-9)


def test_isotropic_scale_recovered_when_enabled() -> None:
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    scale_factor = 1.5
    # Two frames: frame 0 = rest (scale 1.0), frame 1 = scaled (scale s).
    cluster = np.stack([rest, rest * scale_factor], axis=0)
    markers = {f"m{j}": cluster[:, j, :] for j in range(4)}

    fitter = ClusterKabschFitter(enable_scale=True)
    fitted = fitter.fit(StubShape(), _binding(), markers)

    assert np.allclose(fitted.scale[0], 1.0, atol=1e-9)
    assert np.allclose(fitted.scale[1], scale_factor, atol=1e-9)


def test_nan_in_one_marker_invalidates_frame() -> None:
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    markers = _build_cluster(rest, [np.eye(3), np.eye(3), np.eye(3)])
    markers["m0"][1] = np.nan

    fitted = ClusterKabschFitter().fit(StubShape(), _binding(), markers)

    assert bool(fitted.valid_mask[0])
    assert not bool(fitted.valid_mask[1])
    assert bool(fitted.valid_mask[2])


def test_leading_nan_frames_use_first_valid_as_rest() -> None:
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    angle = np.deg2rad(30.0)
    rot = _rotation_z(angle)
    markers = _build_cluster(rest, [np.eye(3), np.eye(3), rot])
    # Mark frame 0 invalid via NaN.
    markers["m1"][0] = np.nan

    fitted = ClusterKabschFitter().fit(StubShape(), _binding(), markers)

    assert not bool(fitted.valid_mask[0])
    assert bool(fitted.valid_mask[1])
    # Rest taken from frame 1 (identity); frame 2 should recover rot.
    assert np.allclose(fitted.rotation_matrix[2], rot, atol=1e-9)


def test_wrong_binding_kind_raises_type_error() -> None:
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    with pytest.raises(TypeError, match="CLUSTER"):
        ClusterKabschFitter().fit(
            StubShape(),
            binding,
            {"a": np.zeros((1, 3)), "b": np.zeros((1, 3))},
        )


def test_all_invalid_frames_returns_default() -> None:
    n = 3
    markers = {f"m{j}": np.full((n, 3), np.nan) for j in range(4)}
    fitted = ClusterKabschFitter().fit(StubShape(), _binding(), markers)
    assert not bool(fitted.valid_mask.any())
