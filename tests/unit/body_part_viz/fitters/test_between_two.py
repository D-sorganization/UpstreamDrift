"""Unit tests for :class:`BetweenTwoMarkersFitter`."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import (
    BindingKind,
    MarkerBinding,
    ShapeFitter,
)
from src.shared.python.body_part_viz.fitters import BetweenTwoMarkersFitter

from ._stubs import StubShape


def _binding(rest: tuple[float, ...] = (1.0,)) -> MarkerBinding:
    return MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("a", "b"),
        rest_dimensions=rest,
    )


def test_implements_shape_fitter_protocol() -> None:
    assert isinstance(BetweenTwoMarkersFitter(), ShapeFitter)


def test_straight_line_centroids_are_midpoints() -> None:
    n = 100
    a = np.zeros((n, 3))
    b = np.zeros((n, 3))
    a[:, 0] = np.linspace(0.0, 1.0, n)
    b[:, 0] = a[:, 0] + 1.0  # constant 1.0 spacing along x

    fitted = BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": a, "b": b})

    assert fitted.centroid.shape == (n, 3)
    assert np.allclose(fitted.centroid, 0.5 * (a + b))
    assert bool(fitted.valid_mask.all())


def test_rotation_matrix_is_orthogonal_with_unit_determinant() -> None:
    n = 20
    a = np.zeros((n, 3))
    b = np.zeros((n, 3))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    b[:, 0] = np.cos(angles)
    b[:, 1] = np.sin(angles)

    fitted = BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": a, "b": b})

    for t in range(n):
        rot = fitted.rotation_matrix[t]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)


def test_rotation_first_column_is_axis() -> None:
    a = np.zeros((1, 3))
    b = np.array([[3.0, 4.0, 0.0]])  # length 5
    fitted = BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": a, "b": b})

    expected_axis = np.array([0.6, 0.8, 0.0])
    assert np.allclose(fitted.rotation_matrix[0, :, 0], expected_axis)
    # Length-only anisotropic scale
    assert np.isclose(fitted.scale[0, 0], 5.0)
    assert np.isclose(fitted.scale[0, 1], 1.0)
    assert np.isclose(fitted.scale[0, 2], 1.0)


def test_axis_near_world_z_uses_y_up() -> None:
    a = np.zeros((1, 3))
    b = np.array([[0.0, 0.0, 2.0]])
    fitted = BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": a, "b": b})

    rot = fitted.rotation_matrix[0]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)


def test_nan_marker_marks_frame_invalid() -> None:
    n = 100
    a = np.zeros((n, 3))
    b = np.zeros((n, 3))
    a[:, 0] = np.linspace(0.0, 1.0, n)
    b[:, 0] = a[:, 0] + 1.0
    a[50:61, :] = np.nan

    fitted = BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": a, "b": b})

    assert not bool(fitted.valid_mask[50:61].any())
    assert bool(fitted.valid_mask[:50].all())
    assert bool(fitted.valid_mask[61:].all())


def test_rest_length_falls_back_to_shape_when_binding_empty() -> None:
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    a = np.zeros((1, 3))
    b = np.array([[2.0, 0.0, 0.0]])
    fitted = BetweenTwoMarkersFitter().fit(
        StubShape(rest_dimensions=(4.0,)), binding, {"a": a, "b": b}
    )
    assert np.isclose(fitted.scale[0, 0], 0.5)


def test_missing_rest_length_raises() -> None:
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    a = np.zeros((1, 3))
    b = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="rest length"):
        BetweenTwoMarkersFitter().fit(
            StubShape(rest_dimensions=()), binding, {"a": a, "b": b}
        )


def test_wrong_binding_kind_raises_type_error() -> None:
    binding = MarkerBinding(kind=BindingKind.CLUSTER, marker_names=("a", "b", "c"))
    with pytest.raises(TypeError, match="BETWEEN_TWO"):
        BetweenTwoMarkersFitter().fit(
            StubShape(),
            binding,
            {"a": np.zeros((1, 3)), "b": np.zeros((1, 3)), "c": np.zeros((1, 3))},
        )


def test_missing_marker_raises() -> None:
    with pytest.raises(KeyError, match="missing marker"):
        BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": np.zeros((1, 3))})


def test_marker_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        BetweenTwoMarkersFitter().fit(
            StubShape(),
            _binding(),
            {"a": np.zeros((1, 2)), "b": np.zeros((1, 2))},
        )


def test_marker_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share shape"):
        BetweenTwoMarkersFitter().fit(
            StubShape(),
            _binding(),
            {"a": np.zeros((1, 3)), "b": np.zeros((2, 3))},
        )


def test_zero_length_segment_raises() -> None:
    a = np.zeros((1, 3))
    b = np.zeros((1, 3))
    with pytest.raises(ValueError, match="coincide"):
        BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": a, "b": b})


def test_all_invalid_frames_returns_zero_centroid() -> None:
    n = 4
    a = np.full((n, 3), np.nan)
    b = np.full((n, 3), np.nan)
    fitted = BetweenTwoMarkersFitter().fit(StubShape(), _binding(), {"a": a, "b": b})
    assert not bool(fitted.valid_mask.any())
    assert np.allclose(fitted.centroid, 0.0)
