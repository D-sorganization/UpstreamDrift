"""Tests for ``body_part_viz._types.FittedShape``.

Covers shape invariants on the centroid / rotation / scale / valid_mask
arrays, dtype enforcement, and the n_frames property.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz._types import FittedShape
from src.shared.python.body_part_viz.bindings import BindingKind, MarkerBinding


def _example_binding() -> MarkerBinding:
    return MarkerBinding(BindingKind.BETWEEN_TWO, ("a", "b"))


def _identity_fit(n_frames: int) -> FittedShape:
    """Build a no-op identity FittedShape with ``n_frames`` frames."""
    return FittedShape(
        shape_id="example",
        binding=_example_binding(),
        centroid=np.zeros((n_frames, 3)),
        rotation_matrix=np.tile(np.eye(3), (n_frames, 1, 1)),
        scale=np.ones((n_frames, 3)),
        valid_mask=np.ones(n_frames, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_happy_path_construction() -> None:
    fs = _identity_fit(10)
    assert fs.n_frames == 10
    assert fs.shape_id == "example"


@pytest.mark.unit
def test_zero_frames_is_valid() -> None:
    """A 0-frame fit (e.g. empty trajectory) is a degenerate but valid case."""
    fs = _identity_fit(0)
    assert fs.n_frames == 0


@pytest.mark.unit
def test_n_frames_matches_centroid_shape() -> None:
    fs = _identity_fit(42)
    assert fs.n_frames == 42


# ---------------------------------------------------------------------------
# Frozen invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fitted_shape_is_frozen() -> None:
    fs = _identity_fit(1)
    with pytest.raises(Exception):  # noqa: B017
        fs.shape_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DbC: shape_id and binding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rejects_empty_shape_id() -> None:
    with pytest.raises(ValueError, match="shape_id must be a non-empty string"):
        FittedShape(
            shape_id="",
            binding=_example_binding(),
            centroid=np.zeros((1, 3)),
            rotation_matrix=np.tile(np.eye(3), (1, 1, 1)),
            scale=np.ones((1, 3)),
            valid_mask=np.ones(1, dtype=bool),
        )


@pytest.mark.unit
def test_rejects_non_marker_binding() -> None:
    with pytest.raises(TypeError, match="binding must be a MarkerBinding"):
        FittedShape(
            shape_id="x",
            binding="not a binding",  # type: ignore[arg-type]
            centroid=np.zeros((1, 3)),
            rotation_matrix=np.tile(np.eye(3), (1, 1, 1)),
            scale=np.ones((1, 3)),
            valid_mask=np.ones(1, dtype=bool),
        )


# ---------------------------------------------------------------------------
# DbC: array shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rejects_non_numpy_arrays() -> None:
    with pytest.raises(TypeError, match="centroid must be a numpy.ndarray"):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=[[0, 0, 0]],  # type: ignore[arg-type]
            rotation_matrix=np.tile(np.eye(3), (1, 1, 1)),
            scale=np.ones((1, 3)),
            valid_mask=np.ones(1, dtype=bool),
        )


@pytest.mark.unit
def test_rejects_centroid_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"centroid must have shape \(T, 3\)"):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=np.zeros((5, 4)),
            rotation_matrix=np.tile(np.eye(3), (5, 1, 1)),
            scale=np.ones((5, 3)),
            valid_mask=np.ones(5, dtype=bool),
        )


@pytest.mark.unit
def test_rejects_mismatched_rotation_matrix() -> None:
    with pytest.raises(ValueError, match=r"rotation_matrix must have shape"):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=np.zeros((5, 3)),
            rotation_matrix=np.tile(np.eye(3), (3, 1, 1)),  # wrong T
            scale=np.ones((5, 3)),
            valid_mask=np.ones(5, dtype=bool),
        )


@pytest.mark.unit
def test_rejects_mismatched_scale() -> None:
    with pytest.raises(ValueError, match=r"scale must have shape"):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=np.zeros((5, 3)),
            rotation_matrix=np.tile(np.eye(3), (5, 1, 1)),
            scale=np.ones((3, 3)),  # wrong T
            valid_mask=np.ones(5, dtype=bool),
        )


@pytest.mark.unit
def test_rejects_mismatched_valid_mask() -> None:
    with pytest.raises(ValueError, match=r"valid_mask must have shape"):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=np.zeros((5, 3)),
            rotation_matrix=np.tile(np.eye(3), (5, 1, 1)),
            scale=np.ones((5, 3)),
            valid_mask=np.ones(3, dtype=bool),  # wrong T
        )


# ---------------------------------------------------------------------------
# DbC: dtypes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_valid_mask_must_be_bool_dtype() -> None:
    with pytest.raises(TypeError, match="valid_mask must have dtype=bool"):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=np.zeros((1, 3)),
            rotation_matrix=np.tile(np.eye(3), (1, 1, 1)),
            scale=np.ones((1, 3)),
            valid_mask=np.ones(1, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# DbC: scale positivity (only on valid frames)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scale_must_be_positive_on_valid_frames() -> None:
    scale = np.ones((3, 3))
    scale[1, 0] = -0.1  # invalid: negative
    with pytest.raises(
        ValueError, match="scale entries on valid frames must be strictly positive"
    ):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=np.zeros((3, 3)),
            rotation_matrix=np.tile(np.eye(3), (3, 1, 1)),
            scale=scale,
            valid_mask=np.ones(3, dtype=bool),
        )


@pytest.mark.unit
def test_scale_must_be_finite_on_valid_frames() -> None:
    scale = np.ones((3, 3))
    scale[1, 0] = np.nan
    with pytest.raises(
        ValueError, match="scale entries on valid frames must be finite"
    ):
        FittedShape(
            shape_id="x",
            binding=_example_binding(),
            centroid=np.zeros((3, 3)),
            rotation_matrix=np.tile(np.eye(3), (3, 1, 1)),
            scale=scale,
            valid_mask=np.ones(3, dtype=bool),
        )


@pytest.mark.unit
def test_scale_invalid_value_ok_on_invalid_frames() -> None:
    """Negative / nan scale on an invalid frame is allowed (it's not used)."""
    scale = np.ones((3, 3))
    scale[1, 0] = -1.0
    scale[2, 1] = np.nan
    valid = np.array([True, False, False])

    # No exception — the bad scales are masked out.
    fs = FittedShape(
        shape_id="x",
        binding=_example_binding(),
        centroid=np.zeros((3, 3)),
        rotation_matrix=np.tile(np.eye(3), (3, 1, 1)),
        scale=scale,
        valid_mask=valid,
    )
    assert fs.n_frames == 3


@pytest.mark.unit
def test_zero_frames_skips_scale_check() -> None:
    """T=0 must not crash the scale-positivity check."""
    fs = FittedShape(
        shape_id="x",
        binding=_example_binding(),
        centroid=np.zeros((0, 3)),
        rotation_matrix=np.zeros((0, 3, 3)),
        scale=np.zeros((0, 3)),
        valid_mask=np.zeros(0, dtype=bool),
    )
    assert fs.n_frames == 0
