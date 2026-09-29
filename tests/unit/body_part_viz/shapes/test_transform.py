"""Tests for the shared _transform helper edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz.shapes._transform import (
    apply_fitted_to_rest_vertices,
)

from ._helpers import make_identity_fitted


def test_transform_rejects_bad_rest_shape() -> None:
    fitted = make_identity_fitted("x", n_frames=1)
    with pytest.raises(ValueError):
        apply_fitted_to_rest_vertices(np.zeros((4, 2)), fitted)


def test_transform_zero_frames_returns_empty_nan_block() -> None:
    fitted = make_identity_fitted("x", n_frames=0)
    out = apply_fitted_to_rest_vertices(np.zeros((3, 3)), fitted)
    assert out.shape == (0, 3, 3)
