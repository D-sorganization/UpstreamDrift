from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.vector_math import row_euclidean_norm


def test_row_euclidean_norm_matches_numpy_axis_norm() -> None:
    values = np.array(
        [
            [3.0, 4.0, 0.0],
            [1.5, -2.0, 6.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        row_euclidean_norm(values), np.linalg.norm(values, axis=1)
    )


def test_row_euclidean_norm_accepts_non_contiguous_views() -> None:
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    view = base[:, ::2]

    np.testing.assert_allclose(row_euclidean_norm(view), np.linalg.norm(view, axis=1))


def test_row_euclidean_norm_rejects_non_matrix_inputs() -> None:
    with pytest.raises(ValueError, match="2-D"):
        row_euclidean_norm(np.array([1.0, 2.0, 3.0]))


def test_row_euclidean_norm_rejects_complex_inputs() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        row_euclidean_norm(np.array([[1.0 + 1.0j, 2.0 + 0.0j]]))
