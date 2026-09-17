"""Engine-independent validation of native model rigid transforms."""

from typing import Any

import numpy as np


def transform(value: Any) -> np.ndarray:
    """Validate a proper finite homogeneous rigid transform."""
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Invalid native rigid transform")
    return matrix
