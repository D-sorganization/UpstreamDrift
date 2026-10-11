"""Unit contract for shared physical inertia validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.core.physical_inertia import validate_physical_inertia

pytestmark = pytest.mark.unit


def test_valid_rotated_inertia_is_preserved() -> None:
    tensor = np.array([[0.04, 0.0, 0.01], [0.0, 0.05, 0.0], [0.01, 0.0, 0.04]])
    np.testing.assert_array_equal(validate_physical_inertia(tensor), tensor)


@pytest.mark.parametrize(
    ("tensor", "reason"),
    [
        (np.full((2, 2), 1.0), "finite 3x3"),
        (np.array([[1.0, 0.2, 0], [0, 1.0, 0], [0, 0, 1.0]]), "symmetric"),
        (np.diag([0.0, 1.0, 1.0]), "positive definite"),
        (np.diag([0.02, 0.02, 0.08]), "triangle inequality"),
    ],
)
def test_invalid_inertia_fails_closed(tensor: np.ndarray, reason: str) -> None:
    with pytest.raises(PreconditionError, match=reason):
        validate_physical_inertia(tensor)
