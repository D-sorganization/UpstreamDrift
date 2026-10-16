"""NSX-61 (#12586): ``reproject_onto_closure`` delegates to ``project_to_closure``."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.nullspace_proposals import (
    reproject_onto_closure,
)

pytestmark = pytest.mark.unit


class _CirclePlant:
    """Closure ``|q|^2 = 1`` in the plane."""

    def closure_pose_residual(self, q):
        return np.array([q @ q - 1.0])

    def closure_rate_matrix(self, q):
        return 2.0 * q[None, :]


def test_plant_projects_onto_the_manifold():
    q = np.array([2.0, 1.0])
    out = reproject_onto_closure(
        q, closure_residual_m=4.0, tol_m=1e-3, plant=_CirclePlant()
    )
    assert abs(out @ out - 1.0) < 1e-10
    # The nearest-point direction is kept: the old stub only rescaled toward 0.
    assert np.allclose(out / np.linalg.norm(out), q / np.linalg.norm(q), atol=1e-6)


def test_within_tolerance_is_returned_unchanged_even_with_plant():
    q = np.array([1.0, 0.0])
    out = reproject_onto_closure(
        q, closure_residual_m=0.0, tol_m=1e-3, plant=_CirclePlant()
    )
    assert np.array_equal(out, q) and out is not q


def test_large_residual_without_a_plant_fails_closed():
    with pytest.raises(ValueError, match="plant"):
        reproject_onto_closure(np.array([2.0, 1.0]), closure_residual_m=4.0, tol_m=1e-3)
