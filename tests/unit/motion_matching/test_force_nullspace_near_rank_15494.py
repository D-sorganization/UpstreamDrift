"""Synthetic numerical-rank and inconsistent-load checks for #15494.

These equations are numerical fixtures, not an admitted swing or load history.
"""

import numpy as np
import pytest

from src.shared.python.motion_matching.force_nullspace import ForceNullSpace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["svd", "qr"])
@pytest.mark.parametrize(("small", "rank"), [(2e-6, 2), (5e-7, 1)])
def test_near_rank_loss_obeys_declared_tolerances(method, small, rank):
    # The third variable is a true null direction. The second becomes a
    # numerical null direction only when its singular value crosses rank_rtol.
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, small, 0.0]])
    space = ForceNullSpace.from_balance(
        matrix,
        np.array([1.0, 0.0]),
        method=method,
        rank_rtol=1e-6,
        balance_tolerance=1e-6,
    )
    analysis = space.analyze()
    assert analysis.rank == rank
    assert analysis.nullity == 3 - rank
    assert analysis.equality_residual <= 1e-6
    assert analysis.null_residual <= 1e-6
    np.testing.assert_allclose(matrix @ space.particular, [1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("method", ["svd", "qr"])
def test_rank_truncated_load_is_rejected_instead_of_repaired(method):
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 5e-7, 0.0]])
    # In the declared rank-one model the second equation has an unavoidable
    # residual of 1e-3, much larger than the 1e-6 balance tolerance.
    with pytest.raises(ValueError, match="inconsistent dynamics equality"):
        ForceNullSpace.from_balance(
            matrix,
            np.array([1.0, 1e-3]),
            method=method,
            rank_rtol=1e-6,
            balance_tolerance=1e-6,
        )
