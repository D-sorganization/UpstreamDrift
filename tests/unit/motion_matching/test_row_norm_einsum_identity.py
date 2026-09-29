"""Row/column norm identity: ``sqrt(einsum)`` equals ``np.linalg.norm``.

Covers the axis patterns converted in the motion-matching einsum
micro-optimisation (axis=0, axis=1, axis=2, and the ``keepdims`` variant),
including NaN/Inf and degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest


def _mixed_float_arrays(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    x = rng.normal(size=shape).astype(np.float64) * 1e3
    # Sprinkle non-finite sentinels through full rows so NaN/Inf propagation
    # matches np.linalg.norm's per-row semantics.
    x[0] = np.nan
    x[1] = np.inf
    x[2, 0] = np.nan
    x[3, -1] = np.inf
    return x


@pytest.mark.parametrize(
    ("shape", "axis", "pattern"),
    [
        ((17, 3), 1, "ij,ij->i"),
        ((13, 7), 0, "ij,ij->j"),
        ((11, 9, 3), 2, "...i,...i->..."),
        ((13, 7, 2), 2, "...i,...i->..."),
    ],
)
def test_row_norm_einsum_identity(shape: tuple[int, ...], axis: int, pattern: str) -> None:
    rng = np.random.default_rng(sum(shape))
    x = _mixed_float_arrays(rng, shape)

    expected = np.linalg.norm(x, axis=axis)
    actual = np.sqrt(np.einsum(pattern, x, x))

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0, equal_nan=True)
    if axis != 0:
        assert not np.isnan(actual).all()  # NaN confined to the sentinel rows


def test_row_norm_einsum_identity_keepdims() -> None:
    """``keepdims=True`` on axis=1 matches sqrt(einsum) plus a trailing axis."""
    shape = (13, 3)
    rng = np.random.default_rng(sum(shape))
    x = _mixed_float_arrays(rng, shape)

    expected = np.linalg.norm(x, axis=1, keepdims=True)
    actual = np.sqrt(np.einsum("ij,ij->i", x, x))[:, None]

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0, equal_nan=True)
    assert actual.shape == expected.shape


def test_row_norm_einsum_identity_edge_shapes() -> None:
    """Empty axis and single-row arrays keep norm/einsum in agreement."""
    empty_rows = np.empty((0, 3))
    np.testing.assert_allclose(
        np.sqrt(np.einsum("ij,ij->i", empty_rows, empty_rows)),
        np.linalg.norm(empty_rows, axis=1),
    )

    rng = np.random.default_rng(11)
    single = rng.normal(size=(1, 3))
    np.testing.assert_allclose(
        np.sqrt(np.einsum("ij,ij->i", single, single)),
        np.linalg.norm(single, axis=1),
    )