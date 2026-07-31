"""Numerical equivalence gate for the ``Bolt`` micro-optimization PRs.

Each Bolt PR swaps a numerical kernel for a faster spelling. The swaps are *not*
automatically bit-identical: ``np.sum``/``np.mean`` use pairwise summation, while
``np.vdot``/``np.einsum`` dispatch to BLAS dot products with a different accumulation
order. In biomechanics and thermodynamics code that difference is worth proving
bounded rather than assuming.

Contract per kernel (DbC):
  precondition  - inputs are finite float64 arrays of the documented shape
  postcondition - optimized(x) == original(x) within the stated tolerance
  invariant     - the swap must not change overflow/underflow behaviour for
                  magnitudes the callers actually produce

Each test states its tolerance explicitly. ``EXACT`` means bit-identical and is
asserted with ``==``; anything else carries a documented relative tolerance.

Run:  pytest test_bolt_kernel_equivalence.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core_only,
    pytest.mark.headless_safe,
]

RNG = np.random.default_rng(20260726)

# Relative tolerance for kernels that change summation order. 1e-12 is ~1000x
# looser than float64 eps (2.2e-16) yet ~1e6x tighter than any physical
# tolerance in the callers (solver residuals are compared against 1e-6..1e-8).
REORDER_RTOL = 1e-12


def _vectors(n: int, size: int, scale: float = 1.0) -> list[np.ndarray]:
    return [RNG.standard_normal(size) * scale for _ in range(n)]


# ---------------------------------------------------------------------------
# PR #8093 / #8108 - constraint_solver: np.linalg.norm(x) -> math.sqrt(np.vdot(x, x))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 1000])
def test_norm_vs_sqrt_vdot_matches(size: int) -> None:
    for v in _vectors(25, size):
        original = float(np.linalg.norm(v))
        optimized = math.sqrt(np.vdot(v, v))
        assert optimized == pytest.approx(
            original, rel=REORDER_RTOL
        ), f"size={size} original={original!r} optimized={optimized!r}"


def test_norm_vs_sqrt_vdot_overflow_behaviour_is_identical() -> None:
    """Both spellings overflow at the same point - the swap adds no new failure mode.

    This test originally asserted the opposite: that ``np.linalg.norm`` scales
    internally and stays finite where the ``vdot`` form overflows. That is false for
    this NumPy build - ``norm`` computes ``sqnorm = x.dot(x)`` and overflows too
    (it emits ``RuntimeWarning: overflow encountered in dot``). Pinning the real
    behaviour: neither form is overflow-safe, so the Bolt swap does not regress
    anything. Callers needing overflow safety must use ``math.hypot`` (see below).
    """
    huge = np.array([1e200, 1e200, 1e200])
    with np.errstate(over="ignore"):
        original = np.linalg.norm(huge)
        optimized = np.sqrt(np.vdot(huge, huge))
    assert not np.isfinite(original)
    assert not np.isfinite(optimized)
    assert np.isinf(original) and np.isinf(optimized)


def test_norm_vs_sqrt_vdot_safe_at_realistic_physics_magnitudes() -> None:
    """Constraint residuals and pendulum states stay far below the overflow regime."""
    for scale in (1e-12, 1e-6, 1.0, 1e3, 1e6):
        for v in _vectors(10, 6, scale=scale):
            assert math.sqrt(np.vdot(v, v)) == pytest.approx(
                float(np.linalg.norm(v)), rel=REORDER_RTOL
            )


# ---------------------------------------------------------------------------
# PR #8095 / #8094 - 3-vectors: np.linalg.norm(v3) -> math.hypot(a, b, c)
# ---------------------------------------------------------------------------


def test_hypot3_matches_norm() -> None:
    for v in _vectors(200, 3):
        assert math.hypot(v[0], v[1], v[2]) == pytest.approx(
            float(np.linalg.norm(v)), rel=REORDER_RTOL
        )


def test_hypot3_is_strictly_more_robust_than_vdot_form() -> None:
    """``math.hypot`` is overflow-safe by construction - this swap is an improvement."""
    huge = (1e200, 1e200, 1e200)
    assert math.isfinite(math.hypot(*huge))
    tiny = (1e-200, 1e-200, 1e-200)
    assert math.hypot(*tiny) > 0.0  # does not underflow to zero


# ---------------------------------------------------------------------------
# PR #7968 - math.hypot(*v) -> math.hypot(v[0], v[1], v[2])
# ---------------------------------------------------------------------------


def test_hypot_star_args_vs_explicit_is_EXACT() -> None:
    for v in _vectors(200, 3):
        assert math.hypot(*v) == math.hypot(v[0], v[1], v[2])


# ---------------------------------------------------------------------------
# PR #8107 / #8106 - rank counting: np.sum(mask) -> mask.sum()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 5, 50, 500])
def test_mask_sum_vs_np_sum_is_EXACT(size: int) -> None:
    for _ in range(25):
        sigma = np.abs(RNG.standard_normal(size))
        tol = float(RNG.uniform(0.0, 1.5))
        mask = sigma > tol
        assert int(mask.sum()) == int(np.sum(mask))


def test_mask_sum_exact_at_boundary() -> None:
    """All-true, all-false, and exact-equality boundaries must agree."""
    sigma = np.array([0.0, 1.0, 1.0, 2.0])
    for tol in (-1.0, 0.0, 1.0, 2.0, 3.0):
        mask = sigma > tol
        assert int(mask.sum()) == int(np.sum(mask))


# ---------------------------------------------------------------------------
# PR #7967 - effective sample size: 1/np.sum(w**2) -> 1/np.vdot(w, w)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [2, 8, 64, 1024])
def test_vdot_vs_sum_of_squares(size: int) -> None:
    for _ in range(25):
        w = RNG.dirichlet(np.ones(size))  # normalized weights, the real input shape
        assert float(np.vdot(w, w)) == pytest.approx(
            float(np.sum(w**2)), rel=REORDER_RTOL
        )


def test_effective_sample_size_reciprocal_agrees() -> None:
    """The caller takes a reciprocal, which amplifies relative error - check it directly."""
    for size in (4, 32, 256):
        w = RNG.dirichlet(np.ones(size))
        assert 1.0 / np.vdot(w, w) == pytest.approx(
            1.0 / np.sum(w**2), rel=REORDER_RTOL
        )


# ---------------------------------------------------------------------------
# PR #7966 / #7889 - RMSE: np.sqrt(np.mean(e**2, axis=0))
#                        -> np.sqrt(np.einsum("ij,ij->j", e, e) / e.shape[0])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("rows", "cols"), [(1, 1), (3, 2), (100, 3), (5000, 6)])
def test_einsum_rmse_matches_mean_rmse(rows: int, cols: int) -> None:
    for _ in range(10):
        err = RNG.standard_normal((rows, cols))
        original = np.sqrt(np.mean(err**2, axis=0))
        optimized = np.sqrt(np.einsum("ij,ij->j", err, err) / err.shape[0])
        np.testing.assert_allclose(optimized, original, rtol=REORDER_RTOL, atol=0.0)


def test_einsum_rmse_scalar_form_matches() -> None:
    """The scalar variant used for ``vector_rmse``."""
    for size in (1, 10, 10_000):
        e = RNG.standard_normal(size)
        original = float(np.sqrt(np.mean(e**2)))
        optimized = float(np.sqrt(np.vdot(e, e) / e.size))
        assert optimized == pytest.approx(original, rel=REORDER_RTOL)


def test_einsum_rmse_zero_error_is_EXACT() -> None:
    """A perfect fit must produce exactly 0.0, not a denormal."""
    err = np.zeros((50, 4))
    optimized = np.sqrt(np.einsum("ij,ij->j", err, err) / err.shape[0])
    assert np.array_equal(optimized, np.zeros(4))


def test_einsum_rmse_survives_wide_dynamic_range() -> None:
    """Mixed-magnitude residuals are the realistic case for club/torque fitting."""
    err = np.column_stack(
        [
            RNG.standard_normal(500) * 1e-8,
            RNG.standard_normal(500) * 1.0,
            RNG.standard_normal(500) * 1e6,
        ]
    )
    original = np.sqrt(np.mean(err**2, axis=0))
    optimized = np.sqrt(np.einsum("ij,ij->j", err, err) / err.shape[0])
    np.testing.assert_allclose(optimized, original, rtol=REORDER_RTOL, atol=0.0)
