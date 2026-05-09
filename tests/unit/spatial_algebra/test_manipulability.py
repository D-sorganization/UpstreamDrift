"""Tests for src.shared.python.spatial_algebra.manipulability (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.manipulability import (
    CATASTROPHIC_SINGULARITY_THRESHOLD,
    SINGULARITY_FALLBACK_THRESHOLD,
    SINGULARITY_WARNING_THRESHOLD,
    SingularityError,
    check_jacobian_conditioning,
    compute_manipulability_ellipsoid,
    compute_manipulability_index,
)


def _identity_jacobian(m: int = 6, n: int = 6) -> np.ndarray:
    """Well-conditioned Jacobian: identity (κ = 1)."""
    return np.eye(m, n)


def _ill_conditioned_jacobian() -> np.ndarray:
    """Jacobian with condition number > 1e12 (catastrophic)."""
    J = np.eye(3)
    J[2, 2] = 1e-14  # σ_min → 0
    return J


class TestThresholdConstants:
    def test_warning_lt_fallback(self) -> None:
        assert SINGULARITY_WARNING_THRESHOLD < SINGULARITY_FALLBACK_THRESHOLD

    def test_fallback_lt_catastrophic(self) -> None:
        assert SINGULARITY_FALLBACK_THRESHOLD < CATASTROPHIC_SINGULARITY_THRESHOLD

    def test_warning_is_1e6(self) -> None:
        assert pytest.approx(1e6) == SINGULARITY_WARNING_THRESHOLD

    def test_catastrophic_is_1e12(self) -> None:
        assert pytest.approx(1e12) == CATASTROPHIC_SINGULARITY_THRESHOLD


class TestCheckJacobianConditioning:
    def test_manipulability_returns_float(self) -> None:
        kappa = check_jacobian_conditioning(_identity_jacobian(), "test_body")
        assert isinstance(kappa, float)

    def test_identity_has_condition_one(self) -> None:
        kappa = check_jacobian_conditioning(_identity_jacobian(), "test_body")
        assert kappa == pytest.approx(1.0, rel=1e-6)

    def test_non_array_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            check_jacobian_conditioning([[1, 0], [0, 1]], "body")  # type: ignore[arg-type]

    def test_empty_body_name_raises(self) -> None:
        with pytest.raises(ValueError):
            check_jacobian_conditioning(_identity_jacobian(), "")

    def test_empty_jacobian_returns_inf(self) -> None:
        J = np.zeros((0, 3))
        kappa = check_jacobian_conditioning(J, "body")
        assert np.isinf(kappa)

    def test_catastrophic_raises_singularity_error(self) -> None:
        J = _ill_conditioned_jacobian()
        with pytest.raises(SingularityError):
            check_jacobian_conditioning(J, "bad_body")

    def test_warn_false_does_not_raise_for_severe(self) -> None:
        # Near-fallback (but not catastrophic): should not raise when warn=False
        J = np.diag([1.0, 1.0, 1e-7])  # κ ≈ 1e7, past warning, below catastrophic
        kappa = check_jacobian_conditioning(J, "body", warn=False)
        assert np.isfinite(kappa)
        assert kappa > SINGULARITY_WARNING_THRESHOLD

    def test_condition_number_positive(self) -> None:
        J = np.array([[2.0, 1.0], [1.0, 3.0]])
        kappa = check_jacobian_conditioning(J, "body")
        assert kappa > 0.0


class TestComputeManipulabilityEllipsoid:
    def test_returns_tuple_of_two(self) -> None:
        radii, axes = compute_manipulability_ellipsoid(_identity_jacobian(3, 3))
        assert isinstance(radii, np.ndarray)
        assert isinstance(axes, np.ndarray)

    def test_identity_has_unit_radii(self) -> None:
        radii, _ = compute_manipulability_ellipsoid(_identity_jacobian(3, 3))
        np.testing.assert_allclose(radii, np.ones(3), atol=1e-10)

    def test_non_array_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            compute_manipulability_ellipsoid([[1, 0], [0, 1]])  # type: ignore[arg-type]

    def test_1d_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            compute_manipulability_ellipsoid(np.array([1.0, 2.0, 3.0]))

    def test_radii_are_sorted_descending(self) -> None:
        J = np.diag([3.0, 2.0, 1.0])
        radii, _ = compute_manipulability_ellipsoid(J)
        # SVD singular values are sorted descending by numpy
        assert radii[0] >= radii[1] >= radii[2]

    def test_radii_non_negative(self) -> None:
        J = np.random.default_rng(0).standard_normal((6, 4))
        radii, _ = compute_manipulability_ellipsoid(J)
        assert np.all(radii >= 0.0)


class TestComputeManipulabilityIndex:
    def test_manipulability_returns_float(self) -> None:
        mu = compute_manipulability_index(_identity_jacobian(3, 3))
        assert isinstance(mu, float)

    def test_identity_has_index_one(self) -> None:
        mu = compute_manipulability_index(_identity_jacobian(3, 3))
        assert mu == pytest.approx(1.0, rel=1e-10)

    def test_non_array_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            compute_manipulability_index([[1, 0], [0, 1]])  # type: ignore[arg-type]

    def test_1d_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            compute_manipulability_index(np.array([1.0, 2.0]))

    def test_rank_deficient_gives_zero_index(self) -> None:
        # Rank-deficient square matrix: one singular value is zero
        J = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        mu = compute_manipulability_index(J)
        assert mu == pytest.approx(0.0, abs=1e-12)

    def test_index_non_negative(self) -> None:
        J = np.random.default_rng(1).standard_normal((4, 4))
        mu = compute_manipulability_index(J)
        assert mu >= 0.0

    def test_index_equals_product_of_singular_values(self) -> None:
        J = np.diag([2.0, 3.0, 5.0])
        mu = compute_manipulability_index(J)
        assert mu == pytest.approx(2.0 * 3.0 * 5.0, rel=1e-10)
