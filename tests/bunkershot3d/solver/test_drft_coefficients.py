"""Tests for 3D-RFT polynomial coefficients (issue #8611).

Validates the coefficient table from Agarwal, Goldman and Kamrin, PNAS 120
(2023), doi:10.1073/pnas.2214017120.
"""

import numpy as np

from bunkershot3d.solver.coefficients import (
    DRFT_COEFFICIENTS,
    compute_alpha_components,
    compute_f_values,
    compute_term_basis,
)


class TestTermBasis:
    """Verify the polynomial term basis x1, x2, x3."""

    def test_term_basis_shape(self) -> None:
        """Basis returns three values for any valid angles."""
        x1, x2, x3 = compute_term_basis(beta=0.0, gamma=0.0, psi=0.0)
        assert np.isfinite(x1)
        assert np.isfinite(x2)
        assert np.isfinite(x3)

    def test_vertical_intrusion_basis(self) -> None:
        """Verify basis for vertical plate intrusion (gamma=pi/2, beta=0)."""
        x1, x2, x3 = compute_term_basis(beta=0.0, gamma=np.pi / 2, psi=0.0)
        assert np.isclose(x1, 1.0)  # sin(gamma) = 1
        assert np.isclose(x2, 1.0)  # cos(beta) = 1
        # x3 = cos(psi)*cos(gamma)*sin(beta) + sin(gamma)*cos(beta)
        #    = 1*0*0 + 1*1 = 1.0
        assert np.isclose(x3, 1.0)

    def test_horizontal_intrusion_basis(self) -> None:
        """Verify basis for horizontal plate (gamma=0, beta=0)."""
        x1, x2, x3 = compute_term_basis(beta=0.0, gamma=0.0, psi=0.0)
        assert np.isclose(x1, 0.0)  # sin(0) = 0
        assert np.isclose(x2, 1.0)  # cos(0) = 1
        assert np.isclose(x3, 0.0)  # term simplifies to 0


class TestCoefficientTable:
    """Validate the 20-term coefficient table from the paper."""

    def test_coefficient_table_shape(self) -> None:
        """Table has 20 terms, 3 coefficients each."""
        assert DRFT_COEFFICIENTS.shape == (20, 3)

    def test_coefficient_table_dtype(self) -> None:
        """Coefficients are float64 for precision."""
        assert DRFT_COEFFICIENTS.dtype == np.float64

    def test_first_row_constant_term(self) -> None:
        """First row is the constant term from the paper."""
        expected = np.array([0.00212, -0.06796, -0.02634])
        np.testing.assert_allclose(DRFT_COEFFICIENTS[0], expected, rtol=1e-4)

    def test_last_row_xyz_term(self) -> None:
        """Last row is the x1*x2*x3 term."""
        expected = np.array([0.15120, -0.33207, -0.27519])
        np.testing.assert_allclose(DRFT_COEFFICIENTS[19], expected, rtol=1e-4)


class TestFValueComputation:
    """Test the f1, f2, f3 polynomial evaluations."""

    def test_f_values_at_origin(self) -> None:
        """At beta=gamma=psi=0, f values come from a known subset of terms."""
        f1, f2, f3 = compute_f_values(beta=0.0, gamma=0.0, psi=0.0)
        # Only terms with x2^n survive (x1=0, x3=0)
        assert np.isfinite(f1)
        assert np.isfinite(f2)
        assert np.isfinite(f3)

    def test_f_values_finite_everywhere(self) -> None:
        """F values are finite for all valid angle combinations."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            beta = rng.uniform(-np.pi / 2, np.pi / 2)
            gamma = rng.uniform(-np.pi / 2, np.pi / 2)
            psi = rng.uniform(-np.pi / 2, np.pi / 2)
            f1, f2, f3 = compute_f_values(beta, gamma, psi)
            assert np.isfinite(f1), f"f1 not finite at {beta=}, {gamma=}, {psi=}"
            assert np.isfinite(f2), f"f2 not finite at {beta=}, {gamma=}, {psi=}"
            assert np.isfinite(f3), f"f3 not finite at {beta=}, {gamma=}, {psi=}"


class TestAlphaComponents:
    """Test the stress ratio components alpha_r, alpha_theta, alpha_z."""

    def test_alpha_components_shape(self) -> None:
        """Alpha returns three components."""
        alpha_r, alpha_theta, alpha_z = compute_alpha_components(
            beta=0.0, gamma=np.pi / 4, psi=0.0
        )
        assert np.isfinite(alpha_r)
        assert np.isfinite(alpha_theta)
        assert np.isfinite(alpha_z)

    def test_vertical_intrusion_alpha_z_dominates(self) -> None:
        """For vertical intrusion (gamma=pi/2), alpha_z should dominate."""
        alpha_r, alpha_theta, alpha_z = compute_alpha_components(
            beta=0.0, gamma=np.pi / 2, psi=0.0
        )
        # alpha_z = -f1*cos(beta) - f2*sin(gamma) - f3 should be significant
        assert abs(alpha_z) > 0.1, "alpha_z too small for vertical intrusion"

    def test_psi_zero_gives_zero_theta(self) -> None:
        """When psi=0, alpha_theta should be zero (no twist)."""
        alpha_r, alpha_theta, alpha_z = compute_alpha_components(
            beta=0.3, gamma=0.5, psi=0.0
        )
        assert np.isclose(alpha_theta, 0.0, atol=1e-10)
