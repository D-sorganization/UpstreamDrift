"""Tests for RFT material scaling (issue #8611).

Validates the material calibration formula from research-digest-addendum.md:
    xi_n = rho_c * g * f_hat(mu_int)
    f_hat = 894*mu^3 - 386*mu^2 + 89*mu
"""

import math

import numpy as np
import pytest

from bunkershot3d.solver.material import (
    compute_f_hat,
    compute_xi_n,
    mu_from_friction_angle_deg,
)


class TestFrictionCoefficient:
    """Test conversion from friction angle to mu."""

    def test_thirty_degrees(self) -> None:
        """tan(30 deg) ~ 0.577."""
        mu = mu_from_friction_angle_deg(30.0)
        assert np.isclose(mu, math.tan(math.radians(30)), rtol=1e-6)

    def test_thirty_three_degrees(self) -> None:
        """tan(33 deg) ~ 0.649, typical for quartz."""
        mu = mu_from_friction_angle_deg(33.0)
        assert np.isclose(mu, math.tan(math.radians(33)), rtol=1e-6)

    def test_forty_degrees(self) -> None:
        """tan(40 deg) ~ 0.839."""
        mu = mu_from_friction_angle_deg(40.0)
        assert np.isclose(mu, 0.839, rtol=0.01)


class TestFHatCubic:
    """Test the f_hat cubic polynomial."""

    def test_f_hat_at_mu_0_6(self) -> None:
        """f_hat(0.6) should give a known value from the table."""
        f_hat = compute_f_hat(0.6)
        # From table: xi_n = 1.53e6 at rho_c=1450, g=9.81
        # => f_hat = 1.53e6 / (1450 * 9.81) ~ 107.6
        expected = 894 * 0.6**3 - 386 * 0.6**2 + 89 * 0.6
        assert np.isclose(f_hat, expected, rtol=1e-6)
        assert np.isclose(f_hat, 107.496, rtol=0.01)

    def test_f_hat_at_mu_0_7(self) -> None:
        """f_hat(0.7) from the research table."""
        f_hat = compute_f_hat(0.7)
        expected = 894 * 0.7**3 - 386 * 0.7**2 + 89 * 0.7
        assert np.isclose(f_hat, expected, rtol=1e-6)

    def test_f_hat_monotonic_in_bunker_range(self) -> None:
        """f_hat should increase with mu in the relevant range [0.5, 0.9]."""
        mus = np.linspace(0.5, 0.9, 50)
        f_hats = [compute_f_hat(m) for m in mus]
        assert all(f_hats[i] < f_hats[i + 1] for i in range(len(f_hats) - 1))


class TestXiN:
    """Test the complete xi_n scaling factor."""

    def test_xi_n_at_reference_conditions(self) -> None:
        """xi_n at rho_c=1550, mu=0.7 (35 deg) should be ~2.73e6."""
        xi_n = compute_xi_n(bulk_density_kg_m3=1550.0, friction_angle_deg=35.0)
        # From research-digest-addendum table
        assert np.isclose(xi_n, 2.73e6, rtol=0.05)

    def test_xi_n_increases_with_density(self) -> None:
        """xi_n should increase with bulk density."""
        xi_low = compute_xi_n(bulk_density_kg_m3=1450.0, friction_angle_deg=33.0)
        xi_high = compute_xi_n(bulk_density_kg_m3=1700.0, friction_angle_deg=33.0)
        assert xi_high > xi_low

    def test_xi_n_increases_with_friction(self) -> None:
        """xi_n should increase with friction angle."""
        xi_low = compute_xi_n(bulk_density_kg_m3=1550.0, friction_angle_deg=31.0)
        xi_high = compute_xi_n(bulk_density_kg_m3=1550.0, friction_angle_deg=40.0)
        assert xi_high > xi_low

    def test_xi_n_is_positive(self) -> None:
        """xi_n must always be positive."""
        for phi in [25.0, 30.0, 33.0, 35.0, 40.0, 45.0]:
            for rho in [1400.0, 1550.0, 1700.0]:
                xi_n = compute_xi_n(bulk_density_kg_m3=rho, friction_angle_deg=phi)
                assert xi_n > 0, f"xi_n negative at phi={phi}, rho={rho}"


class TestKnownTableValues:
    """Cross-check against the research digest table."""

    @pytest.mark.parametrize(
        "rho_c,phi_deg,expected_xi_n",
        [
            (1450, 31, 1.53e6),
            (1550, 35, 2.73e6),
            (1700, 35, 3.00e6),
            (1550, 40, 5.05e6),
        ],
    )
    def test_table_values(
        self, rho_c: float, phi_deg: float, expected_xi_n: float
    ) -> None:
        """Values should match the research digest table within 10%."""
        xi_n = compute_xi_n(bulk_density_kg_m3=rho_c, friction_angle_deg=phi_deg)
        assert np.isclose(xi_n, expected_xi_n, rtol=0.10)
