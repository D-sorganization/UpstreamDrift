"""Unit tests for sensitivity-derived cross-engine tolerance bands (issue #13794).

TDD test suite verifying:
1. Planted 5% mass matrix error fails under derived band while passing legacy 20% bound.
2. Declared model mismatch dynamically widens tolerance band from receipt.
3. Design by Contract: fail-closed validation on missing/invalid sigma or epsilon.
4. Open-loop replay ensemble/shadowing band computed from finite-time amplification.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.pendulum_simulator.sensitivity import SensitivityReceipt
from src.shared.python.pendulum_simulator.tolerance_bands import (
    CrossEngineToleranceBand,
    compute_cross_engine_tolerance_band,
)

pytestmark = [pytest.mark.unit]


def test_red_plant_five_percent_mass_matrix_error_detected() -> None:
    """A planted 5% mass matrix error on a same-spec pair is caught by derived band.

    Under legacy fixed 20% bound (rel_error < 0.20), a 5% error (0.05) passes silently.
    The sensitivity-derived band reduces to numerical floor (<= 1e-6) for same-spec
    pointwise quantities, correctly catching the 5% error.
    """
    # Nominal mass matrix value for a canonical pendulum
    M_nominal = 1.25  # kg * m^2
    # Planted 5% error
    M_planted = M_nominal * 1.05
    rel_error = abs(M_planted - M_nominal) / M_nominal
    assert math.isclose(rel_error, 0.05, rel_tol=1e-4)

    # Legacy bound check demonstrates flaw:
    legacy_bound = 0.20
    assert rel_error < legacy_bound, "Legacy 20% bound fails to catch 5% error"

    # Derived band for same-spec pointwise comparison:
    band = compute_cross_engine_tolerance_band(
        is_pointwise=True,
        epsilon_model=0.0,
        floor=1e-6,
    )
    assert band.is_pointwise
    assert band.tolerance <= 1e-6
    # The derived band strictly rejects the 5% error:
    assert not (rel_error < band.tolerance), (
        f"Derived band {band.tolerance} should reject 5% error {rel_error}"
    )


def test_declared_model_mismatch_widens_band() -> None:
    """Declared model mismatch dynamically widens tolerance band from receipt."""
    receipt = SensitivityReceipt(
        sigma_max=1.25,
        ftle_per_s=0.5,
        agreement_horizon_s=0.2,
        dominant_direction=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        fd_step=1e-6,
        state_scales=(1.0, 1.0, 1.0, 10.0, 10.0, 10.0),
    )
    # Declared model mismatch of 15% (e.g. rod+sphere geom vs point mass)
    epsilon_model = 0.15
    epsilon_integrator = 0.002
    band = compute_cross_engine_tolerance_band(
        receipt=receipt,
        epsilon_model=epsilon_model,
        epsilon_integrator=epsilon_integrator,
        floor=1e-6,
    )
    # tol = sigma * (eps_model + eps_integrator) + floor
    expected_tol = 1.25 * (0.15 + 0.002) + 1e-6
    assert math.isclose(band.tolerance, expected_tol, rel_tol=1e-5)
    assert band.tolerance > 0.15
    assert band.sigma == 1.25
    assert band.epsilon_model == 0.15
    assert band.epsilon_integrator == 0.002


def test_dbc_fails_closed_on_missing_or_invalid_inputs() -> None:
    """DbC: function fails closed with ValueError when sigma or epsilon are missing/invalid."""
    # Negative epsilon_model
    with pytest.raises(ValueError, match="epsilon_model must be non-negative"):
        compute_cross_engine_tolerance_band(epsilon_model=-0.01)

    # Negative epsilon_integrator
    with pytest.raises(ValueError, match="epsilon_integrator must be non-negative"):
        compute_cross_engine_tolerance_band(epsilon_model=0.0, epsilon_integrator=-1e-4)

    # Missing sigma for dynamic trajectory comparison
    with pytest.raises(ValueError, match="sigma must be provided"):
        compute_cross_engine_tolerance_band(is_pointwise=False, epsilon_model=0.05)

    # Non-finite sigma
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        compute_cross_engine_tolerance_band(sigma=float("nan"), epsilon_model=0.05)

    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        compute_cross_engine_tolerance_band(sigma=-1.0, epsilon_model=0.05)

    # Invalid floor
    with pytest.raises(ValueError, match="floor must be finite and positive"):
        compute_cross_engine_tolerance_band(
            is_pointwise=True, epsilon_model=0.0, floor=0.0
        )


def test_ensemble_shadowing_band_finite_time_amplification() -> None:
    """Ensemble/shadowing band bounds open-loop replay deviation over finite horizon."""
    receipt = SensitivityReceipt(
        sigma_max=2.5,
        ftle_per_s=1.2,
        agreement_horizon_s=0.15,
        dominant_direction=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        fd_step=1e-6,
        state_scales=(1.0, 1.0, 1.0, 10.0, 10.0, 10.0),
    )
    # Open-loop replay: same-spec initial conditions with integrator discretization error
    band = compute_cross_engine_tolerance_band(
        receipt=receipt,
        epsilon_model=0.0,
        epsilon_integrator=1e-4,
        floor=1e-6,
    )
    # Shadowing amplification: sigma * eps_integrator + floor
    assert math.isclose(band.tolerance, 2.5 * 1e-4 + 1e-6, rel_tol=1e-5)
    assert "finite-time amplification" in band.documentation
