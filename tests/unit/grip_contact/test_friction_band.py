"""Tests for cited grip friction band and vector combined-load bound (#13890)."""

from __future__ import annotations

import math
import pytest

from src.shared.python.grip_contact.friction_band import (
    GripFrictionBand,
    GRIP_FRICTION_BANDS,
    combined_squeeze_n,
    governing_branch,
    unilateral_friction_capacity,
    get_friction_band,
)
from src.shared.python.grip_contact.parameters import ContactMaterial

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# GripFrictionBand & Provenance Tests
# --------------------------------------------------------------------------- #


def test_grip_friction_band_presets() -> None:
    """Canonical regimes dry, moist, film are present and cited."""
    for regime in ("dry", "moist", "film"):
        band = get_friction_band(regime)
        assert band.regime == regime
        assert 0.0 < band.mu_low <= band.mu_high
        # Every numeric mu carries a citation (DOI or URL) or the tag 'provisional'
        assert (
            "doi:" in band.source.lower()
            or "http" in band.source.lower()
            or "provisional" in band.source.lower()
        )


def test_grip_friction_band_values() -> None:
    """Cited friction values conform to literature ranges."""
    dry = GRIP_FRICTION_BANDS["dry"]
    assert dry.mu_low == 0.5
    assert dry.mu_high == 0.7

    moist = GRIP_FRICTION_BANDS["moist"]
    assert moist.mu_low == 1.0
    assert moist.mu_high == 1.4

    film = GRIP_FRICTION_BANDS["film"]
    assert film.mu_low == 0.2
    assert film.mu_high == 0.4
    assert "provisional" in film.source.lower()


def test_grip_friction_band_validation() -> None:
    """DbC: invalid mu or unknown regime raises ValueError."""
    with pytest.raises(ValueError, match="mu_low"):
        GripFrictionBand(regime="dry", mu_low=-0.1, mu_high=0.7, source="provisional")

    with pytest.raises(ValueError, match="mu_high"):
        GripFrictionBand(regime="dry", mu_low=0.8, mu_high=0.7, source="provisional")

    with pytest.raises(ValueError, match="finite"):
        GripFrictionBand(
            regime="dry", mu_low=float("nan"), mu_high=0.7, source="provisional"
        )

    with pytest.raises(ValueError, match="regime"):
        GripFrictionBand(regime="ice", mu_low=0.1, mu_high=0.2, source="provisional")

    with pytest.raises(ValueError, match="citation"):
        GripFrictionBand(
            regime="dry", mu_low=0.5, mu_high=0.7, source="uncited estimate"
        )


def test_contact_material_has_source() -> None:
    """ContactMaterial gains a source attribute."""
    mat = ContactMaterial()
    assert hasattr(mat, "source")
    assert isinstance(mat.source, str)
    assert len(mat.source) > 0


# --------------------------------------------------------------------------- #
# Combined Squeeze & Governing Branch Tests
# --------------------------------------------------------------------------- #


def test_combined_squeeze_scaling_and_bounds() -> None:
    """Analytic scaling and comparison with standard max() bound."""
    mu = 0.6
    r = 0.015

    # 1. Squeeze proportional to 1/mu on axial and torque branches
    f_ax = 100.0
    s_ax_1 = combined_squeeze_n(f_ax, 0.0, 0.0, mu, r)
    s_ax_2 = combined_squeeze_n(f_ax, 0.0, 0.0, 2 * mu, r)
    assert math.isclose(s_ax_1, 2.0 * s_ax_2, rel_tol=1e-12)

    tau = 2.0
    s_tau_1 = combined_squeeze_n(0.0, tau, 0.0, mu, r)
    s_tau_2 = combined_squeeze_n(0.0, tau, 0.0, 2 * mu, r)
    assert math.isclose(s_tau_1, 2.0 * s_tau_2, rel_tol=1e-12)

    # 2. Torque branch proportional to 1/r, axial independent of r
    s_tau_r2 = combined_squeeze_n(0.0, tau, 0.0, mu, 2 * r)
    assert math.isclose(s_tau_1, 2.0 * s_tau_r2, rel_tol=1e-12)
    s_ax_r2 = combined_squeeze_n(f_ax, 0.0, 0.0, mu, 2 * r)
    assert math.isclose(s_ax_1, s_ax_r2, rel_tol=1e-12)

    # 3. Vector bound equals max() when either tangential component is 0
    from src.shared.python.grip_contact.pad_layout import required_squeeze_n

    req_ax = required_squeeze_n(f_ax, 0.0, 0.0, mu, r)
    assert math.isclose(s_ax_1, req_ax, rel_tol=1e-12)
    req_tau = required_squeeze_n(0.0, tau, 0.0, mu, r)
    assert math.isclose(s_tau_1, req_tau, rel_tol=1e-12)

    # 4. When F_ax == tau / r, vector bound = sqrt(2) * max()
    tau_eq = f_ax * r
    s_comb = combined_squeeze_n(f_ax, tau_eq, 0.0, mu, r)
    req_comb = required_squeeze_n(f_ax, tau_eq, 0.0, mu, r)
    assert math.isclose(s_comb, math.sqrt(2.0) * req_comb, rel_tol=1e-12)

    # 5. max() <= vector <= L1 sum
    l1_sum = (f_ax + tau_eq / r) / mu
    assert req_comb <= s_comb <= l1_sum


def test_governing_branch_classification() -> None:
    """Governing branch correctly identifies axial, torque, transverse, combined."""
    mu = 0.6
    r = 0.015

    # Pure axial
    assert governing_branch(100.0, 0.0, 0.0, mu, r) == "axial"

    # Pure torque
    assert governing_branch(0.0, 2.0, 0.0, mu, r) == "torque"

    # Pure transverse (or transverse dominates)
    assert governing_branch(10.0, 0.1, 500.0, mu, r) == "transverse"

    # Combined tangential (both contribute, exceeding transverse)
    assert governing_branch(100.0, 1.5, 10.0, mu, r) == "combined"


def test_unilateral_friction_capacity() -> None:
    """Capacity is 0 for normal_force <= 0, and mu * f_n for f_n > 0."""
    mu = 0.6
    assert unilateral_friction_capacity(-10.0, mu) == 0.0
    assert unilateral_friction_capacity(0.0, mu) == 0.0
    assert math.isclose(unilateral_friction_capacity(50.0, mu), 30.0, rel_tol=1e-12)
