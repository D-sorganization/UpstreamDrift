"""Tests for GripGeometry source of truth and presets (#13891)."""

from __future__ import annotations

import math
import pytest

from src.shared.python.grip_contact.grip_geometry import (
    GripGeometry,
    get_grip_preset,
    validate_usga_rules,
    check_standoff_compatibility,
    GRIP_GEOMETRY_PRESETS,
    USGA_MAX_GRIP_DIAMETER_M,
    STATION_UPPER_M,
    STATION_LOWER_M,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Presets & Provenance Tests
# --------------------------------------------------------------------------- #


def test_legacy_spec_standoff_preset() -> None:
    """Default preset matches legacy 25.4 mm uniform diameter (standoff 0.0127 m)."""
    geom = get_grip_preset("legacy_spec_standoff")
    assert geom.preset == "legacy_spec_standoff"
    assert math.isclose(geom.d_upper_m, 0.0254, rel_tol=1e-9)
    assert math.isclose(geom.d_lower_m, 0.0254, rel_tol=1e-9)
    assert geom.taper == "zero"
    # Radius is constant 12.7 mm across stations
    assert math.isclose(geom.radius_at(STATION_UPPER_M), 0.0127, rel_tol=1e-9)
    assert math.isclose(geom.radius_at(STATION_LOWER_M), 0.0127, rel_tol=1e-9)
    assert "rhandstandoff" in geom.source.lower()


def test_mens_standard_preset() -> None:
    """Men's standard is 0.900 in (22.86 mm) at 2 in and 0.780 in (19.81 mm) at 6 in."""
    geom = get_grip_preset("mens_standard")
    assert math.isclose(geom.d_upper_m, 0.02286, rel_tol=1e-4)
    assert math.isclose(geom.d_lower_m, 0.01981, rel_tol=1e-4)
    assert geom.taper == "standard"
    assert "hireko" in geom.source.lower() or "mitchell" in geom.source.lower()


def test_all_presets_carry_source() -> None:
    """Every preset carries a vendor citation or standoff provenance."""
    for name, geom in GRIP_GEOMETRY_PRESETS.items():
        assert geom.preset == name
        assert len(geom.source) > 0
        assert len(geom.vendor) > 0
        assert 0.0 < geom.d_lower_m <= USGA_MAX_GRIP_DIAMETER_M
        assert 0.0 < geom.d_upper_m <= USGA_MAX_GRIP_DIAMETER_M


# --------------------------------------------------------------------------- #
# Analytic Geometry & Buildup Tests
# --------------------------------------------------------------------------- #


def test_zero_taper_is_constant_radius() -> None:
    """Zero taper yields identical radius at all positions."""
    geom = GripGeometry(
        d_upper_m=0.024,
        d_lower_m=0.024,
        taper="zero",
        preset="custom",
        source="test",
        vendor="test",
    )
    r1 = geom.radius_at(0.0)
    r2 = geom.radius_at(0.10)
    r3 = geom.radius_at(0.25)
    assert math.isclose(r1, 0.012, rel_tol=1e-9)
    assert math.isclose(r2, 0.012, rel_tol=1e-9)
    assert math.isclose(r3, 0.012, rel_tol=1e-9)


def test_buildup_layers_addition() -> None:
    """Buildup layers add exact declared increment per vendor."""
    # Golf Pride: 1 wrap = +1/64 in = 0.000396875 m
    base = get_grip_preset("mens_standard")
    gp_1 = base.with_buildup(layers=1, source="golf_pride")
    delta_d_gp = gp_1.effective_d_upper_m - base.d_upper_m
    assert math.isclose(delta_d_gp, 0.0254 / 64.0, rel_tol=1e-6)

    # Hireko: 0.007 in per layer = 0.0001778 m
    h_2 = base.with_buildup(layers=2, source="hireko")
    delta_d_h = h_2.effective_d_upper_m - base.d_upper_m
    assert math.isclose(delta_d_h, 2 * 0.007 * 0.0254, rel_tol=1e-6)


def test_radius_interpolation() -> None:
    """radius_at interpolates linearly between upper and lower stations."""
    geom = get_grip_preset("mens_standard")
    r_up = geom.radius_at(STATION_UPPER_M)
    r_low = geom.radius_at(STATION_LOWER_M)
    mid = 0.5 * (STATION_UPPER_M + STATION_LOWER_M)
    r_mid = geom.radius_at(mid)
    assert math.isclose(r_mid, 0.5 * (r_up + r_low), rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# USGA Rules Envelope & Validation Tests
# --------------------------------------------------------------------------- #


def test_usga_rules_validation() -> None:
    """USGA envelope allows <= 44.45 mm and monotonic tapers; rejects bulges and oversize."""
    valid_std = get_grip_preset("mens_standard")
    validate_usga_rules(valid_std)

    # Oversize > 44.45 mm (1.75 in)
    oversize = GripGeometry(
        d_upper_m=0.045,
        d_lower_m=0.040,
        preset="custom",
        source="test",
        vendor="test",
    )
    with pytest.raises(ValueError, match="USGA"):
        validate_usga_rules(oversize)

    # Bulge in middle (lower > upper for standard taper)
    bulge = GripGeometry(
        d_upper_m=0.020,
        d_lower_m=0.025,
        taper="standard",
        preset="custom",
        source="test",
        vendor="test",
    )
    with pytest.raises(ValueError, match="taper"):
        validate_usga_rules(bulge)

    # Reverse taper is valid when explicitly declared
    reverse = GripGeometry(
        d_upper_m=0.020,
        d_lower_m=0.025,
        taper="reverse",
        preset="custom",
        source="test",
        vendor="test",
    )
    validate_usga_rules(reverse)


def test_standoff_compatibility() -> None:
    """Standoff check asserts agreement between grip radius and hand standoff."""
    legacy = get_grip_preset("legacy_spec_standoff")
    assert check_standoff_compatibility(
        legacy, standoff_m=0.0127, station_s=STATION_UPPER_M
    )

    # Mismatch raises error when strict=True
    with pytest.raises(ValueError, match="standoff mismatch"):
        check_standoff_compatibility(
            legacy, standoff_m=0.010, station_s=STATION_UPPER_M, strict=True
        )
