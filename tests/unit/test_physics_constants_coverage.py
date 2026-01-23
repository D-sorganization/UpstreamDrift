"""Unit tests for shared physics constants.
Ensures valid values, units, and citation metadata.
"""

import math

from src.shared.python import physics_constants as pc


def test_mathematical_constants():
    """Verify basic mathematical constants."""
    assert pc.PI == math.pi
    assert pc.E == math.e
    assert pc.PI_HALF == math.pi / 2
    assert pc.PI_QUARTER == math.pi / 4


def test_physical_constant_metadata():
    """Verify PhysicalConstant class carries metadata."""
    gravity = pc.GRAVITY_M_S2

    # Check value
    assert math.isclose(gravity, 9.80665, rel_tol=1e-9)

    # Check metadata attributes
    assert hasattr(gravity, "unit")
    assert hasattr(gravity, "source")
    assert hasattr(gravity, "description")

    # Verify exact metadata
    assert gravity.unit == "m/s^2"
    assert "NIST" in gravity.source or "ISO" in gravity.source
    # float(gravity) gives the value, str() might give repr depending on implementation details
    assert math.isclose(float(gravity), 9.80665)
    assert repr(gravity).startswith("PhysicalConstant")


def test_golf_specific_constants():
    """Verify golf rules constants."""
    # USGA Rule 5-1: Mass <= 1.620 oz (0.04593 kg)
    assert pc.GOLF_BALL_MASS_KG <= 0.046

    # USGA Rule 5-2: Diameter >= 1.680 in (0.04267 m)
    assert pc.GOLF_BALL_DIAMETER_M >= 0.0426


def test_unit_conversions():
    """Verify conversion factors are consistent."""
    # Length
    # Relax tolerance because M_TO_FT (3.28084) is an approximation
    assert math.isclose(1.0 * pc.M_TO_FT * pc.FT_TO_M, 1.0, rel_tol=1e-5)

    # Yard to Meter check
    # YARD_TO_M is 0.9144 (exact)
    # M_TO_YARD is 1.09361 (approx)
    assert math.isclose(1.0 * pc.M_TO_YARD * pc.YARD_TO_M, 1.0, rel_tol=1e-5)

    # Angles
    assert math.isclose(180 * pc.DEG_TO_RAD, math.pi)
    assert math.isclose(math.pi * pc.RAD_TO_DEG, 180)

    # Speed
    # 10 m/s = 36 km/h
    assert math.isclose(10 * pc.MPS_TO_KPH, 36.0)


def test_material_densities():
    """Sanity check material densities."""
    # Steel should be denser than aluminum
    assert pc.STEEL_DENSITY_KG_M3 > pc.ALUMINUM_DENSITY_KG_M3

    # Titanium should be denser than aluminum but lighter than steel
    assert pc.TITANIUM_DENSITY_KG_M3 > pc.ALUMINUM_DENSITY_KG_M3
    assert pc.TITANIUM_DENSITY_KG_M3 < pc.STEEL_DENSITY_KG_M3
