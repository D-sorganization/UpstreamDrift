"""Unit tests for shared physics constants.
Ensures valid values, units, and citation metadata.
"""

import math

import pytest

from src.shared.python.core import physics_constants as pc


@pytest.mark.parametrize(
    "constant,expected",
    [
        (pc.PI, math.pi),
        (pc.E, math.e),
        (pc.PI_HALF, math.pi / 2),
        (pc.PI_QUARTER, math.pi / 4),
    ],
    ids=["PI", "E", "PI_HALF", "PI_QUARTER"],
)
def test_mathematical_constants(constant, expected):
    """Verify basic mathematical constants."""
    assert constant == expected


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


@pytest.mark.parametrize(
    "forward_factor,inverse_factor,description",
    [
        (pc.M_TO_FT, pc.FT_TO_M, "feet_meters"),
        (pc.M_TO_YARD, pc.YARD_TO_M, "yards_meters"),
    ],
    ids=["feet_meters_roundtrip", "yards_meters_roundtrip"],
)
def test_unit_conversion_roundtrips(forward_factor, inverse_factor, description):
    """Verify conversion factor round-trips are consistent."""
    assert math.isclose(1.0 * forward_factor * inverse_factor, 1.0, rel_tol=1e-5)


@pytest.mark.parametrize(
    "input_val,factor,expected",
    [
        (180, pc.DEG_TO_RAD, math.pi),
        (math.pi, pc.RAD_TO_DEG, 180),
        (10, pc.MPS_TO_KPH, 36.0),
    ],
    ids=["deg_to_rad", "rad_to_deg", "mps_to_kph"],
)
def test_unit_conversions(input_val, factor, expected):
    """Verify individual conversion factors."""
    assert math.isclose(input_val * factor, expected)


@pytest.mark.parametrize(
    "heavier,lighter",
    [
        (pc.STEEL_DENSITY_KG_M3, pc.ALUMINUM_DENSITY_KG_M3),
        (pc.TITANIUM_DENSITY_KG_M3, pc.ALUMINUM_DENSITY_KG_M3),
        (pc.STEEL_DENSITY_KG_M3, pc.TITANIUM_DENSITY_KG_M3),
    ],
    ids=["steel_gt_aluminum", "titanium_gt_aluminum", "steel_gt_titanium"],
)
def test_material_densities(heavier, lighter):
    """Sanity check material density ordering."""
    assert heavier > lighter
