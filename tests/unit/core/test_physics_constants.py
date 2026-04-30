"""Tests for src.shared.python.core.physics_constants (Issues #1949, #1744)."""

from __future__ import annotations

import math

import pytest

from src.shared.python.core.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    DRIVER_LENGTH_MAX_M,
    DRIVER_LOFT_TYPICAL_DEG,
    GOLF_BALL_CROSS_SECTIONAL_AREA_M2,
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_DRAG_COEFFICIENT,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
    PI,
    SPATIAL_ANG_DIM,
    SPATIAL_DIM,
    SPATIAL_LIN_DIM,
    PhysicalConstant,
)

# ---------------------------------------------------------------------------
# PhysicalConstant class
# ---------------------------------------------------------------------------


class TestPhysicalConstant:
    def test_is_float_subclass(self) -> None:
        c = PhysicalConstant(9.81, "m/s^2", "Test", "gravity")
        assert isinstance(c, float)

    def test_float_value(self) -> None:
        c = PhysicalConstant(3.14, "dimensionless", "Test")
        assert abs(float(c) - 3.14) < 1e-12

    def test_unit_attribute(self) -> None:
        c = PhysicalConstant(1.0, "kg", "Test")
        assert c.unit == "kg"

    def test_source_attribute(self) -> None:
        c = PhysicalConstant(1.0, "kg", "NIST 2018")
        assert c.source == "NIST 2018"

    def test_description_attribute(self) -> None:
        c = PhysicalConstant(1.0, "kg", "Test", "test constant")
        assert c.description == "test constant"

    def test_description_default_empty(self) -> None:
        c = PhysicalConstant(1.0, "kg", "Test")
        assert c.description == ""

    def test_repr_shows_unit(self) -> None:
        c = PhysicalConstant(9.81, "m/s^2", "Test")
        r = repr(c)
        assert "m/s^2" in r

    def test_arithmetic_works(self) -> None:
        c = PhysicalConstant(2.0, "m", "Test")
        assert c * 3 == pytest.approx(6.0)

    def test_zero_value(self) -> None:
        c = PhysicalConstant(0.0, "dimensionless", "Test")
        assert float(c) == 0.0

    def test_negative_value(self) -> None:
        c = PhysicalConstant(-1.5, "m", "Test")
        assert float(c) < 0


# ---------------------------------------------------------------------------
# Physical constants — values and consistency
# ---------------------------------------------------------------------------


class TestPhysicalConstantValues:
    def test_gravity_standard(self) -> None:
        assert abs(float(GRAVITY_M_S2) - 9.80665) < 1e-5

    def test_air_density(self) -> None:
        assert abs(float(AIR_DENSITY_SEA_LEVEL_KG_M3) - 1.225) < 1e-6

    def test_pi(self) -> None:
        assert abs(PI - math.pi) < 1e-15


class TestSpatialDimensions:
    def test_spatial_dim(self) -> None:
        assert SPATIAL_DIM == 6

    def test_lin_plus_ang(self) -> None:
        assert SPATIAL_LIN_DIM + SPATIAL_ANG_DIM == SPATIAL_DIM


# ---------------------------------------------------------------------------
# Golf ball constants — consistency checks
# ---------------------------------------------------------------------------


class TestGolfBallConstants:
    def test_mass_positive(self) -> None:
        assert float(GOLF_BALL_MASS_KG) > 0

    def test_diameter_positive(self) -> None:
        assert float(GOLF_BALL_DIAMETER_M) > 0

    def test_radius_is_half_diameter(self) -> None:
        assert abs(float(GOLF_BALL_RADIUS_M) - float(GOLF_BALL_DIAMETER_M) / 2) < 1e-12

    def test_cross_section_derived_from_diameter(self) -> None:
        expected = math.pi * (float(GOLF_BALL_DIAMETER_M) / 2) ** 2
        assert abs(float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2) - expected) < 1e-12

    def test_moment_of_inertia_positive(self) -> None:
        assert float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2) > 0

    def test_drag_coefficient_in_range(self) -> None:
        assert 0 < float(GOLF_BALL_DRAG_COEFFICIENT) < 1

    def test_usga_mass_limit(self) -> None:
        # USGA max 1.620 oz = 0.04593 kg
        assert abs(float(GOLF_BALL_MASS_KG) - 0.04593) < 1e-5

    def test_usga_diameter_limit(self) -> None:
        # USGA min 1.680 in = 0.04267 m
        assert abs(float(GOLF_BALL_DIAMETER_M) - 0.04267) < 1e-5


# ---------------------------------------------------------------------------
# Club constants
# ---------------------------------------------------------------------------


class TestClubConstants:
    def test_driver_length_positive(self) -> None:
        assert float(DRIVER_LENGTH_MAX_M) > 0

    def test_driver_loft_reasonable(self) -> None:
        assert 0 < float(DRIVER_LOFT_TYPICAL_DEG) < 20

    def test_driver_length_units(self) -> None:
        assert DRIVER_LENGTH_MAX_M.unit == "m"
