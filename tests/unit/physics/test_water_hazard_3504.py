"""Unit tests for hydrodynamic water-hazard entry kinematics (issue #3504)."""

from __future__ import annotations

import math

import pytest
from src.shared.python.physics.water_hazard import (
    WaterEntryResult,
    water_entry_kinematics,
)

pytestmark = pytest.mark.unit


def test_vertical_impact_does_not_bounce_and_submerges() -> None:
    """A 90 deg vertical impact should fully submerge with bounces=0."""
    result = water_entry_kinematics(
        impact_velocity_m_s=25.0,
        impact_angle_deg=90.0,
    )
    assert isinstance(result, WaterEntryResult)
    assert result.bounces == 0
    assert result.submersion_depth_m > 0.0
    assert math.isfinite(result.submersion_depth_m)


def test_shallow_fast_impact_bounces_once() -> None:
    """Shallow (10 deg) fast (30 m/s) impact triggers one skip."""
    result = water_entry_kinematics(
        impact_velocity_m_s=30.0,
        impact_angle_deg=10.0,
    )
    assert result.bounces == 1


def test_higher_velocity_yields_deeper_submersion() -> None:
    """Increasing the impact speed monotonically increases depth."""
    slow = water_entry_kinematics(
        impact_velocity_m_s=20.0,
        impact_angle_deg=90.0,
    )
    fast = water_entry_kinematics(
        impact_velocity_m_s=40.0,
        impact_angle_deg=90.0,
    )
    assert fast.submersion_depth_m > slow.submersion_depth_m


def test_higher_angle_yields_deeper_submersion() -> None:
    """At fixed speed, a steeper angle gives more downward energy."""
    shallow = water_entry_kinematics(
        impact_velocity_m_s=25.0,
        impact_angle_deg=30.0,
    )
    steep = water_entry_kinematics(
        impact_velocity_m_s=25.0,
        impact_angle_deg=80.0,
    )
    assert steep.submersion_depth_m > shallow.submersion_depth_m


def test_energy_dissipated_does_not_exceed_kinetic_energy() -> None:
    """Energy dissipated must be bounded above by 0.5 m v^2."""
    mass = 0.04593
    velocity = 35.0
    ke = 0.5 * mass * velocity * velocity
    # Submerging case
    submerged = water_entry_kinematics(
        impact_velocity_m_s=velocity,
        impact_angle_deg=60.0,
        ball_mass_kg=mass,
    )
    assert submerged.energy_dissipated_j <= ke + 1e-9
    # Bouncing case
    bouncing = water_entry_kinematics(
        impact_velocity_m_s=velocity,
        impact_angle_deg=10.0,
        ball_mass_kg=mass,
    )
    assert bouncing.energy_dissipated_j <= ke + 1e-9
    assert bouncing.energy_dissipated_j >= 0.0


def test_peak_deceleration_matches_formula() -> None:
    """Peak deceleration equals 0.5 * rho * Cd * A * v_n^2 / m."""
    v = 30.0
    angle_deg = 45.0
    rho = 1000.0
    cd = 0.47
    radius = 0.02135
    mass = 0.04593
    area = math.pi * radius * radius
    v_n = v * math.sin(math.radians(angle_deg))
    expected = 0.5 * rho * cd * area * v_n * v_n / mass

    result = water_entry_kinematics(
        impact_velocity_m_s=v,
        impact_angle_deg=angle_deg,
        ball_mass_kg=mass,
        ball_radius_m=radius,
        water_density_kg_m3=rho,
        drag_coefficient=cd,
    )
    assert math.isclose(result.vertical_decel_peak_m_s2, expected, rel_tol=1e-9)


def test_negative_velocity_raises_value_error() -> None:
    with pytest.raises(ValueError):
        water_entry_kinematics(
            impact_velocity_m_s=-1.0,
            impact_angle_deg=45.0,
        )


def test_angle_above_90_raises_value_error() -> None:
    with pytest.raises(ValueError):
        water_entry_kinematics(
            impact_velocity_m_s=25.0,
            impact_angle_deg=120.0,
        )


def test_nan_velocity_raises_value_error() -> None:
    with pytest.raises(ValueError):
        water_entry_kinematics(
            impact_velocity_m_s=float("nan"),
            impact_angle_deg=45.0,
        )


def test_non_positive_mass_raises_value_error() -> None:
    with pytest.raises(ValueError):
        water_entry_kinematics(
            impact_velocity_m_s=25.0,
            impact_angle_deg=45.0,
            ball_mass_kg=0.0,
        )


def test_non_numeric_angle_raises_type_error() -> None:
    with pytest.raises(TypeError):
        water_entry_kinematics(
            impact_velocity_m_s=25.0,
            impact_angle_deg="forty-five",  # type: ignore[arg-type]
        )


def test_zero_angle_yields_no_submersion_and_one_bounce_when_fast() -> None:
    """Tangential-only impact: no normal component, but skipping rule fires."""
    result = water_entry_kinematics(
        impact_velocity_m_s=25.0,
        impact_angle_deg=0.0,
    )
    assert result.submersion_depth_m == 0.0
    assert result.vertical_decel_peak_m_s2 == 0.0
    assert result.bounces == 1
