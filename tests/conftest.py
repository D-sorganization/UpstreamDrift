"""Shared fixtures and utilities for the Golf Modeling Suite test suite.

This module centralizes common setup logic to improve test orthogonality
and adherence to the DRY principle.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def pendulum_urdf(tmp_path: Path) -> str:
    """Create a standardized simple pendulum URDF for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="pendulum">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""
    urdf_path = tmp_path / "pendulum.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


@pytest.fixture
def clean_pendulum_dynamics():
    """Fixture to provide standardized DoublePendulumDynamics setup for unit tests."""
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DoublePendulumDynamics,
        DoublePendulumParameters,
        LowerSegmentProperties,
        SegmentProperties,
    )

    def _create(m1_kg=1.0, l1_m=1.0):
        upper_segment = SegmentProperties(
            length_m=l1_m,
            mass_kg=m1_kg,
            center_of_mass_ratio=1.0,
            inertia_about_com=0.0,
        )
        # Quasi-massless link 2
        epsilon_kg = 1e-10
        lower_segment = LowerSegmentProperties(
            length_m=1.0,
            shaft_mass_kg=epsilon_kg,
            clubhead_mass_kg=epsilon_kg,
            shaft_com_ratio=0.5,
        )
        params = DoublePendulumParameters(
            upper_segment=upper_segment,
            lower_segment=lower_segment,
            plane_inclination_deg=0.0,
            damping_shoulder=0.0,
            damping_wrist=0.0,
            gravity_enabled=True,
            constrained_to_plane=True,
        )
        return DoublePendulumDynamics(parameters=params)

    return _create
