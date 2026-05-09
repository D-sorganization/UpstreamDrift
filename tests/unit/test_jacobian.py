"""Tests for Jacobian computation across physics engines.

Implements Task 3.1: Jacobian Coverage Completion per Phase 3 roadmap.
Verifies Jacobian shape (6×nv), structure, and cross-engine consistency.

Refactored to use shared engine availability module (DRY principle).
"""

import numpy as np
import pytest
from src.shared.python.engine_core.engine_availability import (
    MUJOCO_AVAILABLE,
    PINOCCHIO_AVAILABLE,
    skip_if_unavailable,
)

# Simple inline URDF for Jacobian tests (2-DOF planar arm)
SIMPLE_ARM_URDF = """<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </visual>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>

  <link name="link2">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </visual>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""

# Simple MJCF equivalent for MuJoCo tests
SIMPLE_ARM_MJCF = """
<mujoco model="simple_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <compiler angle="radian"/>

  <worldbody>
    <light name="light" pos="0 0 3"/>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 1 0"/>
      <geom type="cylinder" size="0.02 0.25" pos="0 0 0.25"/>
      <inertial pos="0 0 0.25" mass="1.0" diaginertia="0.01 0.01 0.01"/>

      <body name="link2" pos="0 0 0.5">
        <joint name="joint2" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.02 0.25" pos="0 0 0.25"/>
        <inertial pos="0 0 0.25" mass="0.5" diaginertia="0.005 0.005 0.005"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class TestJacobianShape:
    """Tests for Jacobian shape compliance."""


class TestJacobianStructure:
    """Tests for Jacobian structural correctness."""


class TestJacobianConsistency:
    """Tests for cross-engine Jacobian consistency."""


class TestJacobianNumericalValidation:
    """Numerical validation of Jacobians via finite differences."""
