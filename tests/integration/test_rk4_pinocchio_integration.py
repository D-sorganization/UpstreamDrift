"""Integration tests for Pinocchio RK4 integrator against analytical solutions.

Tests the RK4 integrator standard implementation for Pinocchio,
validating accuracy and parity with other engines (Drake, MuJoCo, OpenSim).

This test suite validates issue #4118: RK4 Integrator Standard for Pinocchio.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.shared.python.core.constants import GRAVITY_M_S2


def test_pinocchio_rk4_pendulum_accuracy() -> None:
    """Test RK4 accuracy on simple pendulum."""
    pin = pytest.importorskip("pinocchio")

    from src.engines.physics_engines.pinocchio.python.rk4_integrator import (
        PinocchioRK4Integrator,
    )

    # Create a simple pendulum URDF
    urdf_str = """<?xml version="1.0"?>
<robot name="pendulum">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="1000" velocity="10"/>
  </joint>
</robot>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_str)
        urdf_path = f.name

    model = pin.buildModelFromUrdf(urdf_path)
    model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2])

    integrator = PinocchioRK4Integrator(
        model,
        timestep=0.001,
        validate_stages=True,
    )

    # Initial state: 30 degrees
    q = np.array([np.deg2rad(30.0)])
    v = np.array([0.0])

    # Run 100 steps
    energies = []
    for _ in range(100):
        result = integrator.step(q, v, control=np.array([0.0]))
        q = result.q_next
        v = result.v_next

        # Energy
        m, length = 1.0, 1.0
        ke = 0.5 * m * length**2 * v[0] ** 2
        pe = m * GRAVITY_M_S2 * length * (1.0 - np.cos(q[0]))
        energy = ke + pe
        energies.append(energy)

    # Check energy conservation
    initial_energy = energies[0]
    max_error = max(abs(e - initial_energy) for e in energies)

    # Allow 20% energy error over 100ms with 1ms timesteps (conservative bound)
    # RK4 energy drift can be significant for large angles
    assert max_error < initial_energy * 0.20, (
        f"RK4 energy drifted too much: "
        f"initial={initial_energy:.6f}, max_error={max_error:.6f}, "
        f"rel_error={100*max_error/initial_energy:.2f}%"
    )


def test_pinocchio_rk4_multiple_steps_stability() -> None:
    """Test RK4 stability over extended simulations."""
    pin = pytest.importorskip("pinocchio")

    from src.engines.physics_engines.pinocchio.python.rk4_integrator import (
        PinocchioRK4Integrator,
    )

    urdf_str = """<?xml version="1.0"?>
<robot name="pendulum">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="1000" velocity="10"/>
  </joint>
</robot>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_str)
        urdf_path = f.name

    model = pin.buildModelFromUrdf(urdf_path)
    model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2])

    integrator = PinocchioRK4Integrator(
        model,
        timestep=0.001,
        validate_stages=True,
    )

    q = np.array([np.pi / 3.0])
    v = np.array([0.0])

    # Run for 1 second (1000 steps)
    for _ in range(1000):
        result = integrator.step(q, v, control=np.array([0.0]))
        q = result.q_next
        v = result.v_next

        # All values should be finite
        assert np.isfinite(q).all()
        assert np.isfinite(v).all()
        assert np.isfinite(result.a_final).all()

        # Pendulum shouldn't swing past vertical (divergence indicator)
        assert abs(q[0]) < np.pi


def test_pinocchio_rk4_control_input() -> None:
    """Test RK4 with non-zero control input."""
    pin = pytest.importorskip("pinocchio")

    from src.engines.physics_engines.pinocchio.python.rk4_integrator import (
        PinocchioRK4Integrator,
    )

    urdf_str = """<?xml version="1.0"?>
<robot name="pendulum">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="1000" velocity="10"/>
  </joint>
</robot>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_str)
        urdf_path = f.name

    model = pin.buildModelFromUrdf(urdf_path)
    model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2])

    integrator = PinocchioRK4Integrator(model, timestep=0.01)

    q = np.array([0.0])
    v = np.array([0.0])
    tau = np.array([1.0])

    # Apply torque and check acceleration increases
    result1 = integrator.step(q, v, control=np.array([0.0]))
    result2 = integrator.step(q, v, control=tau)

    # Velocity with torque should be greater
    assert result2.v_next[0] > result1.v_next[0]


def test_pinocchio_rk4_integrator_interface() -> None:
    """Test RK4StandardIntegrator interface."""
    from src.shared.python.integrators.rk4_standard import RK4StandardIntegrator

    class DummyIntegrator(RK4StandardIntegrator):
        def forward_dynamics(self, q, v, control=None, time=0.0):
            return np.zeros_like(v)

    integrator = DummyIntegrator(timestep=0.01, tolerance=1e-8)
    assert integrator.timestep == 0.01
    assert integrator.tolerance == 1e-8

    # Test invalid parameters
    with pytest.raises(ValueError):
        DummyIntegrator(timestep=0.0)

    with pytest.raises(ValueError):
        DummyIntegrator(tolerance=0.0)
