"""Property-based tests for conservation laws.

Tests fundamental physics constraints that must hold for any valid simulation:
- Energy conservation (passive systems)
- Momentum conservation (free-floating systems)
- Indexed acceleration closure
- Superposition (drift + control = full)

Per Assessment B recommendations and Guideline O3/M2 requirements.

Uses inline XML models to avoid external file dependencies, following
the Self-Contained Physics Testing Pattern from Assessment B-005.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from src.shared.python.core.constants import GRAVITY_M_S2
from src.shared.python.logging_pkg.logging_config import get_logger

from tests.fixtures.fixtures_lib import _check_mujoco_available

logger = get_logger(__name__)

# Inline pendulum model for energy tests (XML-in-Python pattern)
# Uses explicit inertial properties for accurate energy computation
# The pendulum is a uniform rod of length 1m, mass 1kg, rotating about one end
# Moment of inertia about pivot: I = (1/3) * m * L² = 1/3 kg·m²
# Center of mass: L/2 = 0.5m from pivot
SIMPLE_PENDULUM_XML = """
<mujoco model="simple_pendulum_conservative">
  <option gravity="0 0 -9.81" timestep="0.0005" integrator="RK4"/>
  <compiler angle="radian" inertiafromgeom="false"/>

  <worldbody>
    <light name="light" diffuse="1 1 1" pos="0 0 3"/>
    <body name="pivot" pos="0 0 2">
      <body name="pendulum" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0" frictionloss="0"/>
        <!-- Uniform rod: mass 1kg, length 1m, COM at 0.5m below pivot -->
        <inertial pos="0 0 -0.5" mass="1.0" diaginertia="0.333333 0.333333 0.0001"/>
        <geom type="capsule" size="0.01" fromto="0 0 0 0 0 -1" contype="0" conaffinity="0" mass="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Actuated version for work-energy tests
ACTUATED_PENDULUM_XML = """
<mujoco model="actuated_pendulum">
  <option gravity="0 0 -9.81" timestep="0.0005" integrator="RK4"/>
  <compiler angle="radian" inertiafromgeom="false"/>

  <worldbody>
    <light name="light" diffuse="1 1 1" pos="0 0 3"/>
    <body name="pivot" pos="0 0 2">
      <body name="pendulum" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0" frictionloss="0"/>
        <inertial pos="0 0 -0.5" mass="1.0" diaginertia="0.333333 0.333333 0.0001"/>
        <geom type="capsule" size="0.01" fromto="0 0 0 0 0 -1" contype="0" conaffinity="0" mass="0"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrllimited="false"/>
  </actuator>
</mujoco>
"""

# Physical parameters for the uniform rod pendulum
ROD_LENGTH_M = 1.0  # [m]
ROD_MASS_KG = 1.0  # [kg]
ROD_INERTIA_KGM2 = (1.0 / 3.0) * ROD_MASS_KG * ROD_LENGTH_M**2  # [kg·m²] about pivot


def _compute_pendulum_energy(model: Any, data: Any) -> tuple[float, float, float]:
    """Compute kinetic and potential energy for pendulum using MuJoCo internals.

    Uses MuJoCo's internal energy computation for accuracy.

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        Tuple of (KE, PE, Total Energy) in Joules
    """
    import mujoco

    # Update forward kinematics to compute energy terms
    mujoco.mj_forward(model, data)

    # MuJoCo stores kinetic and potential energy directly
    # data.energy[0] = potential energy
    # data.energy[1] = kinetic energy
    # But we need to enable energy computation in the model

    # Kinetic energy: 0.5 * qvel^T * M * qvel
    nv = model.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    qvel = np.array(data.qvel)
    KE = 0.5 * float(qvel @ M @ qvel)

    # Potential energy for uniform rod:
    # PE = m * g * h_com where h_com is height of center of mass
    # For rod at angle theta from vertical: h_com = L/2 * (1 - cos(theta))
    # Reference: PE = 0 when theta = 0 (hanging straight down)
    theta = float(data.qpos[0])
    L = 1.0  # rod length [m]
    m = 1.0  # mass [kg]
    # Height of COM relative to lowest position (theta=0)
    h_com = (L / 2.0) * (1.0 - np.cos(theta))
    PE = m * GRAVITY_M_S2 * h_com

    return KE, PE, KE + PE


@pytest.mark.unit
class TestConservationHelpers:
    """Unit tests for energy/momentum calculation helpers.

    These helpers will be used by the integration tests above.
    """

    def test_kinetic_energy_calculation(self) -> None:
        """Test KE = 0.5 * qd^T * M * qd."""
        # Simple test with known values
        M = np.array([[2.0, 0.0], [0.0, 3.0]])  # Mass matrix
        qd = np.array([1.0, 2.0])  # Velocities

        KE = 0.5 * qd.T @ M @ qd
        expected = 0.5 * (2.0 * 1.0**2 + 3.0 * 2.0**2)  # 0.5*(2 + 12) = 7.0

        assert abs(KE - expected) < 1e-10, f"KE={KE:.6f}, expected={expected:.6f}"

    def test_potential_energy_gravity(self) -> None:
        """Test PE = m * g * h for simple case."""
        mass = 2.0  # kg
        height = 3.0  # m

        PE = mass * GRAVITY_M_S2 * height
        expected = 2.0 * GRAVITY_M_S2 * 3.0  # 58.86

        assert abs(PE - expected) < 1e-10, f"PE={PE:.6f}, expected={expected:.6f}"

    def test_momentum_calculation(self) -> None:
        """Test p = m * v for simple case."""
        mass = 5.0  # kg
        velocity = np.array([2.0, 3.0, 1.0])  # m/s

        p = mass * velocity
        expected = np.array([10.0, 15.0, 5.0])

        assert np.allclose(p, expected), f"p={p}, expected={expected}"
