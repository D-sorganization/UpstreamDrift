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


@pytest.mark.integration
@pytest.mark.slow
class TestEnergyConservation:
    """Test energy conservation in passive systems per Guideline O3.

    For conservative systems (no damping, no external forces), total mechanical
    energy E = KE + PE should remain constant within numerical integration error.

    Guideline O3 requires <1% energy drift for conservative systems.
    """

    @pytest.mark.skipif(not _check_mujoco_available(), reason="MuJoCo not installed")
    def test_pendulum_energy_conservation_mujoco(self) -> None:
        """Test passive pendulum conserves energy (MuJoCo).

        Uses inline XML model (Assessment B-005 pattern).
        Initial condition: θ = 0.5 rad, θ̇ = 0
        Duration: 5 seconds
        Tolerance: <1% energy drift (Guideline O3)
        """
        import mujoco

        # Load inline model
        model = mujoco.MjModel.from_xml_string(SIMPLE_PENDULUM_XML)
        data = mujoco.MjData(model)

        # Set initial conditions: small angle release
        data.qpos[0] = 0.5  # 0.5 rad from vertical
        data.qvel[0] = 0.0  # Starting from rest
        mujoco.mj_forward(model, data)

        # Record initial energy
        KE0, PE0, E0 = _compute_pendulum_energy(model, data)
        logger.info(f"Initial energy: KE={KE0:.6f}, PE={PE0:.6f}, Total={E0:.6f}")

        # Simulate with zero control
        max_drift_pct = 0.0
        n_steps = int(5.0 / model.opt.timestep)  # 5 seconds

        for step in range(n_steps):
            data.ctrl[:] = 0.0  # Zero torque
            mujoco.mj_step(model, data)

            # Check energy periodically (every 100 steps)
            if step % 100 == 0:
                KE, PE, E = _compute_pendulum_energy(model, data)
                if E0 > 1e-10:  # Avoid division by zero
                    drift_pct = 100 * abs(E - E0) / E0
                    max_drift_pct = max(max_drift_pct, drift_pct)

        # Final check
        KE_final, PE_final, E_final = _compute_pendulum_energy(model, data)
        final_drift_pct = 100 * abs(E_final - E0) / E0 if E0 > 1e-10 else 0.0

        logger.info(
            f"Final energy: KE={KE_final:.6f}, PE={PE_final:.6f}, Total={E_final:.6f}"
        )
        logger.info(f"Energy drift: {final_drift_pct:.4f}% (max: {max_drift_pct:.4f}%)")

        assert max_drift_pct < 1.0, (
            f"Energy drift {max_drift_pct:.2f}% exceeds 1% tolerance (Guideline O3)"
        )

    @pytest.mark.skipif(not _check_mujoco_available(), reason="MuJoCo not installed")
    def test_pendulum_energy_at_extremes(self) -> None:
        """Test energy conservation at motion extremes.

        At highest point: KE ≈ 0, PE = max
        At lowest point: KE = max, PE ≈ 0
        Total should be constant.
        """
        import mujoco

        model = mujoco.MjModel.from_xml_string(SIMPLE_PENDULUM_XML)
        data = mujoco.MjData(model)

        # Start at 0.8 rad to have significant energy
        data.qpos[0] = 0.8
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)

        _, _, E0 = _compute_pendulum_energy(model, data)

        # Simulate for one full period (~2 seconds for 1m pendulum)
        period = 2 * np.pi * np.sqrt(1.0 / GRAVITY_M_S2)  # ~2.0 s
        n_steps = int(period / model.opt.timestep)

        energies = []
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            _, _, E = _compute_pendulum_energy(model, data)
            energies.append(E)

        # All energies should be within 1% of initial
        energies_arr = np.array(energies)
        max_deviation = np.max(np.abs(energies_arr - E0)) / E0 * 100

        assert max_deviation < 1.0, f"Energy variation {max_deviation:.2f}% exceeds 1%"
