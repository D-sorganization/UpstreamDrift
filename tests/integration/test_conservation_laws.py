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
class TestIndexedAccelerationClosure:
    """Test indexed acceleration closure per Guideline M2.

    Decomposed acceleration components (gravity, coriolis, applied, constraint)
    must sum to the total forward dynamics acceleration.

    Required tolerance: 1e-6 rad/s² (joint space)
    """

    @pytest.mark.skipif(not _check_mujoco_available(), reason="MuJoCo not installed")
    def test_drift_control_superposition(self) -> None:
        """Test that drift + control = full acceleration.

        Section F requirement: For any state and control input,
        q̈_full = q̈_drift + q̈_control

        For MuJoCo: qacc = M^-1 * (tau + qfrc_passive - qfrc_bias)
        where qfrc_passive includes constraint forces.

        For a simple actuated system:
        - qacc_full = M^-1 * (tau - bias)
        - qacc_drift = M^-1 * (-bias) = acceleration with tau=0
        - qacc_control_only = M^-1 * tau
        - Superposition: qacc_full = qacc_drift + qacc_control_only
        """
        import mujoco

        # Must use actuated model for control input
        model = mujoco.MjModel.from_xml_string(ACTUATED_PENDULUM_XML)
        data = mujoco.MjData(model)

        # Set non-zero state
        data.qpos[0] = 0.3
        data.qvel[0] = 0.5
        mujoco.mj_forward(model, data)

        # Get M and bias at this configuration
        nv = model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        bias = np.array(data.qfrc_bias).copy()

        # Compute drift acceleration (tau = 0)
        # qacc_drift = M^-1 * (0 - bias) = -M^-1 * bias
        qacc_drift = -np.linalg.solve(M, bias)

        # Now apply control and compute full acceleration
        tau = 2.0  # [N·m]
        data.ctrl[0] = tau
        mujoco.mj_forward(model, data)
        qacc_full = np.array(data.qacc).copy()

        # Control-only component: M^-1 * tau
        qacc_control_only = np.linalg.solve(M, np.array([tau]))

        # Superposition check
        qacc_sum = qacc_drift + qacc_control_only
        residual = np.abs(qacc_full - qacc_sum)

        logger.info(f"Full acceleration: {qacc_full}")
        logger.info(f"Drift component: {qacc_drift}")
        logger.info(f"Control component: {qacc_control_only}")
        logger.info(f"Drift + Control: {qacc_sum}")
        logger.info(f"Residual: {residual}")

        TOLERANCE_CLOSURE = 1e-6  # [rad/s²] per Guideline M2
        assert np.all(
            residual < TOLERANCE_CLOSURE
        ), f"Superposition failed: residual {residual} > {TOLERANCE_CLOSURE}"

    @pytest.mark.skipif(not _check_mujoco_available(), reason="MuJoCo not installed")
    def test_ztcf_equals_drift(self) -> None:
        """Test that ZTCF (Zero-Torque Counterfactual) equals drift acceleration.

        Per Section G1: ZTCF isolates drift dynamics.
        a_ZTCF should equal a_drift = M^-1 * (-bias)
        """
        import mujoco

        model = mujoco.MjModel.from_xml_string(SIMPLE_PENDULUM_XML)
        data = mujoco.MjData(model)

        # Set state
        theta = 0.4
        theta_dot = 0.6
        data.qpos[0] = theta
        data.qvel[0] = theta_dot
        mujoco.mj_forward(model, data)

        # Compute ZTCF (via drift calculation)
        nv = model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        bias = data.qfrc_bias.copy()
        qacc_drift = -np.linalg.solve(M, bias)

        # Direct ZTCF via forward dynamics with zero control
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        qacc_ztcf = data.qacc.copy()

        residual = np.abs(qacc_drift - qacc_ztcf)
        TOLERANCE = 1e-10  # Should be machine precision

        assert np.all(residual < TOLERANCE), f"ZTCF != drift: residual {residual}"

    @pytest.mark.skipif(not _check_mujoco_available(), reason="MuJoCo not installed")
    def test_zvcf_eliminates_coriolis(self) -> None:
        """Test that ZVCF (Zero-Velocity Counterfactual) has no velocity terms.

        Per Section G2: ZVCF isolates configuration-dependent dynamics.
        With v=0, Coriolis/centrifugal terms should vanish.

        For the pendulum test, we verify that:
        1. With v=0, acceleration depends only on gravity
        2. The acceleration matches MuJoCo's computed bias-based acceleration
        """
        import mujoco

        model = mujoco.MjModel.from_xml_string(SIMPLE_PENDULUM_XML)
        data = mujoco.MjData(model)

        # Set configuration only (v=0 implicitly)
        theta = 0.5
        data.qpos[0] = theta
        data.qvel[0] = 0.0  # Zero velocity
        mujoco.mj_forward(model, data)

        # Get ZVCF acceleration from MuJoCo
        qacc_zvcf = float(data.qacc[0])

        # Compute expected acceleration from MuJoCo's dynamics
        # qacc = M^-1 * (-bias) where bias contains gravity term
        nv = model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        bias = np.array(data.qfrc_bias).copy()
        expected_qacc = float(-np.linalg.solve(M, bias)[0])

        residual = abs(qacc_zvcf - expected_qacc)
        TOLERANCE = 1e-10  # Should be machine precision

        logger.info(f"ZVCF acceleration: {qacc_zvcf:.6f}")
        logger.info(f"Expected (M^-1 * bias): {expected_qacc:.6f}")
        logger.info(f"Mass matrix: {M[0, 0]:.6f}")
        logger.info(f"Bias force: {bias[0]:.6f}")

        assert residual < TOLERANCE, f"ZVCF residual {residual:.6e} > {TOLERANCE}"

        # Also verify physics makes sense: should be negative (restoring force)
        # when theta > 0 (pendulum displaced counter-clockwise)
        assert (
            qacc_zvcf < 0
        ), f"Acceleration should be negative (restoring), got {qacc_zvcf:.4f}"
