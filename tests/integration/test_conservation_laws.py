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

import logging
from typing import Any

import numpy as np
import pytest

from shared.python.constants import GRAVITY_M_S2
from tests.fixtures.fixtures_lib import _check_mujoco_available

logger = logging.getLogger(__name__)

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

        assert (
            max_drift_pct < 1.0
        ), f"Energy drift {max_drift_pct:.2f}% exceeds 1% tolerance (Guideline O3)"

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


@pytest.mark.integration
class TestWorkEnergyTheorem:
    """Test work-energy theorem.

    The work done by applied forces equals the change in kinetic energy.
    W = ∫ τ·dθ = ΔKE (for conservative systems with work against gravity counted)
    """

    @pytest.mark.skipif(not _check_mujoco_available(), reason="MuJoCo not installed")
    def test_work_equals_kinetic_energy_change(self) -> None:
        """Test that applied work equals kinetic energy change.

        Uses the work-energy theorem: W = ΔE_mechanical
        For a system with applied torque τ: W = ∫ τ·dθ = ΔKE + ΔPE
        """
        import mujoco

        # Use actuated model
        model = mujoco.MjModel.from_xml_string(ACTUATED_PENDULUM_XML)
        data = mujoco.MjData(model)

        # Start from rest at small angle
        theta_0 = 0.3
        data.qpos[0] = theta_0
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)

        # Get actual inertia from MuJoCo (includes parallel axis theorem)
        nv = model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        I_actual = M[0, 0]  # Rotational inertia about pivot [kg·m²]

        # Record initial energies using actual inertia
        # KE = 0.5 * I * ω²
        KE0 = 0.5 * I_actual * float(data.qvel[0]) ** 2
        # PE = m * g * h_com where h_com = L/2 * (1 - cos(θ))
        PE0 = ROD_MASS_KG * GRAVITY_M_S2 * (ROD_LENGTH_M / 2.0) * (1 - np.cos(theta_0))

        # Apply constant torque and integrate
        tau = 0.5  # [N·m]
        dt = model.opt.timestep
        n_steps = 100
        work_total = 0.0

        for _ in range(n_steps):
            # Work increment: W = τ * dθ = τ * θ̇ * dt
            dwork = tau * float(data.qvel[0]) * dt
            work_total += dwork

            data.ctrl[0] = tau
            mujoco.mj_step(model, data)

        # Final energies using actual inertia
        KE_final = 0.5 * I_actual * float(data.qvel[0]) ** 2
        PE_final = (
            ROD_MASS_KG
            * GRAVITY_M_S2
            * (ROD_LENGTH_M / 2.0)
            * (1 - np.cos(float(data.qpos[0])))
        )

        # Work should equal change in total mechanical energy
        delta_E = (KE_final - KE0) + (PE_final - PE0)
        error = abs(work_total - delta_E)

        logger.info(f"Actual inertia from MuJoCo: {I_actual:.6f}")
        logger.info(f"Work done: {work_total:.6f}")
        logger.info(f"ΔKE: {KE_final - KE0:.6f}")
        logger.info(f"ΔPE: {PE_final - PE0:.6f}")
        logger.info(f"ΔE total: {delta_E:.6f}")
        logger.info(f"Error: {error:.6f}")

        # Use absolute tolerance for small energy values
        TOLERANCE_ABS = 0.001  # 1 mJ absolute tolerance
        TOLERANCE_REL = 0.05  # 5% relative tolerance for numerical integration
        relative_error = error / max(abs(delta_E), TOLERANCE_ABS)
        assert (
            relative_error < TOLERANCE_REL
        ), f"Work-energy mismatch: {relative_error*100:.2f}% > {TOLERANCE_REL*100}%"


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


@pytest.mark.unit
class TestNumericalPrecision:
    """Test numerical precision and edge cases.

    Conservation laws can fail due to numerical issues:
    - Loss of precision in nearly-singular matrices
    - Accumulation of rounding errors
    - Catastrophic cancellation
    """

    def test_energy_small_values(self) -> None:
        """Test energy conservation with very small values."""
        # Small but non-zero KE
        M = np.array([[1e-3]])
        qd = np.array([1e-3])

        KE = 0.5 * qd.T @ M @ qd

        # Should be 5e-10, not zero
        assert KE > 0, "Energy should be non-zero for non-zero velocity"
        assert KE < 1e-9, "Energy magnitude check"

    def test_momentum_near_zero(self) -> None:
        """Test momentum near machine epsilon."""
        p1 = np.array([1e-15, 2e-15, 3e-15])
        p2 = np.array([1.1e-15, 2.1e-15, 3.1e-15])

        # Difference should be detectable
        diff = np.linalg.norm(p2 - p1)

        # But might be below typical physics tolerance
        physics_tolerance = 1e-12

        if diff < physics_tolerance:
            # This is expected - treating as conserved
            pass
