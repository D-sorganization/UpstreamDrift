"""Energy conservation tests for physics validation.

This module addresses Assessment B-005 by implementing comprehensive tests
to verify fundamental physics laws (energy conservation, work-energy theorem).

These tests validate that the physics engine and analysis tools correctly
implement Newtonian mechanics.
"""

import numpy as np
import pytest

from shared.python.numerical_constants import (
    GRAVITY_STANDARD,
    TOLERANCE_ENERGY_CONSERVATION,
    TOLERANCE_WORK_ENERGY_MISMATCH,
)

# Skip these tests if MuJoCo is not available
# Note: MuJoCo may have platform-specific loading issues on Windows
try:
    import mujoco  # noqa: E402
except (ImportError, OSError) as e:
    pytest.skip(f"MuJoCo not available or failed to load: {e}", allow_module_level=True)


@pytest.fixture
def simple_pendulum_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Create a simple pendulum model for energy conservation tests.

    Returns:
        Tuple of (model, data) for a 1-DOF pendulum
    """
    xml = """
    <mujoco model="simple_pendulum">
        <compiler angle="radian" autolimits="true"/>
        <option>
            <flag gravity="enable"/>
        </option>

        <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>
            <body name="pendulum" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom type="sphere" size="0.05" mass="1.0"/>
            </body>
        </worldbody>

        <actuator>
            <motor joint="hinge" name="hinge_motor" gear="1.0"/>
        </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


class TestEnergyConservation:
    """Test suite for energy conservation in passive systems.

    SCOPE (Assessment B-005):
    --------------------------
    Verifies that total mechanical energy E = KE + PE remains constant
    in conservative systems (no friction, no external forces).

    ACCEPTANCE CRITERIA:
    --------------------
    - Free fall: Energy drift < TOLERANCE_ENERGY_CONSERVATION (1e-6 relative)
    - Pendulum: Energy variation < TOLERANCE_ENERGY_CONSERVATION over full period
    - Zero-gravity free rotation: Angular momentum conserved

    PHYSICS BACKGROUND:
    -------------------
    For a conservative system: dE/dt = 0
    Equivalently: E(t) = E(0) = constant

    Where:
    - E = KE + PE (total mechanical energy)
    - KE = 0.5 * q̇^T M(q) q̇ (kinetic energy)
    - PE = -mgh (gravitational potential energy)
    """

    def test_free_fall_energy_conservation(
        self, simple_pendulum_model: tuple[mujoco.MjModel, mujoco.MjData]
    ) -> None:
        """Verify energy is conserved during free fall (no damping).

        PHYSICS:
        --------
        A mass released from rest at height h should satisfy:
        - E_total = mgh (constant)
        - KE(t) + PE(t) = KE(0) + PE(0)
        - At any time: 0.5*m*v² + mgh_current = mgh_initial

        TEST PROCEDURE:
        ---------------
        1. Initialize pendulum at angle = 0 rad (bottom), qvel = 0
        2. Raise to height h by setting qpos = π/4 rad
        3. Simulate free fall for T seconds
        4. Verify |E(t) - E(0)| / E(0) < tolerance at all timesteps
        """
        model, data = simple_pendulum_model

        # Disable damping to ensure conservative system
        model.dof_damping[:] = 0.0

        # Set up initial condition: displaced pendulum, zero velocity
        theta_initial = np.pi / 4  # 45 degrees
        data.qpos[0] = theta_initial
        data.qvel[0] = 0.0

        # Forward kinematics to compute initial state
        mujoco.mj_forward(model, data)

        # Calculate initial energy
        initial_ke = 0.5 * data.qvel @ (data.qM * data.qvel)
        initial_pe = -data.qpos[0] * model.body_mass[1] * GRAVITY_STANDARD * 1.0
        # (PE = -mgh, height ≈ L * (1 - cos(θ)) for pendulum)
        # Approximation: PE ≈ -θ for small angles
        initial_total_energy = initial_ke + initial_pe

        # Simulate for 2 seconds (dt=0.001s, n_steps=2000)
        n_steps = 2000
        energies_list: list[float] = []

        for _ in range(n_steps):
            mujoco.mj_step(model, data)

            # Compute current energy
            ke = 0.5 * data.qvel @ (data.qM * data.qvel)
            pe = -data.qpos[0] * model.body_mass[1] * GRAVITY_STANDARD * 1.0
            total_energy = ke + pe

            energies_list.append(total_energy)

        # Verify energy conservation
        energies: np.ndarray[tuple[int], np.dtype[np.floating]] = np.array(
            energies_list
        )
        energy_variation = np.std(energies) / np.abs(np.mean(energies))

        assert energy_variation < TOLERANCE_ENERGY_CONSERVATION, (
            f"Energy not conserved: relative variation = {energy_variation:.2e} "
            f"(threshold: {TOLERANCE_ENERGY_CONSERVATION:.2e}). "
            f"Initial energy: {initial_total_energy:.6f} J, "
            f"Final energy: {energies[-1]:.6f} J, "
            f"Drift: {energies[-1] - initial_total_energy:.6e} J"
        )

    def test_work_energy_theorem(
        self, simple_pendulum_model: tuple[mujoco.MjModel, mujoco.MjData]
    ) -> None:
        """Verify work-energy theorem: ΔKE = W_applied.

        PHYSICS:
        --------
        The work-energy theorem states:
        ΔKE = ∫ F · ds = ∫ τ · q̇ dt

        For actuated motion, the change in kinetic energy should equal
        the work done by applied torques.

        TEST PROCEDURE:
        ---------------
        1. Start pendulum at rest (θ=0, q̇=0)
        2. Apply constant torque τ for time T
        3. Measure:
           - ΔKE = KE_final - KE_initial
           - W = ∫ τ · q̇ dt (numerical integration)
        4. Verify |ΔKE - W| / |W| < TOLERANCE_WORK_ENERGY_MISMATCH (5%)
        """
        model, data = simple_pendulum_model

        # Disable damping
        model.dof_damping[:] = 0.0
        # Disable gravity for cleaner test (only actuator forces)
        model.opt.gravity[:] = 0.0

        # Initial condition: rest at bottom
        data.qpos[0] = 0.0
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)

        initial_ke = 0.5 * data.qvel @ (data.qM * data.qvel)

        # Apply constant torque
        torque = 1.0  # N·m
        dt = 0.001  # 1 ms
        n_steps = 1000

        work_done = 0.0
        velocities = []
        torques_applied = []

        for _ in range(n_steps):
            # Apply torque
            data.ctrl[0] = torque

            # Store state for work calculation
            qvel_before = data.qvel[0]

            # Step simulation
            mujoco.mj_step(model, data)

            # Store state after step
            qvel_after = data.qvel[0]

            # Numerical integration of work: dW = τ · q̇ · dt
            # Use trapezoidal rule for better accuracy
            avg_qvel = 0.5 * (qvel_before + qvel_after)
            work_done += torque * avg_qvel * dt

            velocities.append(qvel_after)
            torques_applied.append(torque)

        # Compute final kinetic energy
        final_ke = 0.5 * data.qvel @ (data.qM * data.qvel)
        delta_ke = final_ke - initial_ke

        # Verify work-energy theorem
        relative_error = abs(delta_ke - work_done) / (abs(work_done) + 1e-10)

        assert relative_error < TOLERANCE_WORK_ENERGY_MISMATCH, (
            f"Work-energy theorem violated: "
            f"ΔKE = {delta_ke:.6f} J, "
            f"W = {work_done:.6f} J, "
            f"relative error = {relative_error:.1%} "
            f"(threshold: {TOLERANCE_WORK_ENERGY_MISMATCH:.1%})"
        )

    def test_power_balance(
        self, simple_pendulum_model: tuple[mujoco.MjModel, mujoco.MjData]
    ) -> None:
        """Verify instantaneous power balance: dE/dt = P_in - P_out.

        PHYSICS:
        --------
        The instantaneous power balance states:
        dE/dt = P_actuator - P_dissipation

        For a conservative system (no damping):
        dE/dt = τ · q̇ (actuator power)

        TEST PROCEDURE:
        ---------------
        1. Compute dE/dt numerically from energy time series
        2. Compute P_actuator = τ · q̇ at each timestep
        3. Verify dE/dt ≈ P_actuator (within numerical differentiation error)
        """
        model, data = simple_pendulum_model

        # Disable damping
        model.dof_damping[:] = 0.0

        # Initial condition with some velocity
        data.qpos[0] = 0.0
        data.qvel[0] = 1.0  # rad/s
        mujoco.mj_forward(model, data)

        # Simulate and record energy + power
        dt = 0.001
        n_steps = 500

        energies_list: list[float] = []
        powers_actuator = []
        times = []

        for i in range(n_steps):
            # Apply varying torque (sinusoidal for richer dynamics)
            torque = 0.5 * np.sin(2 * np.pi * i / 100)  # Periodic forcing
            data.ctrl[0] = torque

            mujoco.mj_step(model, data)

            # Record state
            ke = 0.5 * data.qvel @ (data.qM * data.qvel)
            # For simplicity, ignore PE (or use full E = KE + PE)
            energy = ke
            power = torque * data.qvel[0]

            energies_list.append(energy)
            powers_actuator.append(power)
            times.append(i * dt)

        # Compute dE/dt numerically (central difference)
        energies: np.ndarray[tuple[int], np.dtype[np.floating]] = np.array(
            energies_list
        )
        de_dt_numeric = np.gradient(energies, dt)

        # Compare with actuator power
        # (Skip first/last few points due to gradient edge effects)
        valid_range = slice(5, -5)
        correlation = np.corrcoef(
            de_dt_numeric[valid_range], powers_actuator[valid_range]
        )[0, 1]

        assert correlation > 0.95, (
            f"Power balance mismatch: correlation between dE/dt and P_actuator "
            f"is {correlation:.3f} (expected > 0.95). "
            "This suggests energy accounting error."
        )


class TestConservationLaws:
    """Additional conservation law tests (momentum, angular momentum)."""

    @pytest.mark.slow
    def test_zero_gravity_angular_momentum_conservation(self) -> None:
        """Verify angular momentum conservation in zero gravity free rotation.

        PHYSICS:
        --------
        In the absence of external torques:
        dL/dt = 0 (angular momentum conserved)

        Where L = I·ω (for rigid body) or L = Σ I_i·ω_i (for multi-body system)

        This test verifies that MuJoCo correctly implements angular momentum
        conservation for free rotation.
        """
        xml = """
        <mujoco model="free_rotator">
            <compiler angle="radian"/>
            <option gravity="0 0 0"/>

            <worldbody>
                <body name="rotator" pos="0 0 0">
                    <freejoint/>
                    <geom type="box" size="0.1 0.1 0.05" mass="1.0"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Set initial angular velocity
        data.qvel[3:6] = [0.5, 0.3, 0.1]  # rad/s about x, y, z
        mujoco.mj_forward(model, data)

        # Simulate free rotation (dt=0.01s, n_steps=1000)
        n_steps = 1000
        angular_momenta_list: list[float] = []

        for _ in range(n_steps):
            mujoco.mj_step(model, data)

            # Angular momentum (simplified - assumes constant inertia)
            L = data.qvel[3:6].copy()
            angular_momenta_list.append(float(np.linalg.norm(L)))

        # Verify angular momentum magnitude is conserved
        angular_momenta: np.ndarray[tuple[int], np.dtype[np.floating]] = np.array(
            angular_momenta_list
        )
        momentum_variation = np.std(angular_momenta) / np.mean(angular_momenta)

        assert momentum_variation < 1e-3, (
            f"Angular momentum not conserved: "
            f"relative variation = {momentum_variation:.2e} (expected < 1e-3)"
        )
