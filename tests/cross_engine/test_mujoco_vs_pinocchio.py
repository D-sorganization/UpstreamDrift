"""Cross-engine validation tests for physics consistency.

This module addresses Assessment C-006 by comparing MuJoCo physics computations
against Pinocchio (an independent rigid body dynamics library) to verify
scientific correctness.

SCIENTIFIC RATIONALE:
---------------------
Single-engine testing can hide engine-specific bugs or implementation errors.
Cross-engine validation ensures our physics results are not artifacts of
MuJoCo's specific algorithms but represent true physics.

TEST PHILOSOPHY:
----------------
1. **Simple Models**: Use analytically tractable systems (pendulum, double pendulum)
2. **Numerical Tolerance**: Allow small differences due to numerical methods
3. **Focus on Core Physics**: Test inverse dynamics, mass matrix, Jacobians
4. **Graceful Degradation**: Skip if Pinocchio not installed

TOLERANCE POLICY (Project Guideline P3):
----------------------------------------
- Relative error < 1e-6 for well-conditioned computations
- Relative error < 1e-4 for Jacobian derivatives (numerical methods differ)
- Absolute error < 1e-10 for near-zero values

REFERENCES:
-----------
- Carpentier et al., "Pinocchio: Fast Forward/Inverse Dynamics for Poly-Articulated Systems" (2019)
- MuJoCo Documentation: https://mujoco.readthedocs.io/
- Project Guidelines: docs/project_design_guidelines.qmd (Section P: Cross-Engine Validation)
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from src.shared.python.engine_availability import (
    MUJOCO_AVAILABLE,
    PINOCCHIO_AVAILABLE,
)

if TYPE_CHECKING:
    import mujoco
    import pinocchio

# Skip all tests if either engine is missing
pytestmark = pytest.mark.skipif(
    not (PINOCCHIO_AVAILABLE and MUJOCO_AVAILABLE),
    reason="Requires both MuJoCo and Pinocchio",
)


def create_simple_pendulum_mujoco() -> tuple[Any, Any]:
    """Create a simple pendulum model in MuJoCo.

    Returns:
        Tuple of (model, data) for MuJoCo
    """
    xml = """
    <mujoco model="pendulum">
        <compiler angle="radian" autolimits="true"/>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.001"/>

        <default>
            <joint damping="0.0" frictionloss="0.0"/>
            <geom density="1000"/>
        </default>

        <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>

            <body name="pendulum" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.01"/>
                <geom name="mass" type="sphere" pos="0 0 -0.5" size="0.05" mass="1.0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def create_simple_pendulum_pinocchio() -> tuple:
    """Create a simple pendulum model in Pinocchio.

    Returns:
        Tuple of (model, data) for Pinocchio
    """
    import pinocchio as pin

    # Create model
    model = pin.Model()

    # World frame
    parent_id = model.getFrameId("universe")

    # Pendulum link parameters
    length = 0.5  # meters
    mass = 1.0  # kg
    radius = 0.05  # meters

    # Inertia of sphere about its center
    I_sphere = (2.0 / 5.0) * mass * radius**2 * np.eye(3)

    # Parallel axis theorem: I_about_joint = I_cm + m * d²
    # where d = distance from joint to COM = length/2
    d = length
    I_parallel = I_sphere + mass * d**2 * np.eye(3)

    # Create inertia object
    inertia = pin.Inertia(mass, np.array([0.0, 0.0, -length]), I_parallel)

    # Add joint (revolute about Y-axis)
    joint_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, 1.0]))
    joint_id = model.addJoint(parent_id, pin.JointModelRY(), joint_placement, "hinge")

    # Add body
    model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())

    # Create data
    data = model.createData()

    return model, data


class TestCrossEngineInverseDynamics:
    """Cross-engine validation for inverse dynamics computations."""

    def test_simple_pendulum_zero_velocity(self) -> None:
        """Verify inverse dynamics match for pendulum at rest.

        PHYSICS:
        --------
        For a pendulum hanging at θ=0 with q̇=0, q̈=0:
        τ = g(q) = m * g * L * cos(θ)

        Both engines should give identical gravity compensation torques.
        """
        # MuJoCo model
        mj_model, mj_data = create_simple_pendulum_mujoco()

        # Pinocchio model
        pin_model, pin_data = create_simple_pendulum_pinocchio()

        # Test configuration: hanging down (θ = 0)
        q = np.array([0.0])
        v = np.array([0.0])
        a = np.array([0.0])

        # MuJoCo inverse dynamics
        mj_data.qpos[:] = q
        mj_data.qvel[:] = v
        mj_data.qacc[:] = a
        mujoco.mj_inverse(mj_model, mj_data)
        tau_mujoco = mj_data.qfrc_inverse.copy()

        # Pinocchio inverse dynamics
        tau_pinocchio = pinocchio.rnea(pin_model, pin_data, q, v, a)

        # Compare
        rel_error = np.abs(tau_mujoco - tau_pinocchio) / (np.abs(tau_pinocchio) + 1e-10)

        assert rel_error[0] < 1e-6, (
            f"Inverse dynamics mismatch at rest: "
            f"MuJoCo={tau_mujoco[0]:.6e}, "
            f"Pinocchio={tau_pinocchio[0]:.6e}, "
            f"rel_error={rel_error[0]:.2e}"
        )

    def test_simple_pendulum_with_motion(self) -> None:
        """Verify inverse dynamics match for pendulum in motion.

        PHYSICS:
        --------
        For a swinging pendulum:
        τ = M(q)q̈ + C(q,q̇)q̇ + g(q)

        Tests both inertial and velocity-dependent terms.
        """
        # MuJoCo model
        mj_model, mj_data = create_simple_pendulum_mujoco()

        # Pinocchio model
        pin_model, pin_data = create_simple_pendulum_pinocchio()

        # Test configuration: swinging
        q = np.array([np.pi / 4])  # 45 degrees
        v = np.array([1.0])  # 1 rad/s
        a = np.array([-0.5])  # decelerating

        # MuJoCo inverse dynamics
        mj_data.qpos[:] = q
        mj_data.qvel[:] = v
        mj_data.qacc[:] = a
        mujoco.mj_inverse(mj_model, mj_data)
        tau_mujoco = mj_data.qfrc_inverse.copy()

        # Pinocchio inverse dynamics
        tau_pinocchio = pinocchio.rnea(pin_model, pin_data, q, v, a)

        # Compare
        rel_error = np.abs(tau_mujoco - tau_pinocchio) / np.abs(tau_pinocchio)

        assert rel_error[0] < 1e-5, (
            f"Inverse dynamics mismatch in motion: "
            f"MuJoCo={tau_mujoco[0]:.6e}, "
            f"Pinocchio={tau_pinocchio[0]:.6e}, "
            f"rel_error={rel_error[0]:.2e}"
        )


class TestCrossEngineMassMatrix:
    """Cross-engine validation for mass matrix computations."""

    def test_mass_matrix_consistency(self) -> None:
        """Verify mass matrix matches between engines.

        PHYSICS:
        --------
        The mass matrix M(q) is configuration-dependent but velocity-independent.
        For a simple pendulum: M = I + m*L²
        """
        # MuJoCo model
        mj_model, mj_data = create_simple_pendulum_mujoco()

        # Pinocchio model
        pin_model, pin_data = create_simple_pendulum_pinocchio()

        # Test at multiple configurations
        test_angles = [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]

        for theta in test_angles:
            q = np.array([theta])

            # MuJoCo mass matrix
            mj_data.qpos[:] = q
            mujoco.mj_forward(mj_model, mj_data)
            M_mujoco = np.zeros((mj_model.nv, mj_model.nv))
            mujoco.mj_fullM(mj_model, M_mujoco, mj_data.qM)

            # Pinocchio mass matrix
            M_pinocchio = pinocchio.crba(pin_model, pin_data, q)

            # Compare
            rel_error = np.abs(M_mujoco - M_pinocchio) / (np.abs(M_pinocchio) + 1e-10)

            assert rel_error[0, 0] < 1e-6, (
                f"Mass matrix mismatch at θ={theta:.3f}: "
                f"MuJoCo={M_mujoco[0, 0]:.6e}, "
                f"Pinocchio={M_pinocchio[0, 0]:.6e}, "
                f"rel_error={rel_error[0, 0]:.2e}"
            )

    def test_mass_matrix_positive_definite(self) -> None:
        """Verify mass matrix is positive definite in both engines.

        PHYSICS:
        --------
        The mass matrix MUST be positive definite (all eigenvalues > 0).
        This is a fundamental requirement of classical mechanics.
        """
        # MuJoCo model
        mj_model, mj_data = create_simple_pendulum_mujoco()

        # Pinocchio model
        pin_model, pin_data = create_simple_pendulum_pinocchio()

        q = np.array([np.pi / 4])

        # MuJoCo mass matrix
        mj_data.qpos[:] = q
        mujoco.mj_forward(mj_model, mj_data)
        M_mujoco = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, M_mujoco, mj_data.qM)

        # Pinocchio mass matrix
        M_pinocchio = pinocchio.crba(pin_model, pin_data, q)

        # Check positive definiteness
        eig_mujoco = np.linalg.eigvalsh(M_mujoco)
        eig_pinocchio = np.linalg.eigvalsh(M_pinocchio)

        assert np.all(eig_mujoco > 0), (
            f"MuJoCo mass matrix not positive definite: "
            f"min eigenvalue = {eig_mujoco.min():.2e}"
        )

        assert np.all(eig_pinocchio > 0), (
            f"Pinocchio mass matrix not positive definite: "
            f"min eigenvalue = {eig_pinocchio.min():.2e}"
        )


class TestCrossEngineJacobians:
    """Cross-engine validation for Jacobian computations."""

    def test_jacobian_consistency(self) -> None:
        """Verify Jacobian matches between engines.

        PHYSICS:
        --------
        The Jacobian J maps joint velocities to Cartesian velocities:
        v = J(q) * q̇

        Both engines should produce identical Jacobians.
        """
        # MuJoCo model
        mj_model, mj_data = create_simple_pendulum_mujoco()

        # Pinocchio model
        pin_model, pin_data = create_simple_pendulum_pinocchio()

        # Test configuration
        q = np.array([np.pi / 4])

        # MuJoCo Jacobian (at end effector)
        mj_data.qpos[:] = q
        mujoco.mj_forward(mj_model, mj_data)

        # Get end effector body ID
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")

        # Compute Jacobian
        jacp_mj = np.zeros((3, mj_model.nv))
        jacr_mj = np.zeros((3, mj_model.nv))
        mujoco.mj_jacBody(mj_model, mj_data, jacp_mj, jacr_mj, body_id)

        # Pinocchio Jacobian
        pinocchio.computeJointJacobians(pin_model, pin_data, q)
        pinocchio.framesForwardKinematics(pin_model, pin_data, q)

        # Get Jacobian at last frame (pendulum end)
        frame_id = pin_model.nframes - 1
        J_pin = pinocchio.getFrameJacobian(
            pin_model, pin_data, frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Extract linear Jacobian (rows 3-6 in Pinocchio's 6D Jacobian)
        jacp_pin = J_pin[3:6, :]

        # Compare (allow slightly larger tolerance for numerical Jacobians)
        abs_error = np.abs(jacp_mj - jacp_pin)
        max_error = np.max(abs_error)

        assert max_error < 1e-4, (
            f"Jacobian mismatch: max error = {max_error:.2e}\n"
            f"MuJoCo:\n{jacp_mj}\n"
            f"Pinocchio:\n{jacp_pin}"
        )


class TestCrossEngineEnergyConsistency:
    """Verify energy calculations are consistent across engines."""

    def test_kinetic_energy_consistency(self) -> None:
        """Verify kinetic energy calculation matches.

        PHYSICS:
        --------
        Kinetic energy: KE = 0.5 * q̇^T * M(q) * q̇
        """
        # MuJoCo model
        mj_model, mj_data = create_simple_pendulum_mujoco()

        # Pinocchio model
        pin_model, pin_data = create_simple_pendulum_pinocchio()

        # Test state
        q = np.array([np.pi / 6])
        v = np.array([2.0])  # 2 rad/s

        # MuJoCo kinetic energy
        mj_data.qpos[:] = q
        mj_data.qvel[:] = v
        mujoco.mj_forward(mj_model, mj_data)
        ke_mujoco = mj_data.energy[0]  # Kinetic energy

        # Pinocchio kinetic energy
        ke_pinocchio = pinocchio.computeKineticEnergy(pin_model, pin_data, q, v)

        # Compare
        rel_error = abs(ke_mujoco - ke_pinocchio) / abs(ke_pinocchio)

        assert rel_error < 1e-6, (
            f"Kinetic energy mismatch: "
            f"MuJoCo={ke_mujoco:.6e}, "
            f"Pinocchio={ke_pinocchio:.6e}, "
            f"rel_error={rel_error:.2e}"
        )


# Integration test combining multiple physics aspects
@pytest.mark.slow
class TestCrossEngineIntegration:
    """Integration tests combining multiple physics computations."""

    def test_equation_of_motion_consistency(self) -> None:
        """Verify full equation of motion: τ = M(q)q̈ + C(q,q̇)q̇ + g(q).

        PHYSICS:
        --------
        This is the fundamental equation of rigid body dynamics.
        Both engines must satisfy it identically.
        """
        # MuJoCo model
        mj_model, mj_data = create_simple_pendulum_mujoco()

        # Pinocchio model
        pin_model, pin_data = create_simple_pendulum_pinocchio()

        # Test state
        q = np.array([np.pi / 3])
        v = np.array([1.5])
        a = np.array([-1.0])

        # === MuJoCo: Compute components ===
        mj_data.qpos[:] = q
        mj_data.qvel[:] = v
        mujoco.mj_forward(mj_model, mj_data)

        # Mass matrix
        M_mj = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, M_mj, mj_data.qM)

        # Bias forces (C*v + g)
        bias_mj = mj_data.qfrc_bias.copy()

        # Reconstruct: τ = M*a + bias
        tau_mj_reconstructed = M_mj @ a + bias_mj

        # Direct inverse dynamics
        mj_data.qacc[:] = a
        mujoco.mj_inverse(mj_model, mj_data)
        tau_mj_direct = mj_data.qfrc_inverse.copy()

        # === Pinocchio: Same computation ===
        M_pin = pinocchio.crba(pin_model, pin_data, q)

        # Bias forces
        bias_pin = pinocchio.rnea(pin_model, pin_data, q, v, np.zeros(1))

        # Reconstruct
        tau_pin_reconstructed = M_pin @ a + bias_pin

        # Direct
        tau_pin_direct = pinocchio.rnea(pin_model, pin_data, q, v, a)

        # === Cross-engine comparison ===
        rel_error = abs(tau_mj_direct[0] - tau_pin_direct[0]) / abs(tau_pin_direct[0])

        assert rel_error < 1e-5, (
            f"Equation of motion mismatch: "
            f"MuJoCo={tau_mj_direct[0]:.6e}, "
            f"Pinocchio={tau_pin_direct[0]:.6e}, "
            f"rel_error={rel_error:.2e}"
        )

        # === Internal consistency (each engine) ===
        mj_internal_error = abs(tau_mj_direct[0] - tau_mj_reconstructed[0])
        pin_internal_error = abs(tau_pin_direct[0] - tau_pin_reconstructed[0])

        assert mj_internal_error < 1e-10, (
            f"MuJoCo internal inconsistency: "
            f"direct={tau_mj_direct[0]:.6e}, "
            f"reconstructed={tau_mj_reconstructed[0]:.6e}"
        )

        assert pin_internal_error < 1e-10, (
            f"Pinocchio internal inconsistency: "
            f"direct={tau_pin_direct[0]:.6e}, "
            f"reconstructed={tau_pin_reconstructed[0]:.6e}"
        )
