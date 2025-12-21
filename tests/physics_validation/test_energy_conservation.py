"""Physics validation tests verifying energy conservation."""

import logging

import numpy as np
import pytest

from shared.python.engine_manager import EngineManager, EngineType
from tests.physics_validation.analytical import AnalyticalBallistic

logger = logging.getLogger(__name__)


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is installed and importable."""
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return probe_result.is_available()


@pytest.mark.skipif(
    not is_engine_available(EngineType.MUJOCO), reason="MuJoCo not installed"
)
def test_mujoco_ballistic_energy_conservation():
    """Verify energy conservation for a falling particle in MuJoCo."""
    import mujoco

    # 1. Setup Simulation
    # Define a simple XML model: a particle falling under gravity
    xml = """
    <mujoco>
        <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>
        <worldbody>
            <body name="ball" pos="0 0 10">
                <joint type="free"/>
                <geom type="sphere" size="0.1" mass="1.0"/>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Analytical Solver
    baseline = AnalyticalBallistic(mass=1.0, g=9.81)

    # Initial State
    mujoco.mj_resetData(model, data)
    initial_energy = baseline.total_energy(height=10.0, velocity=0.0)

    # Run Simulation for 1 second
    steps = 1000
    energies = []

    for _ in range(steps):
        mujoco.mj_step(model, data)

        # Get state
        height = data.qpos[2]  # z position (free joint: x,y,z, qw,qx,qy,qz)

        # Velocity magnitude
        # qvel has 6 DOFs for free joint: vx, vy, vz, wx, wy, wz
        vz = data.qvel[2]
        velocity = abs(vz)

        # Calculate Energy
        current_energy = baseline.total_energy(height, velocity)
        energies.append(current_energy)

    # Validation
    # Energy should be conserved (numerical error only)
    # Allow 0.1% error for RK4 integration
    max_deviation = np.max(np.abs(np.array(energies) - initial_energy))
    percent_error = (max_deviation / initial_energy) * 100

    logger.info(f"MuJoCo Ballistic Energy Error: {percent_error:.4f}%")
    assert percent_error < 0.1, f"Energy not conserved! Error: {percent_error:.4f}%"


@pytest.mark.skipif(
    not is_engine_available(EngineType.PINOCCHIO), reason="Pinocchio not installed"
)
def test_pinocchio_energy_check():
    """Verify energy conservation with Pinocchio using explicit integration."""
    import pinocchio

    # 1. Create Model (Free floating body)
    model = pinocchio.Model()

    # Add a free flyer joint for the body
    joint_name = "particle_joint"
    joint_id = model.addJoint(
        0, pinocchio.JointModelFreeFlyer(), pinocchio.SE3.Identity(), joint_name
    )

    # Add a body with mass 1.0 at origin of joint
    mass = 1.0
    inertia = pinocchio.Inertia.FromSphere(mass, 0.1)
    model.appendBodyToJoint(joint_id, inertia, pinocchio.SE3.Identity())
    model.addBodyFrame("particle", joint_id, pinocchio.SE3.Identity(), 0)

    data = model.createData()

    # 2. Initial Conditions
    # q = [x, y, z, qx, qy, qz, qw] (7D for SE3)
    # v = [vx, vy, vz, wx, wy, wz] (6D se3)
    q = pinocchio.neutral(model)
    q[2] = 10.0  # z = 10m

    v = np.zeros(model.nv)

    # 3. Simulation Loop (Symplectic Euler / Semi-implicit)
    dt = 0.001
    steps = 1000

    # Pinocchio calculates potential energy automatically if gravity is set?
    # No, model.gravity needs to be set.
    model.gravity = pinocchio.Motion(np.array([0, 0, -9.81, 0, 0, 0]))

    # Pre-compute initial energy
    pinocchio.computeTotalEnergy(model, data, q, v)
    initial_energy = data.kinetic_energy + data.potential_energy

    errors = []

    for _ in range(steps):
        # Forward Dynamics (ABA) -> returns acceleration
        a = pinocchio.aba(model, data, q, v, np.zeros(model.nv))

        # Semi-implicit Euler
        v_next = v + a * dt
        q_next = pinocchio.integrate(model, q, v_next * dt)

        q = q_next
        v = v_next

        # Check Energy
        # Note: Symplectic Euler is stable but not perfectly energy conserving
        # It oscillates around the true energy.
        pinocchio.computeMechanicalEnergy(model, data, q, v)
        current_energy = data.mechanical_energy

        errors.append(abs(current_energy - initial_energy))

    max_error = np.max(errors)
    logger.info(f"Max Energy Error (Pinocchio): {max_error:.6f} J")

    # Allow slightly higher error due to simple integrator
    assert max_error < 0.05, f"Pinocchio energy check failed. Error: {max_error}"
