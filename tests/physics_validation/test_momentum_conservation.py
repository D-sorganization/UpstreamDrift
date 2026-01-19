"""Physics validation tests verifying momentum conservation."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.python.engine_manager import EngineManager, EngineType

logger = logging.getLogger(__name__)


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is installed and importable."""
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return bool(probe_result.is_available())


def test_mujoco_momentum_conservation():
    """Verify linear momentum conservation in zero-gravity MuJoCo simulation."""
    if not is_engine_available(EngineType.MUJOCO):
        pytest.skip("MuJoCo not installed")

    import mujoco

    if isinstance(mujoco, MagicMock):
        pytest.skip("mujoco is mocked")

    # 1. Setup Simulation (Two colliding spheres, no gravity, no damping)
    # Using 'free' joint implies 6DOF
    xml = """
    <mujoco>
        <option timestep="0.002" gravity="0 0 0" density="0" viscosity="0"/>
        <worldbody>
            <body name="ball1" pos="-5 0 0">
                <freejoint/>
                <geom type="sphere" size="1.0" mass="1.0" friction="0 0 0" solref="0.02 1"/>
            </body>
            <body name="ball2" pos="5 0 0">
                <freejoint/>
                <geom type="sphere" size="1.0" mass="1.0" friction="0 0 0" solref="0.02 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    # Note: solref="0.02 1" makes collision elastic but likely not perfectly elastic in basic solver without careful tuning.
    # However, the total system momentum should be conserved in the absence of external forces (gravity is zero).
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 2. Initial State
    mujoco.mj_resetData(model, data)

    # Ball 1 moving +x at 1 m/s
    # qvel layout for free joint: [vx, vy, vz, wx, wy, wz]
    # Body 1 qvel is at distinct indices.
    # Model has 2 bodies + world. qvel size = 12 (6 * 2)

    # Set Ball 1 velocity (indices 0-2 for linear)
    data.qvel[0] = 1.0

    # Ball 2 stationary (default)

    # Compute Initial Momentum
    # P = m1*v1 + m2*v2 = 1.0*1.0 + 1.0*0 = 1.0
    initial_momentum_x = 1.0

    # 3. Run Simulation
    # Run long enough for collision
    steps = 5000  # 10 seconds

    momentums = []

    for _ in range(steps):
        mujoco.mj_step(model, data)

        # Calculate system momentum
        # Since geometries are spheres with mass 1.0
        # Check qvels
        v1x = data.qvel[0]
        v2x = data.qvel[6]  # Second joint starts at 6

        p = 1.0 * v1x + 1.0 * v2x
        momentums.append(p)

    # 4. Validation
    # Momentum should remain exactly 1.0 throughout (within float precision)
    # External forces = 0 => dP/dt = 0

    momentums_array = np.array(momentums)
    max_deviation = float(np.max(np.abs(momentums_array - initial_momentum_x)))

    logger.info(f"Max Momentum Deviation: {max_deviation:.6e}")

    # Tolerance: 1e-12 should be achievable for floating point arithmetic if truly conservative
    # But contact solver might introduce slight drift. Let's start with 1e-6.
    assert (
        max_deviation < 1e-6
    ), f"Momentum not conserved. Max deviation: {max_deviation}"


def test_pinocchio_momentum_conservation():
    """Verify momentum conservation for Pinocchio free floating bodies."""
    if not is_engine_available(EngineType.PINOCCHIO):
        pytest.skip("Pinocchio not installed")

    import pinocchio

    if isinstance(pinocchio, MagicMock):
        pytest.skip("pinocchio is mocked")

    # 1. Create Model
    model = pinocchio.Model()
    model.gravity.linear = np.zeros(3)  # No gravity!

    # Add two free bodies
    m = 1.0
    inertia = pinocchio.Inertia.FromSphere(m, 1.0)

    # Body 1
    idx1 = model.addJoint(
        0,
        pinocchio.JointModelFreeFlyer(),
        pinocchio.SE3(np.eye(3), np.array([-5, 0, 0])),
        "ball1",
    )
    model.appendBodyToJoint(idx1, inertia, pinocchio.SE3.Identity())
    model.addBodyFrame("b1", idx1, pinocchio.SE3.Identity(), idx1)

    # Body 2
    idx2 = model.addJoint(
        0,
        pinocchio.JointModelFreeFlyer(),
        pinocchio.SE3(np.eye(3), np.array([5, 0, 0])),
        "ball2",
    )
    model.appendBodyToJoint(idx2, inertia, pinocchio.SE3.Identity())
    model.addBodyFrame("b2", idx2, pinocchio.SE3.Identity(), idx2)

    data = model.createData()

    # 2. Initial State
    q = pinocchio.neutral(model)  # q is 7*2 = 14
    # Set positions (Already set in joint placement? No, joint placement is rest pose. q is relative.)
    # Actually addJoint placement defines the frame of the joint base.
    # But q defines the joint configuration.
    # For FreeFlyer, q=[x,y,z,qx,qy,qz,qw].
    # q stays at the neutral configuration from pinocchio.neutral(model), i.e. each free-flyer at the origin of its joint frame with identity orientation.

    v = np.zeros(model.nv)  # 6*2 = 12

    # Set Ball 1 velocity (vx = 1)
    # Indices 0-5 are for joint 1
    v[0] = 1.0

    # Initial Centroidal Momentum
    # HG = sum(I * v) roughly.
    # Pinocchio has computeCentroidalMomentum
    pinocchio.computeCentroidalMomentum(model, data, q, v)
    h0 = data.hg.copy()  # Spatial momentum (linear + angular)

    # 3. Simulate (No contacts, just flying)
    # If there are no contacts, a simple integration should conserve momentum perfectly.
    # Pinocchio doesn't simulate contacts natively in the core 'aba'.
    # Contacts require a constraint solver (like in specialized extras).
    # Step: Just integrate.

    dt = 0.01
    steps = 100

    for _ in range(steps):
        a = pinocchio.aba(model, data, q, v, np.zeros(model.nv))
        v_next = v + a * dt
        q_next = pinocchio.integrate(model, q, v_next * dt)

        q = q_next
        v = v_next

        pinocchio.computeCentroidalMomentum(model, data, q, v)
        h_current = data.hg

        # Check conservation
        # Linear part is h.linear
        diff = np.linalg.norm(h_current.linear - h0.linear)
        assert diff < 1e-12, f"Momentum drift detected: {diff}"

    logger.info("Pinocchio momentum conserved in free flight.")
