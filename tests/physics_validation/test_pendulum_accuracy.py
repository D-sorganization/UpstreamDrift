"""Physics validation tests verifying pendulum dynamics accuracy."""

import logging
import sys
from unittest.mock import MagicMock

# Mock pydrake tree if missing (before other imports potentially load it)
if "pydrake.multibody.tree" not in sys.modules:
    sys.modules["pydrake.multibody.tree"] = MagicMock()

import numpy as np
import pytest

from shared.python.constants import GRAVITY_M_S2
from shared.python.engine_manager import EngineManager, EngineType
from tests.physics_validation.analytical import AnalyticalPendulum

logger = logging.getLogger(__name__)


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is installed and importable."""
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return bool(probe_result.is_available())


def test_mujoco_pendulum_accuracy():
    """Verify MuJoCo pendulum matches analytical solution."""
    if not is_engine_available(EngineType.MUJOCO):
        pytest.skip("MuJoCo not installed")

    import mujoco

    if isinstance(mujoco, MagicMock):
        pytest.skip("mujoco is mocked")

    # 1. Model: Simple Pendulum (L=1, m=1)
    # Standard pendulum model with point mass at end of massless rod.
    xml = f"""
    <mujoco>
        <option timestep="0.001" gravity="0 0 -{float(GRAVITY_M_S2)}" integrator="RK4"/>
        <worldbody>
            <body>
                <joint name="pin" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom type="cylinder" fromto="0 0 0 0 0 -1" size="0.02" mass="0"/>
                <body pos="0 0 -1">
                    <geom type="sphere" size="0.1" mass="1.0"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 2. Analytical Baseline
    # MuJoCo model is a sphere, so it has rotational inertia I = 2/5 * m * r^2
    # Total inertia about pivot = m * L^2 + 2/5 * m * r^2
    # m=1.0, L=1.0, r=0.1
    i_rod = 1.0 * 1.0**2
    i_sphere = 0.4 * 1.0 * (0.1**2)
    total_inertia = i_rod + i_sphere  # 1.004

    analytical = AnalyticalPendulum(
        length=1.0, mass=1.0, g=GRAVITY_M_S2, inertia=total_inertia
    )

    # 3. Initial Conditions
    # Release from 90 degrees (horizontal)
    initial_theta = np.pi / 2
    mujoco.mj_resetData(model, data)
    data.qpos[0] = initial_theta

    # 4. Simulation Loop
    duration = 2.0  # seconds (approx one full period)
    dt = 0.001
    steps = int(duration / dt)

    errors = []

    for _ in range(steps):
        mujoco.mj_step(model, data)

        # Current state
        theta = data.qpos[0]
        omega = data.qvel[0]

        # Check Total Energy Conservation
        # (This is a strong proxy for dynamic accuracy)
        current_energy = analytical.total_energy(theta, omega)
        initial_energy = analytical.total_energy(initial_theta, 0.0)

        # Error
        error = abs(current_energy - initial_energy)
        errors.append(error)

    # 5. Assertions
    max_energy_error = np.max(errors)
    logger.info(f"Max Energy Error (MuJoCo): {max_energy_error:.6f} J")

    # Allow small numerical integration error
    assert max_energy_error < 0.01, (
        f"MuJoCo pendulum drifted! Max error: {max_energy_error}"
    )


def test_drake_pendulum_accuracy():
    """Verify Drake pendulum matches analytical solution."""
    if not is_engine_available(EngineType.DRAKE):
        pytest.skip("Drake not installed")

    import pydrake.multibody.tree as mut

    if isinstance(mut, MagicMock):
        pytest.skip("pydrake is mocked")
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder

    # 1. Setup Drake System
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    # Create simple pendulum programmatically
    # Mass = 1.0, Length = 1.0
    # Inertia about pivot = m * L^2 = 1.0
    # For a point-mass pendulum, the COM is at (0, 0, -L),
    # i.e., a distance L below the pivot.

    # Add a rigid body for the pendulum
    # SpatialInertia(mass, com, unit_inertia)
    # Point mass m=1 at distance L=1 from origin
    M = 1.0
    L = 1.0

    # Moment of inertia for point mass at distance L is ML^2
    # About COM (the point itself), inertia is 0.
    # About pivot, it is ML^2.
    # Drake requires inertia about COM. For point mass, it's near zero.
    # Let's say it's a small sphere.
    # UnitInertia.PointMass() creates a UnitInertia for a point mass located at
    # a specific position relative to the body origin.
    # We want the mass to be concentated at (0, 0, -L).
    # UnitInertia depends only on geometry (distribution), not mass scale (M).
    # Its argument is the position of the point mass.
    com_vector = [0.0, 0.0, -L]
    unit_inertia = mut.UnitInertia.PointMass(com_vector)
    spatial_inertia = mut.SpatialInertia(M, com_vector, unit_inertia)

    pendulum = plant.AddRigidBody("pendulum", spatial_inertia)

    # Add hinge joint
    # Add hinge joint
    # Use explicit AddJoint for compatibility
    hinge = mut.RevoluteJoint(
        "pivot",
        plant.world_body().body_frame(),
        pendulum.body_frame(),
        [0, 1, 0],  # axis (Y)
        damping=0.0,
    )
    plant.AddJoint(hinge)

    plant.Finalize()
    diagram = builder.Build()

    # 2. Simulator
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # 3. Initial Conditions (Horizontal start)
    # Joint angle 0 = straight down? Usually.
    # We want horizontal. -pi/2 or pi/2.
    initial_theta = np.pi / 2
    plant.SetPositions(plant_context, [initial_theta])
    plant.SetVelocities(plant_context, [0.0])

    # 4. Run
    # Run short steps to verify energy
    analytical = AnalyticalPendulum(length=L, mass=M, g=GRAVITY_M_S2)

    duration = 1.0
    dt = 0.01
    steps = int(duration / dt)

    errors = []

    for _ in range(steps):
        simulator.AdvanceTo(context.get_time() + dt)

        # Get state
        q = plant.GetPositions(plant_context)[0]
        v = plant.GetVelocities(plant_context)[0]

        # Check Energy
        # Drake defines potential energy relative to z=0?
        # Analytical defines PE relative to bottom (z=-L)
        # Let's check Total Mechanical Energy conservation relative to initial

        # Calculate current physical energy from analytical model
        # (Assuming mapping q=0 is down)
        current_energy = analytical.total_energy(q, v)
        initial_energy = analytical.total_energy(initial_theta, 0.0)

        errors.append(abs(current_energy - initial_energy))

    max_error = np.max(errors)
    logger.info(f"Max Energy Error (Drake): {max_error:.6f} J")

    # Drake is usually very accurate
    assert max_error < 0.01, f"Drake conservation failed. Error: {max_error}"
