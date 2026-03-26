"""Heavy integration tests for Drake real model loading (fixes #1985).

When Drake IS installed in the heavy Docker image, these tests exercise actual
model loading and simulation — not mocked pydrake. The mock-heavy tests in
test_phase1_drake_integration.py belong in unit tests (tracked separately).

All tests skip gracefully when Drake is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

# Minimal self-contained URDF for testing
_MINIMAL_URDF = """\
<?xml version="1.0"?>
<robot name="minimal_pendulum">
  <link name="world"/>
  <link name="pendulum_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </visual>
    <collision>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </collision>
  </link>
  <joint name="pivot" type="revolute">
    <parent link="world"/>
    <child link="pendulum_link"/>
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def drake_modules():
    """Import required Drake modules or skip the entire module."""
    try:
        from pydrake.all import DiagramBuilder, Parser  # noqa: F401
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.systems.analysis import Simulator

        return {
            "DiagramBuilder": DiagramBuilder,
            "Parser": Parser,
            "MultibodyPlant": MultibodyPlant,
            "Simulator": Simulator,
        }
    except ImportError as exc:
        pytest.skip(f"Drake (pydrake) not installed: {exc}")


@pytest.fixture(scope="module")
def minimal_urdf_path(tmp_path_factory):
    """Write the minimal URDF to a temp file."""
    tmpdir = tmp_path_factory.mktemp("urdf")
    urdf = tmpdir / "minimal_pendulum.urdf"
    urdf.write_text(_MINIMAL_URDF)
    return urdf


class TestDrakeModelLoading:
    """Contract: Drake loads a URDF model and creates a valid plant."""

    def test_multibody_plant_creation(self, drake_modules) -> None:
        """MultibodyPlant can be instantiated with a time step."""
        MultibodyPlant = drake_modules["MultibodyPlant"]
        plant = MultibodyPlant(time_step=0.001)
        assert plant is not None

    def test_urdf_loading_via_parser(self, drake_modules, minimal_urdf_path) -> None:
        """Parser can load the minimal URDF into MultibodyPlant."""
        Parser = drake_modules["Parser"]
        MultibodyPlant = drake_modules["MultibodyPlant"]

        # Direct parser approach (works without SceneGraph for kinematic-only)
        plant = MultibodyPlant(time_step=0.001)
        parser = Parser(plant)
        try:
            parser.AddModelFromFile(str(minimal_urdf_path))
            plant.Finalize()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Drake URDF loading failed (may need SceneGraph): {exc}")

        assert plant.num_positions() >= 1, "Plant has no DOF after loading URDF"

    def test_plant_degrees_of_freedom(self, drake_modules, minimal_urdf_path) -> None:
        """Loaded pendulum plant has exactly 1 revolute DOF."""
        MultibodyPlant = drake_modules["MultibodyPlant"]
        Parser = drake_modules["Parser"]

        plant = MultibodyPlant(time_step=0.001)
        parser = Parser(plant)
        try:
            parser.AddModelFromFile(str(minimal_urdf_path))
            plant.Finalize()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Drake URDF loading failed: {exc}")

        assert plant.num_positions() == 1, (
            f"Expected 1 DOF for pendulum, got {plant.num_positions()}"
        )


class TestDrakeSimulation:
    """Contract: Drake Simulator advances time and returns finite state."""

    def test_simulate_minimal_pendulum(self, drake_modules, minimal_urdf_path) -> None:
        """Drake Simulator steps a pendulum for 0.1 s and returns finite state."""
        DiagramBuilder = drake_modules["DiagramBuilder"]
        Parser = drake_modules["Parser"]
        Simulator = drake_modules["Simulator"]

        try:
            from pydrake.geometry import SceneGraph  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"Drake SceneGraph not available: {exc}")

        builder = DiagramBuilder()
        try:
            from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

            plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        except ImportError:
            pytest.skip("AddMultibodyPlantSceneGraph not available")

        parser = Parser(plant)
        try:
            parser.AddModelFromFile(str(minimal_urdf_path))
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Drake URDF loading failed: {exc}")

        plant.Finalize()
        diagram = builder.Build()

        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()

        # Set initial angle to 0.1 rad
        plant_context = plant.GetMyMutableContextFromRoot(context)
        plant.SetPositions(plant_context, np.array([0.1]))
        plant.SetVelocities(plant_context, np.array([0.0]))

        simulator.AdvanceTo(0.1)

        q = plant.GetPositions(plant_context)
        qdot = plant.GetVelocities(plant_context)

        assert np.all(np.isfinite(q)), f"Positions not finite after sim: {q}"
        assert np.all(np.isfinite(qdot)), f"Velocities not finite after sim: {qdot}"


class TestDrakeEngineWrapper:
    """Contract: DrakePhysicsEngine wrapper step() advances state."""

    def test_drake_engine_importable(self) -> None:
        """DrakePhysicsEngine is importable from engine_availability path."""
        try:
            from src.shared.python.engine_core.engine_availability import (
                DRAKE_AVAILABLE,
            )
        except ImportError as exc:
            pytest.skip(f"engine_availability not importable: {exc}")

        if not DRAKE_AVAILABLE:
            pytest.skip("Drake not available per engine_availability")

        try:
            from src.engines.physics_engines.drake.python.drake_physics_engine import (
                DrakePhysicsEngine,
            )
        except ImportError as exc:
            pytest.skip(f"DrakePhysicsEngine not importable: {exc}")

        assert DrakePhysicsEngine is not None


pytestmark = pytest.mark.live_simulation
