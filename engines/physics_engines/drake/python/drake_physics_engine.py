"""Drake Physics Engine wrapper implementation.

Wraps pydrake.multibody to provide a compliant PhysicsEngine interface.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

# Pydrake imports likely to fail if not installed, requiring try/except block
# in a real environment, but here we assume the environment satisfies requirements
# or that this module is only imported when safe.
try:
    import pydrake.math  # noqa: F401
    import pydrake.multibody.parsing as mbparsing  # noqa: F401
    import pydrake.multibody.plant as mbp  # noqa: F401
    import pydrake.systems.analysis as analysis
    import pydrake.systems.framework as framework
    from pydrake.all import (
        AddMultibodyPlantSceneGraph,
        DiagramBuilder,
        JacobianWrtVariable,
        LoadModelDirectives,  # noqa: F401
        MultibodyPlant,
        Parser,
        ProcessModelDirectives,  # noqa: F401
        RigidTransform,  # noqa: F401
        RotationMatrix,  # noqa: F401
    )
except ImportError:
    # Just to allow linting/static analysis if pydrake is missing
    pass

from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)


class DrakePhysicsEngine(PhysicsEngine):
    """Encapsulates Drake MultibodyPlant and simulation control.

    Implements the shared PhysicsEngine protocol.
    """

    def __init__(self, time_step: float = 0.001) -> None:
        """Initialize the Drake physics engine.

        Args:
            time_step: Simulation time step in seconds.
        """
        self.builder = DiagramBuilder()
        self.plant: MultibodyPlant
        self.scene_graph: Any
        # AddMultibodyPlantSceneGraph returns (plant, scene_graph)
        result = AddMultibodyPlantSceneGraph(self.builder, time_step)
        self.plant = result[0]
        self.scene_graph = result[1]

        self.diagram: framework.Diagram | None = None
        self.context: framework.Context | None = None
        self.plant_context: framework.Context | None = None

        self.model_name_str: str = ""
        self._is_finalized = False
        self.simulator: analysis.Simulator | None = None

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        return self.model_name_str

    def _ensure_finalized(self) -> None:
        """Finalize the plant and build diagram if not already done."""
        if not self._is_finalized:
            self.plant.Finalize()
            self.diagram = self.builder.Build()
            self.context = self.diagram.CreateDefaultContext()
            self.plant_context = self.plant.GetMyContextFromRoot(self.context)
            # Create persistent simulator to avoid overhead
            self.simulator = analysis.Simulator(self.diagram, self.context)
            self.simulator.Initialize()
            self._is_finalized = True

    def load_from_path(self, path: str) -> None:
        """Load model from file path (URDF, SDF, MJCF if supported)."""
        # Drake Parser supports SDF, URDF, MJCF (experimental)
        parser = Parser(self.plant)
        # We can try to infer model name from path
        model_name = path.split("/")[-1].split(".")[0]
        self.model_name_str = model_name

        try:
            # Add model to plant
            parser.AddModels(path)
        except Exception as e:
            LOGGER.error("Failed to load Drake model from path %s: %s", path, e)
            raise

        # We don't finalize here immediately to allow adding more models if needed?
        # But protocol implies "load then run".
        # So we should finalize implicitly on first usage or explicit?
        # For simplicity, we assume one load call per "engine setup".
        self._ensure_finalized()

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from string content."""
        parser = Parser(self.plant)
        ext = extension if extension else "urdf"  # Default to URDF if unknown?

        try:
            parser.AddModelsFromString(content, ext)
            self.model_name_str = "StringLoadedModel"
        except Exception as e:
            LOGGER.error("Failed to load Drake model from string: %s", e)
            raise

        self._ensure_finalized()

    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        if self.context and self.plant_context and self.simulator:
            # Reset time to zero
            self.context.SetTime(0.0)

            # Reset to default state (positions and velocities)
            self.plant.SetDefaultPositions(self.plant_context)
            self.plant.SetDefaultVelocities(self.plant_context)

            # Re-initialize the simulator with the reset state
            self.simulator.Initialize()

            LOGGER.debug("Drake engine reset to initial state")

    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one time step."""
        self._ensure_finalized()

        if not self.simulator or not self.context:
            LOGGER.error("Cannot step: Simulator not initialized.")
            return

        current_time = self.context.get_time()
        step_size = dt if dt is not None else self.plant.time_step()
        self.simulator.AdvanceTo(current_time + step_size)

    def forward(self) -> None:
        """Compute forward kinematics/dynamics without advancing time."""
        if not self.plant_context:
            LOGGER.warning(
                "Cannot compute forward dynamics: plant context not initialized"
            )
            return

        # Drake uses lazy evaluation, but we can force computation by accessing
        # derived quantities. This ensures all kinematic and dynamic quantities
        # are up-to-date
        try:
            # Force computation of mass matrix (triggers forward dynamics computation)
            _ = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)

            # Force computation of bias forces (ensures kinematics are updated)
            nv = self.plant.num_velocities()
            if nv > 0:
                vdot_zero = np.zeros(nv)
                _ = self.plant.CalcInverseDynamics(
                    self.plant_context,
                    vdot_zero,
                    self.plant.MakeMultibodyForces(self.plant),
                )

            LOGGER.debug("Drake forward dynamics computation completed")
        except Exception as e:
            LOGGER.error(f"Failed to compute forward dynamics: {e}")
            raise

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities)."""
        if not self.plant_context:
            return np.array([]), np.array([])

        q = self.plant.GetPositions(self.plant_context)
        v = self.plant.GetVelocities(self.plant_context)
        return q, v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state."""
        if not self.plant_context:
            return

        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces)."""
        if not self.plant_context:
            return

        # We need to set the actuation input port.
        # This requires fixing the input port to the plant in the diagram.
        # If we didn't export the input port, we can't set it easily.
        # In `__init__`, we should have exported the actuation input.

        # Assuming we can access the input port:
        # self.plant.get_actuation_input_port().FixValue(self.plant_context, u)
        # But input ports are "Fixed" in a Context.

        self.plant.get_actuation_input_port().FixValue(self.plant_context, u)

    def get_time(self) -> float:
        """Get the current simulation time."""
        if self.context:
            return cast(float, self.context.get_time())
        return 0.0

    # -------- Dynamics Interface --------

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q)."""
        if not self.plant_context:
            return np.array([])

        M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
        return cast(np.ndarray, M)

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q)."""
        if not self.plant_context:
            return np.array([])

        # C + g = InverseDynamics(q, v, 0)
        # Using CalcInverseDynamics(context, known_vdot, external_forces)
        # with known_vdot = 0.

        nv = self.plant.num_velocities()
        vdot_zero = np.zeros(nv)
        forces = self.plant.CalcInverseDynamics(
            self.plant_context, vdot_zero, self.plant.MakeMultibodyForces(self.plant)
        )
        return cast(np.ndarray, forces)

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        if not self.plant_context:
            return np.array([])

        # g(q) = GravityForces(context)
        return cast(
            np.ndarray, self.plant.CalcGravityGeneralizedForces(self.plant_context)
        )

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics tau = ID(q, v, a)."""
        if not self.plant_context:
            return np.array([])

        forces = self.plant.CalcInverseDynamics(
            self.plant_context, qacc, self.plant.MakeMultibodyForces(self.plant)
        )
        return cast(np.ndarray, forces)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body."""
        if not self.plant_context:
            return None

        # Find body frame
        try:
            body = self.plant.GetBodyByName(body_name)
            frame = body.body_frame()
        except Exception:
            return None

        # CalcJacobianSpatialVelocity
        # J_spatial (6 x nq)? Or (6 x nv). Drake supports nv.

        # Typically we want Jacobian w.r.t velocities (v).
        J = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            frame,
            np.zeros(3),  # Offset in B?
            self.plant.world_frame(),
            self.plant.world_frame(),  # Expressed in world?
        )
        J = cast(np.ndarray, J)

        # J is (6, nv). Top 3 angular, bottom 3 linear?
        # Drake SpatialVelocity is (w, v) -> Angular, Linear.

        jacr = J[:3, :]
        jacp = J[3:, :]

        return {"linear": jacp, "angular": jacr, "spatial": J}
