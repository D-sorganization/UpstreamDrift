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

from shared.python import constants
from shared.python.interfaces import PhysicsEngine
from shared.python.inertia_ellipse import BodyInertiaData

LOGGER = logging.getLogger(__name__)

DEFAULT_TIME_STEP = float(constants.DEFAULT_TIME_STEP)


class DrakePhysicsEngine(PhysicsEngine):
    """Encapsulates Drake MultibodyPlant and simulation control.

    Implements the shared PhysicsEngine protocol.
    """

    def __init__(self, time_step: float = DEFAULT_TIME_STEP) -> None:
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
        else:
            LOGGER.warning("Attempted to reset Drake engine before initialization.")

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
            LOGGER.error("Failed to compute forward dynamics: %s", e)
            raise

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities)."""
        if not self.plant_context:
            LOGGER.debug("get_state called on uninitialized engine")
            return np.array([]), np.array([])

        q = self.plant.GetPositions(self.plant_context)
        v = self.plant.GetVelocities(self.plant_context)
        return q, v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state."""
        if not self.plant_context:
            LOGGER.warning("set_state called on uninitialized engine")
            return

        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces)."""
        if not self.plant_context:
            LOGGER.warning("set_control called on uninitialized engine")
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

    def get_joint_names(self) -> list[str]:
        """Get list of joint names."""
        if not self.plant:
            return []

        # Drake has actuators.
        names = []
        for i in range(self.plant.num_actuators()):
            actuator_idx = pydrake.multibody.tree.JointActuatorIndex(i)
            act = self.plant.get_joint_actuator(actuator_idx)
            names.append(act.name())

        if not names:
            # If there are no actuators defined, fall back to generic names
            # derived from the number of generalized velocities (dofs).
            for i in range(self.plant.num_velocities()):
                names.append(f"dof_{i}")

        return names

    def get_full_state(self) -> dict[str, Any]:
        """Get complete state in a single batched call (performance optimization).

        PERFORMANCE FIX: Returns all commonly-needed state in one call to avoid
        multiple separate engine queries.

        Returns:
            Dictionary with 'q', 'v', 't', and 'M' (mass matrix).
        """
        if not self.plant_context:
            return {"q": np.array([]), "v": np.array([]), "t": 0.0, "M": None}

        # Get state
        q = self.plant.GetPositions(self.plant_context)
        v = self.plant.GetVelocities(self.plant_context)
        t = float(self.context.get_time()) if self.context else 0.0

        # Compute mass matrix
        # Note: CalcMassMatrixViaInverseDynamics is efficient in Drake
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)

        return {"q": q, "v": v, "t": t, "M": M}

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

    def compute_contact_forces(self) -> np.ndarray:
        """Compute total contact forces (ground reaction force, GRF).

        Notes:
            This Drake wrapper currently returns a placeholder zero vector for the
            GRF. Retrieving precise contact forces in Drake requires querying
            :class:`ContactResults` from the simulation :class:`Context`, which is
            typically produced and managed by a :class:`Simulator` and is not
            readily accessible through this lightweight wrapper interface.

            If you need accurate GRFs, integrate directly with Drake's
            ``MultibodyPlant`` and ``Simulator`` APIs and accumulate forces from
            the ``ContactResults`` output.

        Returns:
            f: (3,) vector representing total ground reaction force (currently
                always zeros as a placeholder).
        """
        if not self.plant_context:
            return np.zeros(3)

        # In Drake, contact forces are typically accessed via GetContactResults.
        # This requires the context to have been updated with contact results.
        #
        # NOTE: This implementation assumes CalcContactResults has been called
        # by simulation or we force it here if possible.
        # But contact results are usually output of Simulator.
        #
        # For simplicity in this wrapper, we try to access generalized contact forces
        # or return zero if not easily accessible without full simulation integration.
        #
        # Placeholder: Retrieving precise GRF in Drake requires querying ContactResults
        # from the Context, summing up forces on 'ground' bodies.
        #
        # As a simplified proxy, we can inspect generalized contact forces if available
        # but that's in joint space.

        LOGGER.warning(
            "DrakePhysicsEngine.compute_contact_forces currently returns a "
            "placeholder zero GRF vector. Precise contact forces require querying "
            "ContactResults from a Simulator-managed Context, which is not exposed "
            "through this wrapper. For accurate GRFs, use Drake's MultibodyPlant/"
            "Simulator APIs directly."
        )

        return np.zeros(3)

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

        return {
            "linear": jacp,
            "angular": jacr,
            "spatial": J,  # Standard: Angular (0-3), Linear (3-6)
        }

    # -------- Section F: Drift-Control Decomposition --------

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control inputs.

        Section F Implementation: Uses Drake's CalcInverseDynamics with zero
        applied forces and then solves for acceleration via M^-1 * (bias forces).

        Returns:
            q_ddot_drift: Drift acceleration vector (nv,) [rad/s² or m/s²]
        """
        if not self.plant_context:
            return np.array([])

        # Get mass matrix
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)

        # Get bias forces (C(q,v)v + g(q))
        nv = self.plant.num_velocities()
        vdot_zero = np.zeros(nv)
        bias = self.plant.CalcInverseDynamics(
            self.plant_context, vdot_zero, self.plant.MakeMultibodyForces(self.plant)
        )

        # Drift = -M^-1 * bias
        # (Since bias = M*0 + C + g, we have C + g = bias, and drift acc = -M^-1*(C+g))
        # Actually: tau = M*a + C + g, so if tau=0: 0 = M*a + C + g → a = -M^-1*(C+g)
        a_drift = -np.linalg.solve(M, bias)

        return cast(np.ndarray, a_drift)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration from applied torques only.

        Section F Implementation: Computes M(q)^-1 * tau to isolate control component.

        Args:
            tau: Applied generalized forces (nv,) [N·m or N]

        Returns:
            q_ddot_control: Control acceleration vector (nv,) [rad/s² or m/s²]
        """
        if not self.plant_context:
            return np.array([])

        # Get mass matrix
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)

        # Control component: M^-1 * tau
        a_control = np.linalg.solve(M, tau)

        return cast(np.ndarray, a_control)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero, preserving state.
        This isolates drift (gravity + Coriolis) from control effects.

        **Purpose**: Answer "What would happen if all actuators turned off?"

        **Physics**: With τ=0, acceleration is purely passive:
            q̈_ZTCF = M(q)⁻¹ · (-C(q,v) - g(q))

        Args:
            q: Joint positions (n_q,) [rad or m]
            v: Joint velocities (n_v,) [rad/s or m/s]

        Returns:
            q̈_ZTCF: Acceleration under zero applied torque (n_v,) [rad/s² or m/s²]
        """
        if not self.plant_context:
            return np.array([])

        # Save current state
        saved_q = self.plant.GetPositions(self.plant_context)
        saved_v = self.plant.GetVelocities(self.plant_context)

        try:
            # Set to counterfactual state
            self.plant.SetPositions(self.plant_context, q)
            self.plant.SetVelocities(self.plant_context, v)

            # Compute mass matrix at counterfactual state
            M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)

            # Compute bias forces at counterfactual state (C(q,v) + g(q))
            nv = self.plant.num_velocities()
            vdot_zero = np.zeros(nv)
            bias = self.plant.CalcInverseDynamics(
                self.plant_context,
                vdot_zero,
                self.plant.MakeMultibodyForces(self.plant),
            )

            # ZTCF: τ = 0, so M*a + bias = 0 → a = -M^-1 * bias
            a_ztcf = -np.linalg.solve(M, bias)

            return cast(np.ndarray, a_ztcf)

        finally:
            # Restore original state
            self.plant.SetPositions(self.plant_context, saved_q)
            self.plant.SetVelocities(self.plant_context, saved_v)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero, preserving
        configuration. This isolates configuration-dependent effects (gravity)
        from velocity-dependent effects (Coriolis, centrifugal).

        **Purpose**: Answer "What acceleration would occur if motion FROZE?"

        **Physics**: With v=0, acceleration has no velocity-dependent terms:
            q̈_ZVCF = M(q)⁻¹ · (-g(q) + τ)

        Args:
            q: Joint positions (n_q,) [rad or m]

        Returns:
            q̈_ZVCF: Acceleration with v=0 (n_v,) [rad/s² or m/s²]
        """
        if not self.plant_context:
            return np.array([])

        # Save current state
        saved_q = self.plant.GetPositions(self.plant_context)
        saved_v = self.plant.GetVelocities(self.plant_context)

        try:
            # Set to counterfactual configuration with v=0
            self.plant.SetPositions(self.plant_context, q)
            self.plant.SetVelocities(self.plant_context, np.zeros_like(saved_v))

            # Compute mass matrix at counterfactual configuration
            M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)

            # With v=0, bias = g(q) only (no Coriolis terms)
            # Use gravity forces directly
            g = self.plant.CalcGravityGeneralizedForces(self.plant_context)

            # Get current control (preserved for ZVCF)
            # Note: In Drake, we need to read from actuator input or assume zero
            # For simplicity, assume current actuation is zero unless set
            # In a full implementation, we would read from the actuation input port
            tau = np.zeros(self.plant.num_velocities())

            # ZVCF: M*a + g = τ → a = M^-1 * (τ - g)
            # Note: g is the gravity force vector, not gravity generalized force
            # CalcGravityGeneralizedForces returns -g in the equation M*a + c + g = τ
            a_zvcf = np.linalg.solve(M, tau - g)

            return cast(np.ndarray, a_zvcf)

        finally:
            # Restore original state
            self.plant.SetPositions(self.plant_context, saved_q)
            self.plant.SetVelocities(self.plant_context, saved_v)

    # -------- Inertia Ellipse Visualization Support --------

    def get_body_names(self) -> list[str]:
        """Get list of all body names in the model.

        Returns:
            List of body name strings
        """
        if not self.plant:
            return []

        names = []
        # Iterate over all model instances and collect body names
        for model_idx in range(self.plant.num_model_instances()):
            try:
                model_instance = pydrake.multibody.tree.ModelInstanceIndex(model_idx)
                for body_idx in self.plant.GetBodyIndices(model_instance):
                    body = self.plant.get_body(body_idx)
                    name = body.name()
                    if name and name not in names and name != "world":
                        names.append(name)
            except Exception:
                pass

        return names

    def get_body_inertia_data(self, body_name: str) -> BodyInertiaData | None:
        """Get inertia data for a specific body.

        Args:
            body_name: Name of the body

        Returns:
            BodyInertiaData for the body, or None if not found
        """
        if not self.plant_context:
            return None

        try:
            body = self.plant.GetBodyByName(body_name)
        except Exception:
            return None

        # Get spatial inertia
        spatial_inertia = body.default_spatial_inertia()

        # Extract mass
        mass = float(spatial_inertia.get_mass())

        # Get center of mass in body frame
        com_body = np.array(spatial_inertia.get_com())

        # Get rotational inertia about COM in body frame
        # UnitInertia is I/m, so multiply by mass to get actual inertia
        unit_inertia = spatial_inertia.get_unit_inertia()
        rotational_inertia = unit_inertia.CopyToFullMatrix3()
        inertia_local = mass * np.array(rotational_inertia)

        # Get body pose in world frame
        body_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
        rotation = np.array(body_pose.rotation().matrix())
        position = np.array(body_pose.translation())

        # Transform COM to world frame
        com_world = position + rotation @ com_body

        return BodyInertiaData(
            name=body_name,
            mass=mass,
            com_world=com_world,
            inertia_local=inertia_local,
            rotation=rotation,
        )

    def get_all_body_inertia_data(self) -> list[BodyInertiaData]:
        """Get inertia data for all bodies in the model.

        Returns:
            List of BodyInertiaData for all bodies
        """
        body_names = self.get_body_names()
        result = []
        for name in body_names:
            data = self.get_body_inertia_data(name)
            if data is not None:
                result.append(data)
        return result
