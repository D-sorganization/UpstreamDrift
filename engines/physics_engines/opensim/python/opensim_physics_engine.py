"""OpenSim Physics Engine implementation."""

from __future__ import annotations

import logging
import os

import numpy as np

from shared.python.interfaces import PhysicsEngine

# Configure logging
logger = logging.getLogger(__name__)

try:
    import opensim
except ImportError:
    opensim = None
    logger.warning(
        "OpenSim python package not found. OpenSimPhysicsEngine will not function fully."
    )


class OpenSimPhysicsEngine(PhysicsEngine):
    """OpenSim Physics Engine Implementation."""

    def __init__(self) -> None:
        self._model = None
        self._state = None
        self._manager = None
        self._model_path = ""
        self._time_step = 0.01

        if opensim is None:
            logger.error("OpenSim library is not installed.")

    @property
    def model_name(self) -> str:
        if self._model:
            return self._model.getName()
        return "OpenSim_NoModel"

    def load_from_path(self, path: str) -> None:
        if opensim is None:
            raise ImportError("OpenSim library not installed")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            self._model = opensim.Model(path)
            self._model_path = path
            if self._model is None:
                raise ValueError("Failed to create OpenSim Model object")
            # Initialize the system and state
            self._state = self._model.initSystem()
            self._manager = opensim.Manager(self._model)
            logger.info(f"Loaded OpenSim model from {path}")
        except Exception as e:
            logger.error(f"Failed to load OpenSim model: {e}")
            raise

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        # OpenSim serialization usually requires a file. We can write to a temp file.
        raise NotImplementedError("OpenSim load_from_string not yet implemented")

    def reset(self) -> None:
        if self._model and self._state:
            # Re-initialize the system to defaults
            self._state = self._model.initializeState()
            self._model.equilibrateMuscles(self._state)
            self._manager.setSessionTime(0.0)
            self._manager.setIntegrator(opensim.RungeKuttaMersonIntegrator(self._model))

    def step(self, dt: float | None = None) -> None:
        if not self._model or not self._state:
            return

        step_size = dt if dt is not None else self._time_step
        current_time = self._state.getTime()

        # Integrate to new time
        self._manager.setInitialTime(current_time)
        self._manager.setFinalTime(current_time + step_size)

        # Integrate
        self._manager.integrate(current_time + step_size)

        # Update our internal state reference?
        # The manager updates the state in the model, but let's be sure.
        # Actually opensim Manager behavior varies by version.
        # Usually manager.integrate(t_final) updates the state.

    def forward(self) -> None:
        if self._model and self._state:
            self._model.realizeDynamics(self._state)

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._model or not self._state:
            return np.array([]), np.array([])

        # Q (Coordinates)
        n_q = self._model.getNumCoordinates()
        q_vec = self._state.getQ()
        q = np.array([q_vec.get(i) for i in range(n_q)])

        # U (Speeds)
        n_u = self._model.getNumSpeeds()
        u_vec = self._state.getU()
        v = np.array([u_vec.get(i) for i in range(n_u)])

        return q, v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        if not self._model or not self._state:
            return

        # Set Q
        n_q = self._model.getNumCoordinates()
        if len(q) == n_q:
            q_vec = opensim.Vector(n_q)
            for i in range(n_q):
                q_vec.set(i, float(q[i]))
            self._state.setQ(q_vec)

        # Set U
        n_u = self._model.getNumSpeeds()
        if len(v) == n_u:
            u_vec = opensim.Vector(n_u)
            for i in range(n_u):
                u_vec.set(i, float(v[i]))
            self._state.setU(u_vec)

        self._model.realizeVelocity(self._state)

    def set_control(self, u: np.ndarray) -> None:
        if not self._model:
            return
        # This is complex in OpenSim (Controls vs Actuators).
        # Assuming u maps to actuators in order.
        # This might need refinement.
        pass

    def get_time(self) -> float:
        if self._state:
            return self._state.getTime()
        return 0.0

    def compute_mass_matrix(self) -> np.ndarray:
        if not self._model or not self._state:
            return np.array([])

        matter = self._model.getMatterSubsystem()
        n_u = self._model.getNumSpeeds()
        m_mat = opensim.Matrix()
        matter.calcM(self._state, m_mat)

        # Convert opensim Matrix to numpy
        res = np.zeros((n_u, n_u))
        for r in range(n_u):
            for c in range(n_u):
                res[r, c] = m_mat.get(r, c)
        return res

    def compute_bias_forces(self) -> np.ndarray:
        # This is not direct in OpenSim High-level API.
        # Would need to use InverseDynamicsSolver with zero Acc?
        # Or Simbody directly.
        return np.array([])

    def compute_gravity_forces(self) -> np.ndarray:
        if not self._model or not self._state:
            return np.array([])

        # matter = self._model.getMatterSubsystem()
        # g_force = opensim.Vector()
        # This might not be exposed directly on MatterSubsystem in all versions.
        # Fallback to ID with zero qacc and zero vel?
        return np.array([])

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        if not self._model or not self._state:
            return np.array([])

        # InverseDynamicsSolver
        # Note: OpenSim ID usually solves for Gen Forces, not actuator torques directly in the same way.
        return np.array([])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        if not self._model or not self._state:
            return None

        # Access body by name
        try:
            self._model.getBodySet().get(body_name)
        except Exception:
            # Fallback if accessed by index or other means fail
            try:
                self._model.getBodySet().get(body_name)
            except Exception:
                raise ValueError(f"Body {body_name} not found in model") from None

        # matter = self._model.getMatterSubsystem()
        # matter.calcStationJacobian(...)
        return None
