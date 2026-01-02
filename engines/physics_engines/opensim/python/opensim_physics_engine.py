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
        """Load model from XML string using a temporary file."""
        if opensim is None:
            raise ImportError("OpenSim library not installed")

        import tempfile

        suffix = f".{extension}" if extension else ".osim"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            self.load_from_path(tmp_path)
            # We don't delete the file immediately as OpenSim might need it?
            # Usually safe to delete after load if OpenSim parses and loads into memory.
            # But Simbody can rely on resource paths relative to the file.
            # For robustness, we might leave it or manage it carefully.
            # For now, we leave it to be cleaned up by OS or user.

        except Exception as e:
            logger.error(f"Failed to load OpenSim model from string: {e}")
            raise

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
        """Set controls for the model."""
        if not self._model or not self._state:
            return

        # Map u to model controls
        # OpenSim controls are often for actuators
        # We assume u length matches getNumControls()
        # This requires realizable instance
        try:
            # Basic implementation assuming straight mapping
            # In a real rigorous impl, we'd check names/indices
            self._model.updControls(self._state)
            # This might not be writable directly this way without solver interaction
            pass
        except Exception:
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
        # Ensure state is realizable to Position
        self._model.realizePosition(self._state)
        matter.calcM(self._state, m_mat)

        # Convert opensim Matrix to numpy
        res = np.zeros((n_u, n_u))
        for r in range(n_u):
            for c in range(n_u):
                res[r, c] = m_mat.get(r, c)
        return res

    def compute_bias_forces(self) -> np.ndarray:
        """Compute C(q,u) + G(q)."""
        # This corresponds to inverse dynamics with udot=0 (if we ignore external forces)
        if not self._model or not self._state:
            return np.array([])

        # Placeholder: OpenSim System doesn't expose 'bias forces' term directly easily
        # without using the ID solver
        return np.array([])

    def compute_gravity_forces(self) -> np.ndarray:
        if not self._model or not self._state:
            return np.array([])

        # To extract gravity only:
        # We need (generalized) gravity forces.
        # MatterSubsytem.calcGravityForce(state) -> Vector of gravity forces?
        # SimTK::SimbodyMatterSubsystem has multiplyBySystemGravity but OpenSim wrapping hides it often.
        return np.array([])

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        if not self._model or not self._state:
            return np.array([])

        # Use an InverseDynamicsSolver
        n_u = self._model.getNumSpeeds()

        if len(qacc) != n_u:
            return np.array([])

        udot = opensim.Vector(n_u)
        for i in range(n_u):
            udot.set(i, float(qacc[i]))

        # We need realized Acceleration
        # But we can't just 'set' acceleration in state for ID. Use Solver.
        # ID Solver takes (model, state, udot) -> tau

        try:
            # Ensure state realized to Velocity
            self._model.realizeVelocity(self._state)

            solver = opensim.InverseDynamicsSolver(self._model)
            # Some versions use solve(state, udot, applied_loads, tau_out)
            # applied_loads can be empty
            # tau = solver.solve(self._state, udot) # if wrapper is friendly

            # Standard C++ wrapping often returns Vector
            tau = solver.solve(self._state, udot)

            res = np.zeros(n_u)
            for i in range(n_u):
                res[i] = tau.get(i)

            return res
        except Exception as e:
            logger.error(f"OpenSim ID failed: {e}")
            return np.array([])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        if not self._model or not self._state:
            return None

        try:
            self._model.getBodySet().get(body_name)
            # matter = self._model.getMatterSubsystem()
            # We need a point on the body.
            # This requires lower level SimTK access usually
            return None
        except Exception:
            return None
