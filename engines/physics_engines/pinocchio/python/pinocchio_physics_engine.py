"""Pinocchio Physics Engine wrapper implementation.

Wraps pinocchio to provide a compliant PhysicsEngine interface.
"""

from __future__ import annotations

import logging

import numpy as np

# Pinocchio imports likely to fail if not installed
try:
    import pinocchio as pin
except ImportError:
    pass

from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)


class PinocchioPhysicsEngine(PhysicsEngine):
    """Encapsulates Pinocchio model, data, and simulation control.

    Implements the shared PhysicsEngine protocol.
    """

    def __init__(self) -> None:
        """Initialize the Pinocchio physics engine."""
        self.model: pin.Model | None = None
        self.data: pin.Data | None = None
        self.model_path: str = ""
        self.model_name_str: str = ""

        # State
        self.q: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.a: np.ndarray = np.array([])
        self.tau: np.ndarray = np.array([])
        self.time: float = 0.0

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        if self.model:
            return self.model.name
        return self.model_name_str

    def load_from_path(self, path: str) -> None:
        """Load model from file path (URDF)."""
        # Pinocchio typically loads URDFs
        if not path.endswith(".urdf"):
            LOGGER.warning("Pinocchio loader expects URDF, got: %s", path)

        try:
            self.model = pin.buildModelFromUrdf(path)
            self.data = self.model.createData()
            self.model_path = path
            self.model_name_str = self.model.name

            # Initialize state
            self.q = pin.neutral(self.model)
            self.v = np.zeros(self.model.nv)
            self.a = np.zeros(self.model.nv)
            self.tau = np.zeros(self.model.nv)
            self.time = 0.0

        except Exception as e:
            LOGGER.error("Failed to load Pinocchio model from path %s: %s", path, e)
            raise

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from string content."""
        if extension != "urdf":
            LOGGER.warning("Pinocchio load_from_string mostly supports URDF.")

        try:
            self.model = pin.buildModelFromXML(content)
            self.data = self.model.createData()
            self.model_name_str = "StringLoadedModel"

            self.q = pin.neutral(self.model)
            self.v = np.zeros(self.model.nv)
            self.a = np.zeros(self.model.nv)
            self.tau = np.zeros(self.model.nv)
            self.time = 0.0

        except Exception as e:
            LOGGER.error("Failed to load Pinocchio model from string: %s", e)
            raise

    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        if self.model:
            self.q = pin.neutral(self.model)
            self.v = np.zeros(self.model.nv)
            self.a = np.zeros(self.model.nv)
            self.tau = np.zeros(self.model.nv)
            self.time = 0.0
            # Refresh data
            self.forward()

    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one time step."""
        if self.model is None or self.data is None:
            return

        time_step = dt if dt is not None else 0.001

        # Explicit Forward Dynamics
        # a = ABA(q, v, tau)
        self.a = pin.aba(self.model, self.data, self.q, self.v, self.tau)

        # Semi-implicit Euler integration? Or symplectic Euler?
        # v_next = v + a * dt
        # q_next = integrate(q, v_next * dt)

        self.v += self.a * time_step
        self.q = pin.integrate(self.model, self.q, self.v * time_step)

        self.time += time_step

    def forward(self) -> None:
        """Compute forward kinematics/dynamics without advancing time."""
        if self.model is None or self.data is None:
            return

        pin.forwardKinematics(self.model, self.data, self.q, self.v, self.a)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities)."""
        return self.q.copy(), self.v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state."""
        if self.model is None:
            return

        if len(q) == self.model.nq:
            self.q = q.copy()
        if len(v) == self.model.nv:
            self.v = v.copy()

    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces)."""
        if self.model is None:
            return
        if len(u) == self.model.nv:
            self.tau = u.copy()

    def get_time(self) -> float:
        """Get the current simulation time."""
        return self.time

    # -------- Dynamics Interface --------

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        # CRBA computes the upper triangular part of the joint space inertia
        # matrix and stores it in data.M
        pin.crba(self.model, self.data, self.q)

        # data.M is symmetric, but pinocchio implementation specifics might
        # require filling? Typically pinocchio fills the upper triangle.

        M = self.data.M.copy()
        # Fill lower triangle if needed (Pinocchio's M is usually symmetric
        # dense if using modern bindings, but let's ensure it maps correctly
        # to numpy) But specifically crba fills the upper triangle.

        # For safety/consistency:
        M = np.triu(M) + np.triu(M, 1).T
        return M

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        # rnea(q, v, 0) -> M a + b
        # If a=0, result is b = C(q,v) + g(q)

        a0 = np.zeros(self.model.nv)
        b = pin.rnea(self.model, self.data, self.q, self.v, a0)
        return b

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        # computeGeneralizedGravity(model, data, q)
        return pin.computeGeneralizedGravity(self.model, self.data, self.q)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics tau = ID(q, v, a)."""
        if self.model is None or self.data is None:
            return np.array([])

        tau = pin.rnea(self.model, self.data, self.q, self.v, qacc)
        return tau

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body."""
        if self.model is None or self.data is None:
            return None

        if not self.model.existBodyName(body_name):
            # Pinocchio bodies vs frames...
            # existBodyName checks for link/joint?
            # maybe existFrame?
            if not self.model.existFrame(body_name):
                return None
            frame_id = self.model.getFrameId(body_name)
        else:
            # It's a joint name or body name? Pinocchio organizes by joints.
            # body name usually maps to frame.
            frame_id = self.model.getFrameId(body_name)

        # computeJointJacobians needs to be called first (done in forward)
        # getFrameJacobian

        J = pin.getFrameJacobian(
            self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # J is (6, nv).
        # Top 3 linear, bottom 3 angular?
        # Pinocchio Motion ordering is Linear, Angular.

        jacp = J[:3, :]
        jacr = J[3:, :]

        return {"linear": jacp, "angular": jacr, "spatial": J}
