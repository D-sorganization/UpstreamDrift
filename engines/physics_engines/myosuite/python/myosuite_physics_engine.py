"""MyoSuite Physics Engine Implementation.

Wraps MyoSuite (OpenAI Gym-based) environments into the PhysicsEngine protocol.
Documentation: https://myosuite.readthedocs.io/en/latest/
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)

try:
    import gym
    import myosuite  # noqa: F401

    MYOSUITE_AVAILABLE = True
except ImportError:
    MYOSUITE_AVAILABLE = False
    LOGGER.warning("MyoSuite not installed. MyoSuitePhysicsEngine will not function.")


class MyoSuitePhysicsEngine(PhysicsEngine):
    """MyoSuite Engine Wrapper.

    Treats 'model paths' as Gym Environment IDs (e.g. 'myoElbowPose1D6MRandom-v0').
    Accesses underlying MuJoCo simulation for dynamics where possible.
    """

    def __init__(self) -> None:
        """Initialize."""
        self.env: gym.Env | None = None
        self.sim: Any = None  # Underlying mujoco simulation object
        self.env_id: str = ""
        self._dt = 0.002  # Default

    @property
    def model_name(self) -> str:
        return self.env_id if self.env_id else "MyoSuite_NoModel"

    def load_from_path(self, path: str) -> None:
        """Load environment by ID (passed as path)."""
        if not MYOSUITE_AVAILABLE:
            raise ImportError("MyoSuite not installed")

        # Heuristic: if path ends with .xml, user might be confused.
        # But for MyoSuite, we expect an Env ID.
        # We'll treat the 'path' argument as the Env ID.
        env_id = path.strip()

        try:
            self.env = gym.make(env_id)
            self.env_id = env_id
            self.env.reset()

            # Access underlying sim
            # specific to myosuite/mujoco-py structure
            if hasattr(self.env, "sim"):
                self.sim = self.env.sim
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "sim"):
                self.sim = self.env.unwrapped.sim
            else:
                LOGGER.warning(
                    "Could not access underlying MuJoCo sim object in MyoSuite env"
                )

            if self.sim:
                self._dt = self.sim.model.opt.timestep

        except Exception as e:
            LOGGER.error("Failed to load MyoSuite environment '%s': %s", env_id, e)
            raise

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Loading from string is not supported for Gym environments."""
        LOGGER.error(
            "MyoSuite does not support loading from string (requires Env ID registration)"
        )
        raise RuntimeError(
            "MyoSuite does not support loading from string (requires Env ID registration)"
        )

    def reset(self) -> None:
        """Reset environment."""
        if self.env:
            self.env.reset()

    def step(self, dt: float | None = None) -> None:
        """Step simulation."""
        if not self.env:
            return

        # Gym steps by fixed internal dt (usually frame_skip * model_dt).
        # We can't easily force arbitrary dt without hacking the sim.
        # We'll just call step(). Controls should be set beforehand.

        # We need the action.
        # In PhysicsEngine protocol, set_control() sets the action buffer?
        # But Gym step() TAKES the action.
        # This is a protocol mismatch.
        # Capability: If self.sim is present, we can just step self.sim directly?
        # But that bypasses MyoSuite's muscle activation dynamics (if implemented in python steps).
        # MyoSuite generally puts muscle dynamics in the MuJoCo model or usage of sim.step().

        # Strategy: Use sim.step() if available to respect 'PhysicsEngine' low-level vibe,
        # preserving state.
        if self.sim:
            # Handle dt override if possible
            if dt is not None:
                old_dt = self.sim.model.opt.timestep
                self.sim.model.opt.timestep = dt
                self.sim.step()
                self.sim.model.opt.timestep = old_dt
            else:
                self.sim.step()
        else:
            # Fallback to env.step() with zero action if we can't control it
            # (Bad, but fallback)
            zero_action = self.env.action_space.sample() * 0
            self.env.step(zero_action)

    def forward(self) -> None:
        """Compute forward dynamics."""
        if self.sim:
            self.sim.forward()

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get qpos, qvel."""
        if not self.sim:
            return np.array([]), np.array([])
        return (np.array(self.sim.data.qpos[:]), np.array(self.sim.data.qvel[:]))

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        if not self.sim:
            return

        # MjSim (mujoco-py) usually allows direct assignment
        if len(q) == len(self.sim.data.qpos):
            self.sim.data.qpos[:] = q
        if len(v) == len(self.sim.data.qvel):
            self.sim.data.qvel[:] = v

        self.sim.forward()

    def set_control(self, u: np.ndarray) -> None:
        """Set control (ctrl)."""
        if not self.sim:
            return

        if len(u) == self.sim.data.ctrl.shape[0]:
            self.sim.data.ctrl[:] = u

    def get_time(self) -> float:
        if self.sim:
            return float(self.sim.data.time)
        return 0.0

    def compute_mass_matrix(self) -> np.ndarray:
        if not self.sim:
            return np.array([])

        # Try generic mujoco approach assuming 'mujoco' lib is available
        try:
            import mujoco

            nv = self.sim.model.nv
            M = np.zeros((nv, nv))
            mujoco.mj_fullM(self.sim.model, M, self.sim.data.qM)
            return M
        except Exception as e:
            LOGGER.error("Failed to compute mass matrix: %s", e)
            return np.array([])

    def compute_bias_forces(self) -> np.ndarray:
        if self.sim:
            return np.array(self.sim.data.qfrc_bias)
        return np.array([])

    def compute_gravity_forces(self) -> np.ndarray:
        # Not easily exposed separately in basic bindings without extra calc
        return np.array([])

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        # Requires calling mj_inverse
        if not self.sim:
            return np.array([])

        try:
            import mujoco

            self.sim.data.qacc[:] = qacc
            mujoco.mj_inverse(self.sim.model, self.sim.data)
            return np.array(self.sim.data.qfrc_inverse)
        except Exception as e:
            LOGGER.error("Failed to compute inverse dynamics: %s", e)
            return np.array([])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        if not self.sim:
            return None

        try:
            import mujoco

            body_id = mujoco.mj_name2id(
                self.sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id == -1:
                return None

            jacp = np.zeros((3, self.sim.model.nv))
            jacr = np.zeros((3, self.sim.model.nv))
            mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, body_id)

            return {"linear": jacp, "angular": jacr, "spatial": np.vstack([jacr, jacp])}
        except Exception as e:
            LOGGER.error("Failed to compute Jacobian for body '%s': %s", body_name, e)
            return None
