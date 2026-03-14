"""Golf Swing Pendulum Physics Engine.

This adapter exposes the ``double_pendulum_golf`` simulator from the shared
Tools repository (``vendor/ud-tools``) as a first-class UpstreamDrift
PhysicsEngine.

The Tools model uses **relative coordinates with point-mass clubhead**,
Coulomb+viscous damping, joint-limit barriers, and a polynomial torque API.
It is a lightweight counterpart to UpstreamDrift's own
``DoublePendulumDynamics`` (distributed inertia, simpleeval expressions).

Both models coexist and can be compared; this engine routes users to the
Tools model through the same ``PhysicsEngine`` protocol so they behave
identically from the orchestration layer's perspective.

Architecture (DRY)
------------------
Inherits ``BasePhysicsEngine`` for:
  - checkpoint save / restore
  - model-name tracking
  - string representation

Design by Contract
------------------
  - Precondition:  engine must be initialised before calling dynamics methods.
  - Postcondition: all returned arrays contain finite values.

Integration Pattern (how shared tools are incorporated)
-------------------------------------------------------
1. The Tools repository is vendored as the ``vendor/ud-tools`` git submodule.
2. ``pyproject.toml`` adds ``vendor/ud-tools/src/shared/python`` to pytest
   ``pythonpath`` so shared utilities are importable without installation.
3. The pendulum simulator package lives at
   ``vendor/ud-tools/src/pendulum_simulator/src``; we import it lazily inside
   this module to avoid forcing a PyQt6 import at module load.
4. The nightly CI job ``check-vendor-freshness`` ensures the submodule pointer
   stays current with the Tools ``main`` branch (see
   ``.github/workflows/vendor-freshness.yml``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine
from src.shared.python.engine_core.checkpoint import StateCheckpoint
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------

_TOOLS_PENDULUM_AVAILABLE: bool | None = None
_TOOLS_PACKAGE_ROOT = (
    Path(__file__).parent.parent.parent.parent.parent.parent.parent
    / "vendor"
    / "ud-tools"
    / "src"
    / "pendulum_simulator"
    / "src"
)


def _check_tools_pendulum() -> bool:
    """Return True if the Tools pendulum package is importable."""
    global _TOOLS_PENDULUM_AVAILABLE  # noqa: PLW0603
    if _TOOLS_PENDULUM_AVAILABLE is not None:
        return _TOOLS_PENDULUM_AVAILABLE
    try:
        import sys

        tools_path = str(_TOOLS_PACKAGE_ROOT)
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        import double_pendulum_golf  # noqa: F401

        _TOOLS_PENDULUM_AVAILABLE = True
    except ImportError:
        _TOOLS_PENDULUM_AVAILABLE = False
        logger.warning(
            "Tools pendulum package not available. "
            "Ensure vendor/ud-tools submodule is initialised: "
            "git submodule update --init vendor/ud-tools"
        )
    return _TOOLS_PENDULUM_AVAILABLE  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Default physical parameters (golf-swing presets from Tools README)
# ---------------------------------------------------------------------------

_DEFAULT_GOLF_PARAMS: dict[str, float] = {
    "m1": 5.0,  # arm mass (kg)
    "m2": 0.30,  # shaft mass (kg)
    "L1": 0.65,  # arm length (m)
    "L2": 1.10,  # shaft length (m)
    "mClub": 0.20,  # clubhead mass (kg)
    "g": 9.81,  # gravity (m/s²)
    "b1": 0.4,  # shoulder damping (N·m·s/rad)
    "b2": 0.05,  # wrist damping (N·m·s/rad)
    "mu1": 0.0,  # Coulomb friction shoulder
    "mu2": 0.0,  # Coulomb friction wrist
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GolfSwingPendulumEngine(BasePhysicsEngine):
    """Physics engine wrapping the Tools ``double_pendulum_golf`` model.

    This provides a **simpler** relative-coordinate Lagrangian model with
    an explicit clubhead point mass, compared to the primary
    ``PendulumPhysicsEngine`` which uses distributed inertia.

    Use this engine when you need:
    - Explicit clubhead dynamics (impact analysis)
    - Coulomb friction at the wrist joint
    - Polynomial torque profiles (classical golf-swing coaching presets)
    - Direct comparison with the Tools GUI simulator

    State vector convention (matches Tools):
        [theta1, phi, dtheta1, dphi]
        theta1 : absolute angle of arm segment from downward vertical (rad)
        phi    : angle of club segment RELATIVE to arm (rad)
    """

    ENGINE_NAME = "GolfSwingPendulum"

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise the golf-swing pendulum engine.

        Parameters
        ----------
        params:
            Override any entry in the default golf-swing physical parameters.
            Keys match ``PendulumParams`` field names (m1, m2, L1, L2, mClub,
            g, b1, b2, mu1, mu2).
        """
        super().__init__()
        self._params_override = params or {}
        self._state: np.ndarray = np.zeros(4)  # [theta1, phi, dtheta1, dphi]
        self.time: float = 0.0
        self._tau: np.ndarray = np.zeros(2)  # [tau_shoulder, tau_wrist]

        # Lazy initialisation — avoid PyQt6 import at load time
        self._dynamics: Any = None
        self._pendulum_params: Any = None

        self.model_name_str = self.ENGINE_NAME
        self._is_initialized = False

        if _check_tools_pendulum():
            self._lazy_init()

    # ------------------------------------------------------------------
    # BasePhysicsEngine hooks
    # ------------------------------------------------------------------

    @property
    def engine_type(self) -> str:
        return "golf_swing_pendulum"

    def _load_from_path_impl(self, path: str) -> None:
        """No-op: golf-swing pendulum is a parametric model."""

    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        """No-op: golf-swing pendulum is a parametric model."""

    def load_from_path(self, path: str) -> None:
        """Golf-swing pendulum is parametric; path is ignored."""
        logger.debug(
            "%s is a parametric model. Path '%s' is ignored.",
            self.ENGINE_NAME,
            path,
        )

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Golf-swing pendulum is parametric; content is ignored."""
        logger.debug("%s ignores load_from_string.", self.ENGINE_NAME)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        """Import Tools package and build model objects."""
        import sys  # noqa: PLC0415

        tools_path = str(_TOOLS_PACKAGE_ROOT)
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)

        from double_pendulum_golf.physics import PendulumParams  # noqa: PLC0415

        merged = {**_DEFAULT_GOLF_PARAMS, **self._params_override}
        self._pendulum_params = PendulumParams(**merged)

        # Default: shoulder-driven swing (passive wrist)
        self._tau = np.zeros(2)
        self._is_initialized = True
        logger.info("%s initialised with params: %s", self.ENGINE_NAME, merged)

    # ------------------------------------------------------------------
    # Core simulation interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to the initial configuration (hanging at rest)."""
        self._state = np.zeros(4)
        self.time = 0.0
        self._tau = np.zeros(2)

    def step(self, dt: float | None = None) -> None:
        """Advance state by one RK4 step.

        Precondition: engine must be initialised.
        """
        if not self._is_initialized:
            logger.warning("GolfSwingPendulumEngine: step called before init.")
            return

        step_size = dt if dt is not None else 0.01

        from double_pendulum_golf.physics import equations_of_motion  # noqa: PLC0415

        def torque_func(t: float) -> tuple[float, float]:  # noqa: ARG001
            return float(self._tau[0]), float(self._tau[1])

        deriv = equations_of_motion(
            self._state, self.time, self._pendulum_params, torque_func
        )

        # RK4 integration
        k1 = deriv
        k2 = equations_of_motion(
            self._state + 0.5 * step_size * k1,
            self.time + 0.5 * step_size,
            self._pendulum_params,
            torque_func,
        )
        k3 = equations_of_motion(
            self._state + 0.5 * step_size * k2,
            self.time + 0.5 * step_size,
            self._pendulum_params,
            torque_func,
        )
        k4 = equations_of_motion(
            self._state + step_size * k3,
            self.time + step_size,
            self._pendulum_params,
            torque_func,
        )

        self._state = self._state + (step_size / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.time += step_size

    def forward(self) -> None:
        """Compute forward kinematics (no time advance)."""

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities).

        positions : [theta1, phi]   (rad)
        velocities: [dtheta1, dphi] (rad/s)
        """
        return self._state[:2].copy(), self._state[2:].copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set joint positions and velocities."""
        if len(q) >= 2 and len(v) >= 2:
            self._state = np.array([float(q[0]), float(q[1]), float(v[0]), float(v[1])])

    def set_control(self, u: np.ndarray) -> None:
        """Set applied torques [tau_shoulder, tau_wrist] (N·m)."""
        if len(u) >= 2:
            self._tau = np.array([float(u[0]), float(u[1])])

    def get_time(self) -> float:
        """Return current simulation time (s)."""
        return self.time

    # ------------------------------------------------------------------
    # Dynamics interface
    # ------------------------------------------------------------------

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the 2×2 mass matrix M(q).

        Postcondition: returned matrix is symmetric positive-definite.
        """
        if not self._is_initialized:
            return np.eye(2)

        from double_pendulum_golf.physics import mass_matrix  # noqa: PLC0415

        phi = float(self._state[1])
        return mass_matrix(phi, self._pendulum_params)

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces = Coriolis + gravity + damping + friction.

        Postcondition: returned vector is finite.
        """
        if not self._is_initialized:
            return np.zeros(2)

        from double_pendulum_golf.physics import (  # noqa: PLC0415
            coriolis_vector,
            friction_torque_vector,
            gravity_vector,
        )

        theta1, phi, dtheta1, dphi = self._state
        C = coriolis_vector(phi, dtheta1, dphi, self._pendulum_params)
        G = gravity_vector(theta1, phi, self._pendulum_params)
        F = friction_torque_vector(dtheta1, dphi, self._pendulum_params)
        # Convention: M*q_ddot = tau - (C + G - F_dissipative)
        # bias = C + G + (-F) so that M*q_ddot + bias = tau
        return C + G - F

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravitational torque vector G(q)."""
        if not self._is_initialized:
            return np.zeros(2)

        from double_pendulum_golf.physics import gravity_vector  # noqa: PLC0415

        theta1, phi = float(self._state[0]), float(self._state[1])
        return gravity_vector(theta1, phi, self._pendulum_params)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute required joint torques τ = M(q)·q̈ + bias(q, q̇).

        Parameters
        ----------
        qacc : np.ndarray, shape (2,)
            Desired angular accelerations [rad/s²].
        """
        assert qacc is not None, "qacc must be provided"
        if not self._is_initialized or len(qacc) < 2:
            return np.zeros(2)

        M = self.compute_mass_matrix()
        bias = self.compute_bias_forces()
        return M @ qacc[:2] + bias

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (zero-torque) acceleration q̈_drift = -M⁻¹·bias."""
        if not self._is_initialized:
            return np.zeros(2)

        M = self.compute_mass_matrix()
        bias = self.compute_bias_forces()
        return np.linalg.solve(M, -bias)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration M⁻¹·τ.

        Parameters
        ----------
        tau : np.ndarray, shape (2,)
            Applied joint torques [N·m].
        """
        assert tau is not None, "tau must be provided"
        if not self._is_initialized or len(tau) < 2:
            return np.zeros(2)

        M = self.compute_mass_matrix()
        return np.linalg.solve(M, tau[:2])

    def compute_jacobian(self, body_name: str) -> dict[str, Any] | None:
        """Compute Jacobian for a named body ('wrist' or 'tip').

        Returns the 2×2 geometric Jacobian mapping joint velocities to
        Cartesian velocities at the requested point.  Returns ``None`` for
        unknown body names.
        """
        assert body_name is not None, "body_name must be provided"
        if not self._is_initialized:
            return None

        body = body_name.lower()
        if body not in {"wrist", "tip"}:
            logger.warning("Unknown body '%s'. Valid: 'wrist', 'tip'.", body_name)
            return None

        theta1, phi = float(self._state[0]), float(self._state[1])
        L1 = self._pendulum_params.L1
        L2 = self._pendulum_params.L2
        abs2 = theta1 + phi

        if body == "wrist":
            # d(wrist)/d(theta1), d(wrist)/d(phi)
            J = np.array(
                [
                    [L1 * np.cos(theta1), 0.0],
                    [L1 * np.sin(theta1), 0.0],
                ]
            )
        else:  # tip
            J = np.array(
                [
                    [L1 * np.cos(theta1) + L2 * np.cos(abs2), L2 * np.cos(abs2)],
                    [L1 * np.sin(theta1) + L2 * np.sin(abs2), L2 * np.sin(abs2)],
                ]
            )

        return {"J": J, "body": body_name}

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual at a given state (q, v)."""
        assert q is not None, "q must be provided"
        if not self._is_initialized or len(q) < 2 or len(v) < 2:
            return np.zeros(2)

        orig = self._state.copy()
        try:
            self._state = np.array([float(q[0]), float(q[1]), float(v[0]), float(v[1])])
            return self.compute_drift_acceleration()
        finally:
            self._state = orig

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual at position q with current control."""
        assert q is not None, "q must be provided"
        if not self._is_initialized or len(q) < 2:
            return np.zeros(2)

        orig = self._state.copy()
        try:
            self._state = np.array([float(q[0]), float(q[1]), 0.0, 0.0])
            g = self.compute_gravity_forces()
            M = self.compute_mass_matrix()
            return np.linalg.solve(M, -g + self._tau)
        finally:
            self._state = orig

    # ------------------------------------------------------------------
    # Checkpoint hooks (BasePhysicsEngine DRY contract)
    # ------------------------------------------------------------------

    def _get_extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "state": self._state.tolist(),
            "tau": self._tau.tolist(),
            "params_override": self._params_override,
        }

    def _restore_extra_checkpoint_state(self, checkpoint: StateCheckpoint) -> None:
        assert checkpoint is not None, "checkpoint must be provided"
        self.time = checkpoint.timestamp
        es = checkpoint.engine_state
        if "state" in es:
            self._state = np.array(es["state"])
        if "tau" in es:
            self._tau = np.array(es["tau"])

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def forward_kinematics(self) -> dict[str, tuple[float, float]]:
        """Return Cartesian positions of {shoulder, wrist, tip} (metres)."""
        if not self._is_initialized:
            return {"shoulder": (0.0, 0.0), "wrist": (0.0, 0.0), "tip": (0.0, 0.0)}

        from double_pendulum_golf.physics import forward_kinematics  # noqa: PLC0415

        theta1, phi = float(self._state[0]), float(self._state[1])
        return forward_kinematics(theta1, phi, self._pendulum_params)

    def clubhead_speed(self) -> float:
        """Return the current clubhead (tip) speed (m/s)."""
        if not self._is_initialized:
            return 0.0

        from double_pendulum_golf.physics import joint_velocities  # noqa: PLC0415

        return joint_velocities(self._state, self._pendulum_params)["tip_speed"]

    def total_energy(self) -> float:
        """Return total mechanical energy T + V (J)."""
        if not self._is_initialized:
            return 0.0

        from double_pendulum_golf.physics import total_energy  # noqa: PLC0415

        return total_energy(self._state, self._pendulum_params)

    # ------------------------------------------------------------------
    # Availability guard (graceful degradation)
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if the Tools pendulum package is importable."""
        return _check_tools_pendulum()
