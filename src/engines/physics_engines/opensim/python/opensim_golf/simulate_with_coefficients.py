"""OpenSim forward simulator with polynomial torque input (issue #4120).

Implements forward dynamics integration using OpenSim's Manager with a custom
PolynomialTorqueController. Returns canonical SimOut trajectory aligned to
the simulation grid.

Public API:
    SimOptions             -- frozen dataclass for simulation parameters
    SimOut                 -- frozen dataclass for trajectory output
    simulate_with_coefficients -- main entry point
    synthesize_target_from_coefficients -- TDD oracle helper
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts.decorators import (
    postcondition,
    precondition,
)
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)

logger = logging.getLogger(__name__)

# Try to import OpenSim; skip tests if unavailable
try:
    import opensim

    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False

__all__ = [
    "SimOptions",
    "SimOut",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]

# Global model cache (per-process, thread-unsafe by design)
_CACHED_MODEL: Any = None
_CACHED_STATE: Any = None
_GOLFER_OSIM_PATH: str | None = None


def _get_golfer_osim_path() -> str:
    """Locate golf_humanoid.osim in the package.

    Returns:
        Path to the OpenSim model file

    Raises:
        FileNotFoundError: If golf_humanoid.osim not found
    """
    # golf_humanoid.osim lives in src/engines/physics_engines/opensim/models/
    osim_path = Path(__file__).parent.parent / "models" / "golf_humanoid.osim"
    if not osim_path.exists():
        raise FileNotFoundError(
            f"golf_humanoid.osim not found at {osim_path}. "
            "Please run scripts/build_humanoid_osim.py to generate it."
        )
    return str(osim_path)


def _load_opensim_model() -> tuple[Any, Any]:
    """Load OpenSim model and initialize state, cached at module level.

    Returns:
        (model, initial_state) tuple

    Raises:
        ImportError: If OpenSim not available
        FileNotFoundError: If model file not found
    """
    global _CACHED_MODEL, _CACHED_STATE, _GOLFER_OSIM_PATH

    if _CACHED_MODEL is not None and _CACHED_STATE is not None:
        # Reset state to initial condition
        _CACHED_STATE = _CACHED_MODEL.initSystem()
        return _CACHED_MODEL, _CACHED_STATE

    if not OPENSIM_AVAILABLE:
        raise ImportError("OpenSim is not available; cannot load model")

    _GOLFER_OSIM_PATH = _get_golfer_osim_path()
    _CACHED_MODEL = opensim.Model(_GOLFER_OSIM_PATH)
    _CACHED_STATE = _CACHED_MODEL.initSystem()

    logger.info(f"Loaded OpenSim model from {_GOLFER_OSIM_PATH}")
    logger.info(
        f"Model: nq={_CACHED_MODEL.getNumCoordinates()}, "
        f"nu={_CACHED_MODEL.getNumSpeeds()}"
    )

    return _CACHED_MODEL, _CACHED_STATE


@dataclass(frozen=True)
class SimOptions:
    """Simulation options for forward integration.

    Attributes:
        t_final:    Final simulation time (seconds). Must be positive.
        dt:         Integration step size (seconds). Must be positive and <= t_final.
        integrator: Integration method. "rk4" is recommended; "semiexplicit" is faster.
        tolerance:  Integrator tolerance (default 1e-5).
    """

    t_final: float = 1.0
    dt: float = 1e-3
    integrator: Literal["rk4", "semiexplicit"] = "rk4"
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        """Validate simulation parameters."""
        if not (self.t_final > 0):
            raise ValueError("t_final must be positive")
        if not (self.dt > 0):
            raise ValueError("dt must be positive")
        if not (self.dt <= self.t_final):
            raise ValueError("dt must not exceed t_final")
        if not (self.tolerance > 0):
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class SimOut:
    """Complete trajectory from forward simulation.

    Mirrors the canonical schema from the cross-engine spec.

    Attributes:
        time:            (N,) time grid in seconds, starts at 0, strictly increasing
        q:               (N, nq) joint positions (rad)
        qd:              (N, nv) joint velocities (rad/s)
        qdd:             (N, nv) joint accelerations (rad/s^2)
        tau:             (N, nv) joint torques (N·m)
        grip:            (N, 3) mid-hands position (m), world frame
        grip_quat:       (N, 4) mid-hands quaternion [w, x, y, z], world frame
        clubhead:        (N, 3) club face position (m), world frame
        club_quat:       (N, 4) club orientation quaternion [w, x, y, z], world frame
        solver_status:   "success" | "warning" | "failed"
        duration_s:      Wall-clock simulation time (seconds)
    """

    time: NDArray[np.float64]
    q: NDArray[np.float64]
    qd: NDArray[np.float64]
    qdd: NDArray[np.float64]
    tau: NDArray[np.float64]
    grip: NDArray[np.float64]
    grip_quat: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    solver_status: str
    duration_s: float = 0.0


def _validate_theta(theta: NDArray[np.float64], n_joints: int) -> NDArray[np.float64]:
    """Validate theta vector format and contents.

    Args:
        theta: Coefficient vector
        n_joints: Expected number of joints

    Returns:
        Validated theta as ndarray

    Raises:
        ValueError: If theta is non-finite or wrong length
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D, got shape {theta.shape}")
    expected_len = n_joints * 7
    if len(theta) != expected_len:
        raise ValueError(
            f"theta must have length {expected_len} (n_joints={n_joints} * 7), "
            f"got {len(theta)}"
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain only finite values (no NaN or Inf)")
    return theta


def _extract_frames(
    model: Any, state: Any
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Extract mid-hands (grip) and club-head frames from OpenSim state.

    Args:
        model: OpenSim Model
        state: OpenSim State with current kinematics

    Returns:
        (grip_pos, grip_quat, clubhead_pos, club_quat)

    Raises:
        RuntimeError: If frames not found in model
    """
    # Realize to position stage for kinematics
    model.realizePosition(state)

    # Get body set and frame references
    body_set = model.getBodySet()

    try:
        # Try to get grip frame from right hand
        right_hand = body_set.get("right_hand")
        grip_transform = right_hand.getTransformInGround(state)
    except Exception:
        logger.warning("right_hand frame not found, using alternate grip location")
        try:
            grip_body = body_set.get("hand_r")
            grip_transform = grip_body.getTransformInGround(state)
        except Exception as e:
            raise RuntimeError(
                "Could not find grip frame (tried right_hand, hand_r)"
            ) from e

    try:
        # Try to get clubhead frame
        club_body = body_set.get("club")
        club_transform = club_body.getTransformInGround(state)
    except Exception as e:
        raise RuntimeError("Could not find club body") from e

    # Extract positions (translation)
    grip_pos = np.array(
        [
            grip_transform.p().get(0),
            grip_transform.p().get(1),
            grip_transform.p().get(2),
        ],
        dtype=np.float64,
    )

    clubhead_pos = np.array(
        [
            club_transform.p().get(0),
            club_transform.p().get(1),
            club_transform.p().get(2),
        ],
        dtype=np.float64,
    )

    # Convert rotation matrices to quaternions [w, x, y, z]
    # OpenSim uses [x, y, z, w] (Eigen convention); we convert to [w, x, y, z]
    grip_rot = grip_transform.R()
    club_rot = club_transform.R()

    grip_quat = _rotation_matrix_to_quaternion(grip_rot)
    club_quat = _rotation_matrix_to_quaternion(club_rot)

    return grip_pos, grip_quat, clubhead_pos, club_quat


def _rotation_matrix_to_quaternion(rot: Any) -> NDArray[np.float64]:
    """Convert OpenSim rotation matrix to quaternion [w, x, y, z].

    Args:
        rot: OpenSim Rotation (SimTK::Rotation)

    Returns:
        (4,) quaternion [w, x, y, z]
    """
    # Extract rotation matrix elements
    mat = np.array(
        [
            [rot.get(0, 0), rot.get(0, 1), rot.get(0, 2)],
            [rot.get(1, 0), rot.get(1, 1), rot.get(1, 2)],
            [rot.get(2, 0), rot.get(2, 1), rot.get(2, 2)],
        ],
        dtype=np.float64,
    )

    # Convert rotation matrix to quaternion using Shepperd's method
    trace = np.trace(mat)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    # Normalize and ensure w is positive (canonical orientation)
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 1e-10:
        quat = quat / quat_norm
    if quat[0] < 0:
        quat = -quat

    return quat


@precondition(lambda theta, opts: OPENSIM_AVAILABLE, "OpenSim must be available")
@precondition(lambda theta, opts: opts is not None, "opts must be provided")
@postcondition(
    lambda result: result.solver_status in ("success", "warning", "failed"),
    "solver_status must be 'success', 'warning', or 'failed'",
)
@postcondition(
    lambda result: np.all(np.isfinite(result.q)),
    "q must be finite",
)
@postcondition(
    lambda result: len(result.time) == result.q.shape[0],
    "time and q must have matching row counts",
)
def simulate_with_coefficients(
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: dict[str, NDArray[np.float64]] | None = None,
) -> SimOut:
    """Forward-simulate the golfer + club with polynomial torque input.

    Uses OpenSim's Manager with a custom PolynomialTorqueController to integrate
    the forward dynamics equations.

    Args:
        theta: (n_joints * 7,) polynomial torque coefficients. Each joint
               has 7 coefficients [a0, a1, ..., a6] for tau(t) = sum a_k * t^k.
        options: SimOptions with t_final, dt, integrator. Defaults provided.
        initial_pose: Optional dict with keys 'q', 'qd' to override initial state.
                      If not provided, uses neutral pose with zero velocity.

    Returns:
        SimOut: Complete trajectory containing time, states, torques, and
                grip/club kinematics at each step.

    Raises:
        ValueError: If theta is non-finite, wrong length, or opts invalid.
        FileNotFoundError: If golf_humanoid.osim not found.
        ImportError: If OpenSim unavailable.
        RuntimeError: If simulation fails or frames not found.
    """
    wall_start = time.time()

    if options is None:
        options = SimOptions()

    # Load model and initialize state
    model, state = _load_opensim_model()
    n_joints = model.getNumSpeeds()

    # Validate inputs
    theta = _validate_theta(theta, n_joints)

    # Set up controller
    from .controller import PolynomialTorqueController

    controller = PolynomialTorqueController(theta, n_joints)
    model.addController(controller)

    # Initialize state with provided or default initial conditions
    if initial_pose is not None:
        q_init = np.asarray(
            initial_pose.get("q", np.zeros(model.getNumCoordinates())), dtype=np.float64
        )
        qd_init = np.asarray(
            initial_pose.get("qd", np.zeros(n_joints)), dtype=np.float64
        )

        # Set state
        q_vec = state.getQ()
        for i in range(len(q_init)):
            q_vec.set(i, float(q_init[i]))
        state.setQ(q_vec)

        u_vec = state.getU()
        for i in range(len(qd_init)):
            u_vec.set(i, float(qd_init[i]))
        state.setU(u_vec)
    else:
        # Default: neutral pose with zero velocity
        q_init = np.zeros(model.getNumCoordinates())
        qd_init = np.zeros(n_joints)

    # Build time grid
    n_steps = int(np.ceil(options.t_final / options.dt)) + 1
    time_grid = np.linspace(0, options.t_final, n_steps)

    # Set up integrator
    integrator_name = (
        "RungeKuttaMerson" if options.integrator == "rk4" else "SemiExplicitEuler2"
    )
    integrator = getattr(opensim, integrator_name)(model)
    integrator.setAccuracy(options.tolerance)

    manager = opensim.Manager(model)
    manager.setIntegrator(integrator)

    # Allocate output arrays
    N = len(time_grid)
    time_out = time_grid.copy()
    q_out = np.zeros((N, model.getNumCoordinates()))
    qd_out = np.zeros((N, n_joints))
    qdd_out = np.zeros((N, n_joints))
    tau_out = np.zeros((N, n_joints))
    grip_out = np.zeros((N, 3))
    grip_quat_out = np.zeros((N, 4))
    clubhead_out = np.zeros((N, 3))
    club_quat_out = np.zeros((N, 4))

    # Store initial state
    q_vec = state.getQ()
    u_vec = state.getU()
    for i in range(model.getNumCoordinates()):
        q_out[0, i] = q_vec.get(i)
    for i in range(n_joints):
        qd_out[0, i] = u_vec.get(i)

    # Initial kinematics
    try:
        grip_out[0], grip_quat_out[0], clubhead_out[0], club_quat_out[0] = (
            _extract_frames(model, state)
        )
    except RuntimeError as e:
        logger.error(f"Failed to extract initial frames: {e}")
        return SimOut(
            time=time_out,
            q=q_out,
            qd=qd_out,
            qdd=qdd_out,
            tau=tau_out,
            grip=grip_out,
            grip_quat=grip_quat_out,
            clubhead=clubhead_out,
            club_quat=club_quat_out,
            solver_status="failed",
            duration_s=time.time() - wall_start,
        )

    # Initial torques and accelerations
    model.realizeDynamics(state)
    udot_vec = state.getUDot()
    for i in range(n_joints):
        qdd_out[0, i] = udot_vec.get(i)
        # Initial torques from controller
        tau_out[0, i] = controller.tau_at(0.0, i)

    # Integration loop
    solver_status = "success"
    try:
        for i in range(N - 1):
            t_curr = time_grid[i]
            t_next = time_grid[i + 1]

            # Integrate to next time step
            manager.setInitialTime(t_curr)
            manager.setFinalTime(t_next)
            manager.integrate(t_next)

            # Snapshot state at next time
            q_vec = state.getQ()
            u_vec = state.getU()
            for j in range(model.getNumCoordinates()):
                q_out[i + 1, j] = q_vec.get(j)
            for j in range(n_joints):
                qd_out[i + 1, j] = u_vec.get(j)

            # Compute acceleration and torques at next time
            model.realizeDynamics(state)
            udot_vec = state.getUDot()
            for j in range(n_joints):
                qdd_out[i + 1, j] = udot_vec.get(j)
                tau_out[i + 1, j] = controller.tau_at(t_next, j)

            # Extract grip and clubhead frames
            (
                grip_out[i + 1],
                grip_quat_out[i + 1],
                clubhead_out[i + 1],
                club_quat_out[i + 1],
            ) = _extract_frames(model, state)

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        solver_status = "failed"

    # Pack result
    wall_duration = time.time() - wall_start

    return SimOut(
        time=time_out,
        q=q_out,
        qd=qd_out,
        qdd=qdd_out,
        tau=tau_out,
        grip=grip_out,
        grip_quat=grip_quat_out,
        clubhead=clubhead_out,
        club_quat=club_quat_out,
        solver_status=solver_status,
        duration_s=wall_duration,
    )


def synthesize_target_from_coefficients(
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
) -> ClubTarget:
    """TDD oracle: synthesize a ClubTarget from polynomial torque coefficients.

    Runs simulate_with_coefficients and repackages SimOut into the canonical
    ClubTarget format. Used by tests to construct ground-truth (theta, target)
    pairs for recovery tests.

    Args:
        theta: Polynomial torque coefficients
        options: Simulation options (defaults if None)

    Returns:
        ClubTarget: Canonical motion-matching target schema

    Raises:
        ValueError: If theta invalid or simulation fails
    """
    if options is None:
        options = SimOptions()

    result = simulate_with_coefficients(theta, options)

    if result.solver_status != "success":
        raise ValueError(f"Simulation failed with status: {result.solver_status}")

    # Use grip position as the primary anchor (butt in the canonical schema)
    # and clubhead as secondary
    target = ClubTarget(
        time=result.time,
        butt=result.grip,
        clubhead=result.clubhead,
        club_quat=result.club_quat,
        impact_idx=max(1, len(result.time) // 2),  # Default: impact at midpoint
        source=SourceProvenance(
            filename="synthesized",
            format="opensim_rk4",
            subject_id="synthetic",
            trial_id="synthesized",
            sha256="",
        ),
    )

    return target
