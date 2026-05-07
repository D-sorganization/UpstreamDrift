"""Pinocchio forward simulator with polynomial torque input (issue #4118).

Implements RK4 integration + Articulated Body Algorithm (ABA) for the golfer +
club system. Polynomial torques are specified by degree-6 coefficients, one
vector (7 coeffs) per joint.

Public API:
    SimOptions         -- frozen dataclass for simulation parameters
    SimOut             -- frozen dataclass for trajectory output
    simulate_with_coefficients -- main entry point
    synthesize_target_from_coefficients -- TDD oracle helper
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts.decorators import (
    postcondition,
    precondition,
)

logger = logging.getLogger(__name__)

# Try to import Pinocchio; skip tests if unavailable
try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False

__all__ = [
    "SimOptions",
    "SimOut",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]

# Global model and data cache (per-process, thread-unsafe by design)
_CACHED_MODEL: pin.Model | None = None
_CACHED_DATA: pin.Data | None = None
_GOLFER_URDF_PATH: str | None = None


def _get_golfer_urdf_path() -> str:
    """Locate golfer.urdf in the package."""
    from pathlib import Path

    # golfer.urdf lives in src/engines/physics_engines/pinocchio/models/generated/
    urdf_path = (
        Path(__file__).parent.parent.parent / "models" / "generated" / "golfer.urdf"
    )
    if not urdf_path.exists():
        raise FileNotFoundError(f"golfer.urdf not found at {urdf_path}")
    return str(urdf_path)


def _load_pinocchio_model() -> tuple[pin.Model, pin.Data]:
    """Load Pinocchio model and data, cached at module level."""
    global _CACHED_MODEL, _CACHED_DATA, _GOLFER_URDF_PATH

    if _CACHED_MODEL is not None and _CACHED_DATA is not None:
        return _CACHED_MODEL, _CACHED_DATA

    if not PINOCCHIO_AVAILABLE:
        raise ImportError("Pinocchio is not available; cannot load model")

    _GOLFER_URDF_PATH = _get_golfer_urdf_path()
    _CACHED_MODEL = pin.buildModelFromUrdf(_GOLFER_URDF_PATH)
    _CACHED_DATA = _CACHED_MODEL.createData()

    logger.info(f"Loaded Pinocchio model from {_GOLFER_URDF_PATH}")
    logger.info(f"Model: nq={_CACHED_MODEL.nq}, nv={_CACHED_MODEL.nv}")

    return _CACHED_MODEL, _CACHED_DATA


@dataclass(frozen=True)
class SimOptions:
    """Simulation options for forward integration.

    Attributes:
        t_final:    Final simulation time (seconds). Must be positive.
        dt:         Integration step size (seconds). Must be positive and <= t_final.
        integrator: Integration method. Either "rk4" or "semi_implicit".
    """

    t_final: float = 1.0
    dt: float = 1e-3
    integrator: Literal["rk4", "semi_implicit"] = "rk4"

    def __post_init__(self) -> None:
        """Validate simulation parameters."""
        if not (self.t_final > 0):
            raise ValueError("t_final must be positive")
        if not (self.dt > 0):
            raise ValueError("dt must be positive")
        if not (self.dt <= self.t_final):
            raise ValueError("dt must not exceed t_final")


@dataclass(frozen=True)
class SimOut:
    """Complete trajectory from forward simulation.

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


def _evaluate_torque_polynomial(t: float, coeffs: NDArray[np.float64]) -> float:
    """Evaluate polynomial tau(t) = sum_k a_k * t^k for a single joint.

    Args:
        t: Time (seconds)
        coeffs: (7,) array [a0, a1, ..., a6]

    Returns:
        tau(t) scalar
    """
    if len(coeffs) != 7:
        raise ValueError(f"coeffs must be length 7, got {len(coeffs)}")
    result = 0.0
    for k, a_k in enumerate(coeffs):
        result += a_k * (t**k)
    return result


def _evaluate_torque_vector(
    t: float, theta: NDArray[np.float64], n_joints: int
) -> NDArray[np.float64]:
    """Evaluate polynomial torques at time t for all joints.

    Args:
        t: Time (seconds)
        theta: (n_joints * 7,) flat coefficient vector
        n_joints: Number of joints

    Returns:
        (n_joints,) torque vector
    """
    tau = np.zeros(n_joints)
    for j in range(n_joints):
        coeffs = theta[j * 7 : (j + 1) * 7]
        tau[j] = _evaluate_torque_polynomial(t, coeffs)
    return tau


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
    model: pin.Model, data: pin.Data
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Extract mid-hands (grip) and club-head frames from Pinocchio data.

    Args:
        model: Pinocchio model
        data: Pinocchio data with current kinematics

    Returns:
        (grip_pos, grip_quat, clubhead_pos, club_quat)
    """
    # Get frame IDs
    try:
        grip_frame_id = model.getFrameId("mid_hands")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"mid_hands frame not found: {e}, trying hand_left")
        grip_frame_id = model.getFrameId("hand_left")

    try:
        clubhead_frame_id = model.getFrameId("club_head")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"club_head frame not found: {e}, trying club_shaft")
        clubhead_frame_id = model.getFrameId("club_shaft")

    # Get SE3 transforms
    grip_se3 = data.oMf[grip_frame_id]
    clubhead_se3 = data.oMf[clubhead_frame_id]

    # Extract positions
    grip_pos = grip_se3.translation.copy()
    clubhead_pos = clubhead_se3.translation.copy()

    # Convert rotation matrices to quaternions [w, x, y, z]
    grip_quat = pin.Quaternion(grip_se3.rotation)
    club_quat = pin.Quaternion(clubhead_se3.rotation)

    grip_quat_array = np.array([grip_quat.w, grip_quat.x, grip_quat.y, grip_quat.z])
    club_quat_array = np.array([club_quat.w, club_quat.x, club_quat.y, club_quat.z])

    return grip_pos, grip_quat_array, clubhead_pos, club_quat_array


@precondition(lambda theta, opts: PINOCCHIO_AVAILABLE, "Pinocchio must be available")
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
    initial_pose: dict | None = None,
) -> SimOut:
    """Forward-simulate the golfer + club with polynomial torque input.

    Uses RK4 integration with Pinocchio's ABA (Articulated Body Algorithm)
    to compute dynamics.

    Args:
        theta: (n_joints * 7,) polynomial torque coefficients where n_joints is
               the number of DOFs in the model (43 for golfer.urdf). Each joint
               has 7 coefficients [a0, a1, ..., a6] for tau(t) = sum a_k * t^k.
        options: SimOptions with t_final, dt, integrator. Defaults provided.
        initial_pose: Optional dict with keys 'q', 'qd' to override initial state.
                      If not provided, uses neutral pose with zero velocity.

    Returns:
        SimOut: Complete trajectory containing time, states, torques, and
                grip/club kinematics at each step.

    Raises:
        ValueError: If theta is non-finite, wrong length, or opts invalid.
        FileNotFoundError: If golfer.urdf not found.
        ImportError: If Pinocchio unavailable.
    """
    if options is None:
        options = SimOptions()

    # Load model and data
    model, data = _load_pinocchio_model()

    # Validate inputs
    theta = _validate_theta(theta, model.nv)

    # Initialize state
    if initial_pose is not None:
        q = np.asarray(initial_pose.get("q", pin.neutral(model)), dtype=np.float64)
        qd = np.asarray(initial_pose.get("qd", np.zeros(model.nv)), dtype=np.float64)
    else:
        q = pin.neutral(model)
        qd = np.zeros(model.nv)

    # Build time grid
    n_steps = int(np.ceil(options.t_final / options.dt)) + 1
    time_grid = np.linspace(0, options.t_final, n_steps)

    # Allocate output arrays
    N = len(time_grid)
    time_out = time_grid.copy()
    q_out = np.zeros((N, model.nq))
    qd_out = np.zeros((N, model.nv))
    qdd_out = np.zeros((N, model.nv))
    tau_out = np.zeros((N, model.nv))
    grip_out = np.zeros((N, 3))
    grip_quat_out = np.zeros((N, 4))
    clubhead_out = np.zeros((N, 3))
    club_quat_out = np.zeros((N, 4))

    # Store initial state
    q_out[0] = q
    qd_out[0] = qd

    # Forward kinematics at t=0
    pin.forwardKinematics(model, data, q, qd, np.zeros(model.nv))
    pin.updateFramePlacements(model, data)
    grip_out[0], grip_quat_out[0], clubhead_out[0], club_quat_out[0] = _extract_frames(
        model, data
    )

    # Initial acceleration
    tau_init = _evaluate_torque_vector(time_grid[0], theta, model.nv)
    qdd_init = cast(NDArray[np.float64], pin.aba(model, data, q, qd, tau_init))
    qdd_out[0] = qdd_init
    tau_out[0] = tau_init

    # RK4 or semi-implicit Euler integration
    for i in range(N - 1):
        t_curr = time_grid[i]
        dt = min(options.dt, time_grid[i + 1] - t_curr)

        if options.integrator == "rk4":
            q, qd = _rk4_step(model, data, q, qd, t_curr, dt, theta)
        elif options.integrator == "semi_implicit":
            q, qd = _semi_implicit_step(model, data, q, qd, t_curr, dt, theta)
        else:
            raise ValueError(f"unknown integrator {options.integrator!r}")

        # Store state at next step
        i_next = i + 1
        q_out[i_next] = q
        qd_out[i_next] = qd

        # Compute acceleration and torque at next time
        t_next = time_grid[i_next]
        tau_next = _evaluate_torque_vector(t_next, theta, model.nv)
        qdd_next = cast(NDArray[np.float64], pin.aba(model, data, q, qd, tau_next))

        qdd_out[i_next] = qdd_next
        tau_out[i_next] = tau_next

        # Forward kinematics for grip and clubhead
        pin.forwardKinematics(model, data, q, qd, qdd_next)
        pin.updateFramePlacements(model, data)
        (
            grip_out[i_next],
            grip_quat_out[i_next],
            clubhead_out[i_next],
            club_quat_out[i_next],
        ) = _extract_frames(model, data)

    # Pack result
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
        solver_status="success",
    )


def _rk4_step(
    model: pin.Model,
    data: pin.Data,
    q: NDArray[np.float64],
    qd: NDArray[np.float64],
    t: float,
    dt: float,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RK4 integration step on the tangent space.

    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Current positions
        qd: Current velocities
        t: Current time
        dt: Step size
        theta: Polynomial coefficients

    Returns:
        (q_next, qd_next)
    """
    q0 = q.copy()
    qd0 = qd.copy()

    # Stage 1
    tau1 = _evaluate_torque_vector(t, theta, model.nv)
    qdd1 = cast(NDArray[np.float64], pin.aba(model, data, q0, qd0, tau1))

    # Stage 2
    qd2 = qd0 + 0.5 * dt * qdd1
    q2 = pin.integrate(model, q0, 0.5 * dt * qd0)
    tau2 = _evaluate_torque_vector(t + 0.5 * dt, theta, model.nv)
    qdd2 = cast(NDArray[np.float64], pin.aba(model, data, q2, qd2, tau2))

    # Stage 3
    qd3 = qd0 + 0.5 * dt * qdd2
    q3 = pin.integrate(model, q0, 0.5 * dt * qd2)
    tau3 = _evaluate_torque_vector(t + 0.5 * dt, theta, model.nv)
    qdd3 = cast(NDArray[np.float64], pin.aba(model, data, q3, qd3, tau3))

    # Stage 4
    qd4 = qd0 + dt * qdd3
    q4 = pin.integrate(model, q0, dt * qd3)
    tau4 = _evaluate_torque_vector(t + dt, theta, model.nv)
    qdd4 = cast(NDArray[np.float64], pin.aba(model, data, q4, qd4, tau4))

    # Weighted average
    qd_avg = (qd0 + 2.0 * qd2 + 2.0 * qd3 + qd4) / 6.0
    qd_next = qd0 + dt * (qdd1 + 2.0 * qdd2 + 2.0 * qdd3 + qdd4) / 6.0
    q_next = pin.integrate(model, q0, dt * qd_avg)

    return q_next, qd_next


def _semi_implicit_step(
    model: pin.Model,
    data: pin.Data,
    q: NDArray[np.float64],
    qd: NDArray[np.float64],
    t: float,
    dt: float,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Semi-implicit (symplectic) Euler integration step.

    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Current positions
        qd: Current velocities
        t: Current time
        dt: Step size
        theta: Polynomial coefficients

    Returns:
        (q_next, qd_next)
    """
    tau = _evaluate_torque_vector(t, theta, model.nv)
    qdd = cast(NDArray[np.float64], pin.aba(model, data, q, qd, tau))

    qd_next = qd + dt * qdd
    q_next = pin.integrate(model, q, dt * qd_next)

    return q_next, qd_next


def synthesize_target_from_coefficients(
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
) -> object:
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
    # Import here to avoid loading pandas/c3d at module level
    from src.shared.python.motion_matching.club_target import (
        ClubTarget,
        SourceProvenance,
    )

    if options is None:
        options = SimOptions()

    result = simulate_with_coefficients(theta, options)

    # Use grip position as the primary anchor (butt in the canonical schema)
    # and clubhead as secondary
    target = ClubTarget(
        time=result.time,
        butt=result.grip,
        clubhead=result.clubhead,
        club_quat=result.club_quat,
        impact_idx=len(result.time) // 2,  # Default: impact at midpoint
        source=SourceProvenance(
            filename="synthesized",
            format="pinocchio_rk4",
            subject_id="synthetic",
            trial_id="synthesized",
            sha256="",
        ),
    )

    return target
