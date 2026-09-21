"""Initial state mapping and forward kinematics agreement (TB-03 #10588).

Maps 2D in-plane landmark positions at t0 into double pendulum angular state
coordinates (theta1 from downward vertical, wrist angle theta2), verifies forward
kinematics closure, and estimates angular velocities without crossing NaN gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InitialStateMapping:
    """Calibrated initial dynamic state and forward kinematics verification.

    Attributes:
        theta1_rad: Upper segment angle from downward vertical [rad].
        theta2_rad: Wrist angle (lower segment relative to upper) [rad].
        omega1_rad_s: Upper segment angular velocity [rad/s].
        omega2_rad_s: Lower segment relative angular velocity [rad/s].
        theta_club_abs_rad: Absolute club angle from downward vertical [rad].
        closure_residual_m: Discrepancy between FK pose and target coordinates [m].
    """

    theta1_rad: float
    theta2_rad: float
    omega1_rad_s: float = 0.0
    omega2_rad_s: float = 0.0
    theta_club_abs_rad: float = 0.0
    closure_residual_m: float = 0.0

    def compute_fk(
        self, p_hub_2d: np.ndarray, arm_length_m: float, club_length_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics positions in 2D plane coordinates."""
        hub = np.asarray(p_hub_2d, dtype=float)
        th1 = self.theta1_rad
        abs_club = th1 + self.theta2_rad
        grip = hub + np.array(
            [
                arm_length_m * math.sin(th1),
                -arm_length_m * math.cos(th1),
            ]
        )
        head = grip + np.array(
            [
                club_length_m * math.sin(abs_club),
                -club_length_m * math.cos(abs_club),
            ]
        )
        return grip, head


def _wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _estimate_velocity_at_index(times: np.ndarray, traj: np.ndarray, idx: int) -> float:
    """Estimate derivative d(traj)/dt at idx using central or one-sided finite differences."""
    n = len(traj)
    if not (0 <= idx < n):
        raise ValueError(f"Index {idx} out of range [0, {n})")
    if not np.isfinite(traj[idx]):
        raise ValueError(f"Trajectory at index {idx} is not finite")

    # Check adjacency for missing data
    if idx > 0 and not np.isfinite(traj[idx - 1]):
        raise ValueError(f"Index {idx} is adjacent to missing / NaN data at {idx - 1}")
    if idx + 1 < n and not np.isfinite(traj[idx + 1]):
        raise ValueError(f"Index {idx} is adjacent to missing / NaN data at {idx + 1}")

    if idx > 0 and idx + 1 < n:
        # Central difference
        dt = float(times[idx + 1] - times[idx - 1])
        if dt <= 0.0:
            raise ValueError("Time values must be strictly increasing")
        return float((traj[idx + 1] - traj[idx - 1]) / dt)
    if idx + 1 < n:
        # Forward difference
        dt = float(times[idx + 1] - times[idx])
        if dt <= 0.0:
            raise ValueError("Time values must be strictly increasing")
        return float((traj[idx + 1] - traj[idx]) / dt)
    # Backward difference
    dt = float(times[idx] - times[idx - 1])
    if dt <= 0.0:
        raise ValueError("Time values must be strictly increasing")
    return float((traj[idx] - traj[idx - 1]) / dt)


def estimate_initial_state(
    p_hub_2d: np.ndarray,
    p_grip_2d: np.ndarray,
    p_head_2d: np.ndarray,
    arm_length_m: float,
    club_length_m: float,
    *,
    times: np.ndarray | None = None,
    theta1_traj: np.ndarray | None = None,
    theta2_traj: np.ndarray | None = None,
    t0_idx: int = 0,
) -> InitialStateMapping:
    """Map 2D in-plane landmarks to double pendulum state coordinates."""
    hub = np.asarray(p_hub_2d, dtype=float)
    grip = np.asarray(p_grip_2d, dtype=float)
    head = np.asarray(p_head_2d, dtype=float)

    delta_arm = grip - hub
    # Downward vertical is [0, -1], so u = sin(th), -v = cos(th) -> th = atan2(u, -v)
    th1 = float(math.atan2(delta_arm[0], -delta_arm[1]))

    delta_club = head - grip
    abs_club = float(math.atan2(delta_club[0], -delta_club[1]))
    th2 = float(_wrap_to_pi(abs_club - th1))

    # Compute FK closure residual
    fk_grip = hub + np.array(
        [
            arm_length_m * math.sin(th1),
            -arm_length_m * math.cos(th1),
        ]
    )
    fk_head = fk_grip + np.array(
        [
            club_length_m * math.sin(abs_club),
            -club_length_m * math.cos(abs_club),
        ]
    )
    err_grip = float(np.linalg.norm(fk_grip - grip))
    err_head = float(np.linalg.norm(fk_head - head))
    closure = max(err_grip, err_head)

    # Estimate angular velocities if trajectories provided
    om1 = 0.0
    om2 = 0.0
    if times is not None and theta1_traj is not None:
        om1 = _estimate_velocity_at_index(times, theta1_traj, t0_idx)
    if times is not None and theta2_traj is not None:
        om2 = _estimate_velocity_at_index(times, theta2_traj, t0_idx)

    return InitialStateMapping(
        theta1_rad=th1,
        theta2_rad=th2,
        omega1_rad_s=om1,
        omega2_rad_s=om2,
        theta_club_abs_rad=abs_club,
        closure_residual_m=closure,
    )
