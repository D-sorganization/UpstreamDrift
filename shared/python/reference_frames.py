"""Reference Frame Transformations for Forces and Torques.

Guideline E4 Implementation: Reference Frame Representations.

Provides transformations between multiple reference frames for force/torque analysis:
- Global (world) frame: Fixed inertial reference
- Local (body) frame: Attached to each segment
- Swing plane frame: Golf-specific task-oriented coordinates

Also includes Functional Swing Plane (FSP) computation for post-simulation analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

# Geometric computation tolerance
GEOMETRIC_TOLERANCE = 1e-10  # [unitless] For near-zero vector magnitude checks


class ReferenceFrame(Enum):
    """Enumeration of supported reference frames."""

    GLOBAL = auto()  # Fixed inertial (world) frame
    LOCAL = auto()  # Body-attached local frame
    SWING_PLANE = auto()  # Golf swing plane frame


@dataclass
class SwingPlaneFrame:
    """Swing plane reference frame definition.

    Attributes:
        origin: Origin point in global frame [m]
        normal: Normal vector (out of plane) [unitless]
        in_plane_x: In-plane X axis (tangent to swing direction) [unitless]
        in_plane_y: In-plane Y axis (perpendicular in plane) [unitless]
        grip_axis: Axis along golf grip/shaft [unitless]
        fitting_rmse: RMS deviation of trajectory from plane [m]
        fitting_window_ms: Time window used for fitting [ms]
    """

    origin: np.ndarray
    normal: np.ndarray
    in_plane_x: np.ndarray
    in_plane_y: np.ndarray
    grip_axis: np.ndarray
    fitting_rmse: float = 0.0
    fitting_window_ms: float = 100.0


@dataclass
class WrenchInFrame:
    """Force and torque (wrench) in a specific reference frame.

    Attributes:
        force: Force vector [N] (3,)
        torque: Torque vector [N·m] (3,)
        frame: Reference frame of the wrench
        body_name: Name of body this wrench acts on (if applicable)
        point: Application point in the specified frame [m] (3,)
    """

    force: np.ndarray
    torque: np.ndarray
    frame: ReferenceFrame
    body_name: str = ""
    point: np.ndarray | None = None


def compute_rotation_matrix_from_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    """Create rotation matrix from orthonormal axis vectors.

    The rotation matrix R transforms from the local frame to global frame:
        v_global = R @ v_local

    Args:
        x_axis: Local X axis expressed in global frame (3,)
        y_axis: Local Y axis expressed in global frame (3,)
        z_axis: Local Z axis expressed in global frame (3,)

    Returns:
        R: 3x3 rotation matrix (columns are local axes in global frame)
    """
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


def transform_wrench_to_frame(
    wrench: WrenchInFrame,
    target_frame: ReferenceFrame,
    rotation_to_target: np.ndarray,
) -> WrenchInFrame:
    """Transform a wrench from one reference frame to another.

    Args:
        wrench: Wrench in source frame
        target_frame: Target reference frame
        rotation_to_target: Rotation matrix R where v_target = R @ v_source

    Returns:
        Wrench in target frame
    """
    # Force transforms like a vector: f_target = R @ f_source
    force_target = rotation_to_target @ wrench.force

    # Torque transforms like a pseudovector: tau_target = R @ tau_source
    torque_target = rotation_to_target @ wrench.torque

    point_target = None
    if wrench.point is not None:
        point_target = rotation_to_target @ wrench.point

    return WrenchInFrame(
        force=force_target,
        torque=torque_target,
        frame=target_frame,
        body_name=wrench.body_name,
        point=point_target,
    )


def fit_instantaneous_swing_plane(
    clubhead_velocity: np.ndarray,
    grip_position: np.ndarray,
    clubhead_position: np.ndarray,
) -> SwingPlaneFrame:
    """Compute instantaneous swing plane from current state.

    The swing plane is defined by the clubhead velocity direction and
    the grip-to-clubhead vector.

    Args:
        clubhead_velocity: Clubhead velocity vector [m/s] (3,)
        grip_position: Grip center position [m] (3,)
        clubhead_position: Clubhead position [m] (3,)

    Returns:
        SwingPlaneFrame for the current instant
    """
    # Grip axis (shaft direction)
    grip_to_club = clubhead_position - grip_position
    grip_axis_length = np.linalg.norm(grip_to_club)
    if grip_axis_length < GEOMETRIC_TOLERANCE:
        LOGGER.warning("Grip and clubhead positions too close")
        grip_axis = np.array([0.0, 0.0, 1.0])
    else:
        grip_axis = grip_to_club / grip_axis_length

    # In-plane X is the velocity direction (tangent to swing)
    vel_magnitude = np.linalg.norm(clubhead_velocity)
    if vel_magnitude < GEOMETRIC_TOLERANCE:
        LOGGER.warning("Clubhead velocity too small for plane fitting")
        in_plane_x = np.array([1.0, 0.0, 0.0])
    else:
        in_plane_x = clubhead_velocity / vel_magnitude

    # Normal is perpendicular to both velocity and grip axis
    normal = np.cross(grip_axis, in_plane_x)
    normal_mag = np.linalg.norm(normal)
    if normal_mag < GEOMETRIC_TOLERANCE:
        # Velocity is parallel to shaft - degenerate case
        LOGGER.warning("Degenerate swing plane: velocity parallel to shaft")
        normal = np.array([0.0, 1.0, 0.0])
    else:
        normal = normal / normal_mag

    # In-plane Y is perpendicular to both normal and in-plane X
    in_plane_y = np.cross(normal, in_plane_x)
    in_plane_y = in_plane_y / np.linalg.norm(in_plane_y)

    return SwingPlaneFrame(
        origin=clubhead_position,
        normal=normal,
        in_plane_x=in_plane_x,
        in_plane_y=in_plane_y,
        grip_axis=grip_axis,
        fitting_rmse=0.0,
        fitting_window_ms=0.0,
    )


def fit_functional_swing_plane(
    clubhead_trajectory: np.ndarray,
    timestamps: np.ndarray,
    impact_time: float,
    window_ms: float = 100.0,
) -> SwingPlaneFrame:
    """Compute Functional Swing Plane (FSP) from trajectory data.

    The FSP is the best-fit plane to the clubhead trajectory near impact,
    providing a stable reference for post-simulation force decomposition.

    Args:
        clubhead_trajectory: Clubhead positions over time [m] (N, 3)
        timestamps: Time values [s] (N,)
        impact_time: Time of impact [s]
        window_ms: Fitting window around impact [ms] (default: 100)

    Returns:
        SwingPlaneFrame representing the FSP
    """
    # Convert window to seconds
    window_s = window_ms / 1000.0
    half_window = window_s / 2.0

    # Find points within the fitting window
    mask = (timestamps >= impact_time - half_window) & (
        timestamps <= impact_time + half_window
    )
    window_points = clubhead_trajectory[mask]

    if len(window_points) < 3:
        LOGGER.warning(
            f"Only {len(window_points)} points in FSP window. "
            "Using all available points."
        )
        window_points = clubhead_trajectory

    # Fit plane using SVD (same as SwingPlaneAnalyzer)
    centroid = np.mean(window_points, axis=0)
    centered = window_points - centroid

    _, s, vh = np.linalg.svd(centered)
    normal = vh[2, :]  # Smallest singular value direction

    # Ensure consistent normal direction (pointing "up" relative to swing)
    if normal[2] < 0:
        normal = -normal

    # Compute RMSE
    deviations = np.dot(centered, normal)
    rmse = float(np.sqrt(np.mean(deviations**2)))

    # In-plane axes: X along principal direction, Y perpendicular
    in_plane_x = vh[0, :]  # Largest singular value direction
    in_plane_x = in_plane_x / np.linalg.norm(in_plane_x)

    in_plane_y = np.cross(normal, in_plane_x)
    in_plane_y = in_plane_y / np.linalg.norm(in_plane_y)

    # Grip axis approximation (from first to last point in window)
    if len(window_points) >= 2:
        grip_axis = window_points[-1] - window_points[0]
        grip_length = np.linalg.norm(grip_axis)
        if grip_length > 1e-10:
            grip_axis = grip_axis / grip_length
        else:
            grip_axis = in_plane_x.copy()
    else:
        grip_axis = in_plane_x.copy()

    return SwingPlaneFrame(
        origin=centroid,
        normal=normal,
        in_plane_x=in_plane_x,
        in_plane_y=in_plane_y,
        grip_axis=grip_axis,
        fitting_rmse=rmse,
        fitting_window_ms=window_ms,
    )


def decompose_wrench_in_swing_plane(
    wrench: WrenchInFrame,
    swing_plane: SwingPlaneFrame,
) -> dict[str, float]:
    """Decompose a wrench into swing plane components.

    Args:
        wrench: Wrench in global frame
        swing_plane: Swing plane frame definition

    Returns:
        Dictionary with decomposed components:
            - force_in_plane: Force magnitude tangent to swing plane [N]
            - force_out_of_plane: Force perpendicular to swing plane [N]
            - force_along_grip: Force along grip axis [N]
            - torque_in_plane: Torque tangent to swing plane [N·m]
            - torque_out_of_plane: Torque perpendicular to swing plane [N·m]
            - torque_about_grip: Moment about grip axis [N·m]
    """
    if wrench.frame != ReferenceFrame.GLOBAL:
        LOGGER.warning("Wrench should be in global frame for decomposition")

    f = wrench.force
    tau = wrench.torque

    # Force decomposition
    force_out_of_plane = float(np.dot(f, swing_plane.normal))
    force_in_plane_x = float(np.dot(f, swing_plane.in_plane_x))
    force_in_plane_y = float(np.dot(f, swing_plane.in_plane_y))
    force_in_plane = float(np.sqrt(force_in_plane_x**2 + force_in_plane_y**2))
    force_along_grip = float(np.dot(f, swing_plane.grip_axis))

    # Torque decomposition
    torque_out_of_plane = float(np.dot(tau, swing_plane.normal))
    torque_in_plane_x = float(np.dot(tau, swing_plane.in_plane_x))
    torque_in_plane_y = float(np.dot(tau, swing_plane.in_plane_y))
    torque_in_plane = float(np.sqrt(torque_in_plane_x**2 + torque_in_plane_y**2))
    torque_about_grip = float(np.dot(tau, swing_plane.grip_axis))

    return {
        "force_in_plane": force_in_plane,
        "force_out_of_plane": force_out_of_plane,
        "force_along_grip": force_along_grip,
        "torque_in_plane": torque_in_plane,
        "torque_out_of_plane": torque_out_of_plane,
        "torque_about_grip": torque_about_grip,
    }


class ReferenceFrameTransformer:
    """High-level interface for reference frame transformations.

    Manages transformations between global, local, and swing plane frames
    for force/torque analysis throughout a golf swing simulation.
    """

    def __init__(self) -> None:
        """Initialize the transformer."""
        self.swing_plane: SwingPlaneFrame | None = None
        self.body_rotations: dict[str, np.ndarray] = {}

    def set_swing_plane(self, swing_plane: SwingPlaneFrame) -> None:
        """Set the current swing plane frame.

        Args:
            swing_plane: Swing plane frame definition
        """
        self.swing_plane = swing_plane

    def set_body_rotation(self, body_name: str, R: np.ndarray) -> None:
        """Set the rotation matrix for a body frame.

        Args:
            body_name: Name of the body
            R: Rotation matrix (global to local: v_local = R^T @ v_global)
        """
        self.body_rotations[body_name] = R

    def global_to_local(self, wrench: WrenchInFrame, body_name: str) -> WrenchInFrame:
        """Transform wrench from global frame to body-local frame.

        Args:
            wrench: Wrench in global frame
            body_name: Target body name

        Returns:
            Wrench in local frame
        """
        if body_name not in self.body_rotations:
            raise ValueError(f"No rotation matrix for body '{body_name}'")

        R = self.body_rotations[body_name]
        # Local = R^T @ Global
        return transform_wrench_to_frame(wrench, ReferenceFrame.LOCAL, R.T)

    def global_to_swing_plane(self, wrench: WrenchInFrame) -> WrenchInFrame:
        """Transform wrench from global frame to swing plane frame.

        Args:
            wrench: Wrench in global frame

        Returns:
            Wrench in swing plane frame
        """
        if self.swing_plane is None:
            raise ValueError("Swing plane not set")

        R = compute_rotation_matrix_from_axes(
            self.swing_plane.in_plane_x,
            self.swing_plane.in_plane_y,
            self.swing_plane.normal,
        )
        # Swing = R^T @ Global
        return transform_wrench_to_frame(wrench, ReferenceFrame.SWING_PLANE, R.T)

    def get_swing_plane_decomposition(self, wrench: WrenchInFrame) -> dict[str, float]:
        """Get swing plane component decomposition of a wrench.

        Args:
            wrench: Wrench in global frame

        Returns:
            Dictionary of decomposed components
        """
        if self.swing_plane is None:
            raise ValueError("Swing plane not set")

        return decompose_wrench_in_swing_plane(wrench, self.swing_plane)
