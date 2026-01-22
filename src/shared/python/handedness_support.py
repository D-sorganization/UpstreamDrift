"""Left-Handed Player Support Module.

Guideline B6 Implementation: Left-Handed Player Support.

Provides symmetric model flip functionality including:
- Mirror all kinematic chains about sagittal plane
- Preserve joint conventions (sign flips for asymmetric joints)
- Automatic handedness detection from model metadata
- GUI-ready toggle support for left/right-handed visualization
- Trajectory mirroring with validation

Reference frame convention:
- X: Forward (toward target)
- Y: Left (toward left side of player)
- Z: Up (vertical)

Mirroring is about the XZ plane (sagittal plane),
which flips the Y coordinate.
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

# Threshold for determining if joint axis is aligned with Y-axis (mirror plane normal)
# A value of 0.9 means ~25° from Y-axis is considered "aligned"
Y_AXIS_ALIGNMENT_THRESHOLD = 0.9
# Threshold for detecting significant Y component in prismatic joint axis
Y_AXIS_SIGNIFICANCE_THRESHOLD = 0.1


class Handedness(Enum):
    """Player handedness enumeration."""

    RIGHT_HANDED = auto()
    LEFT_HANDED = auto()


@dataclass
class MirrorTransform:
    """Transformation matrices for mirroring about sagittal plane.

    Attributes:
        position_mirror: Matrix to mirror position vectors (3, 3)
        velocity_mirror: Matrix to mirror velocity vectors (3, 3)
        rotation_mirror: Matrix to transform rotation matrices (3, 3)
        angular_velocity_mirror: Matrix to mirror angular velocity (3, 3)
    """

    position_mirror: np.ndarray
    velocity_mirror: np.ndarray
    rotation_mirror: np.ndarray
    angular_velocity_mirror: np.ndarray


# Standard sagittal plane mirror (flip Y axis)
SAGITTAL_MIRROR = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def create_mirror_transform() -> MirrorTransform:
    """Create standard sagittal plane mirror transform.

    The sagittal plane is the XZ plane (vertical plane through
    the body dividing left and right).

    Returns:
        MirrorTransform with all necessary transformation matrices
    """
    # Position mirrors directly (flip Y)
    position_mirror = SAGITTAL_MIRROR.copy()

    # Velocity mirrors the same way
    velocity_mirror = SAGITTAL_MIRROR.copy()

    # Rotation matrix transformation: R' = M @ R @ M^T
    # For reflection, M = M^T = M^-1
    rotation_mirror = SAGITTAL_MIRROR.copy()

    # Angular velocity is a pseudovector, so it flips opposite
    # ω' = -M @ ω for reflection (or equivalently, M for Y-flip)
    angular_velocity_mirror = SAGITTAL_MIRROR.copy()

    return MirrorTransform(
        position_mirror=position_mirror,
        velocity_mirror=velocity_mirror,
        rotation_mirror=rotation_mirror,
        angular_velocity_mirror=angular_velocity_mirror,
    )


def mirror_position(
    position: np.ndarray,
    transform: MirrorTransform | None = None,
) -> np.ndarray:
    """Mirror a position vector about the sagittal plane.

    Args:
        position: Position vector [m] (3,) or (N, 3)
        transform: Mirror transform (uses default if None)

    Returns:
        Mirrored position vector
    """
    if transform is None:
        transform = create_mirror_transform()

    if position.ndim == 1:
        return np.asarray(transform.position_mirror @ position)
    else:
        # Handle (N, 3) trajectory
        return np.asarray((transform.position_mirror @ position.T).T)


def mirror_velocity(
    velocity: np.ndarray,
    transform: MirrorTransform | None = None,
) -> np.ndarray:
    """Mirror a velocity vector about the sagittal plane.

    Args:
        velocity: Velocity vector [m/s] (3,) or (N, 3)
        transform: Mirror transform (uses default if None)

    Returns:
        Mirrored velocity vector
    """
    if transform is None:
        transform = create_mirror_transform()

    if velocity.ndim == 1:
        return np.asarray(transform.velocity_mirror @ velocity)
    else:
        return np.asarray((transform.velocity_mirror @ velocity.T).T)


def mirror_rotation_matrix(
    rotation: np.ndarray,
    transform: MirrorTransform | None = None,
) -> np.ndarray:
    """Mirror a rotation matrix about the sagittal plane.

    For a reflection matrix M, the mirrored rotation is:
    R' = M @ R @ M

    Since M = M^T = M^-1 for our sagittal mirror.

    Args:
        rotation: Rotation matrix (3, 3)
        transform: Mirror transform (uses default if None)

    Returns:
        Mirrored rotation matrix (3, 3)
    """
    if transform is None:
        transform = create_mirror_transform()

    M = transform.rotation_mirror
    return np.asarray(M @ rotation @ M)


def mirror_angular_velocity(
    omega: np.ndarray,
    transform: MirrorTransform | None = None,
) -> np.ndarray:
    """Mirror an angular velocity vector about the sagittal plane.

    Angular velocity is a pseudovector. Under reflection:
    - Components about axes parallel to mirror plane flip sign
    - Components about the mirror normal preserve sign

    For Y-flip (sagittal plane), X and Z components flip.

    Args:
        omega: Angular velocity [rad/s] (3,) or (N, 3)
        transform: Mirror transform (uses default if None)

    Returns:
        Mirrored angular velocity
    """
    if transform is None:
        transform = create_mirror_transform()

    # Pseudovector behavior: flip X and Z, preserve Y
    # This is equivalent to multiplying by -SAGITTAL_MIRROR
    pseudovector_mirror = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    if omega.ndim == 1:
        return np.asarray(pseudovector_mirror @ omega)
    else:
        return np.asarray((pseudovector_mirror @ omega.T).T)


def mirror_joint_configuration(
    q: np.ndarray,
    joint_types: list[str],
    joint_axes: list[np.ndarray],
) -> np.ndarray:
    """Mirror joint configuration for left-handed model.

    Different joint types require different sign conventions:
    - Revolute about Y: preserve sign (rotation in sagittal plane)
    - Revolute about X or Z: flip sign
    - Prismatic along Y: flip sign
    - Prismatic along X or Z: preserve sign

    Args:
        q: Joint configuration (N,)
        joint_types: List of joint types ("revolute" or "prismatic")
        joint_axes: List of joint axis vectors (N, 3)

    Returns:
        Mirrored joint configuration
    """
    q_mirrored = q.copy()

    for i, (jtype, axis) in enumerate(zip(joint_types, joint_axes, strict=True)):
        axis_np = np.array(axis) if not isinstance(axis, np.ndarray) else axis

        if jtype.lower() == "revolute":
            # Revolute joints: flip if axis is parallel to mirror plane
            # (i.e., has significant X or Z component)
            if abs(axis_np[1]) < Y_AXIS_ALIGNMENT_THRESHOLD:  # Not a Y-axis rotation
                q_mirrored[i] = -q[i]
        elif jtype.lower() == "prismatic":
            # Prismatic joints: flip if axis crosses mirror plane
            # (i.e., has significant Y component)
            if abs(axis_np[1]) > Y_AXIS_SIGNIFICANCE_THRESHOLD:  # Has Y component
                q_mirrored[i] = -q[i]

    return q_mirrored


def mirror_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
    orientations: np.ndarray | None = None,
    angular_velocities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Mirror a complete trajectory for left-handed visualization.

    Args:
        positions: Position trajectory [m] (N, 3)
        velocities: Velocity trajectory [m/s] (N, 3), optional
        orientations: Rotation matrices (N, 3, 3), optional
        angular_velocities: Angular velocity trajectory [rad/s] (N, 3), optional

    Returns:
        Dictionary with mirrored trajectories
    """
    transform = create_mirror_transform()

    result = {
        "positions": mirror_position(positions, transform),
    }

    if velocities is not None:
        result["velocities"] = mirror_velocity(velocities, transform)

    if orientations is not None:
        mirrored_orientations = np.zeros_like(orientations)
        for i in range(len(orientations)):
            mirrored_orientations[i] = mirror_rotation_matrix(
                orientations[i], transform
            )
        result["orientations"] = mirrored_orientations

    if angular_velocities is not None:
        result["angular_velocities"] = mirror_angular_velocity(
            angular_velocities, transform
        )

    return result


def detect_handedness_from_metadata(
    metadata: dict,
) -> Handedness:
    """Detect player handedness from model metadata.

    Looks for common metadata keys indicating handedness:
    - "handedness": "left" or "right"
    - "player_handedness": "L" or "R"
    - "is_left_handed": True/False

    Args:
        metadata: Model metadata dictionary

    Returns:
        Detected Handedness (defaults to RIGHT_HANDED if not found)
    """
    # Check various common keys
    handedness_keys = ["handedness", "player_handedness", "hand"]
    for key in handedness_keys:
        if key in metadata:
            val = str(metadata[key]).lower()
            if val in ("left", "l", "lh", "left_handed", "left-handed"):
                return Handedness.LEFT_HANDED
            elif val in ("right", "r", "rh", "right_handed", "right-handed"):
                return Handedness.RIGHT_HANDED

    # Check boolean flags
    if metadata.get("is_left_handed", False):
        return Handedness.LEFT_HANDED

    # Default to right-handed
    LOGGER.debug("No handedness metadata found, defaulting to right-handed")
    return Handedness.RIGHT_HANDED


def validate_mirror_trajectory(
    original_positions: np.ndarray,
    mirrored_positions: np.ndarray,
) -> dict[str, float | bool]:
    """Validate that mirrored trajectory is geometrically correct.

    Checks:
    - Y coordinates are flipped
    - X and Z coordinates are preserved
    - Path length is preserved

    Args:
        original_positions: Original trajectory (N, 3)
        mirrored_positions: Mirrored trajectory (N, 3)

    Returns:
        Dictionary with validation results
    """
    # Check coordinate transformations
    y_flipped = np.allclose(original_positions[:, 1], -mirrored_positions[:, 1])
    x_preserved = np.allclose(original_positions[:, 0], mirrored_positions[:, 0])
    z_preserved = np.allclose(original_positions[:, 2], mirrored_positions[:, 2])

    # Check path length preservation
    original_path_length = np.sum(
        np.linalg.norm(np.diff(original_positions, axis=0), axis=1)
    )
    mirrored_path_length = np.sum(
        np.linalg.norm(np.diff(mirrored_positions, axis=0), axis=1)
    )
    path_length_preserved = np.isclose(
        original_path_length, mirrored_path_length, rtol=1e-10
    )

    all_valid = y_flipped and x_preserved and z_preserved and path_length_preserved

    return {
        "valid": all_valid,
        "y_flipped": y_flipped,
        "x_preserved": x_preserved,
        "z_preserved": z_preserved,
        "path_length_preserved": path_length_preserved,
        "original_path_length": float(original_path_length),
        "mirrored_path_length": float(mirrored_path_length),
    }


def validate_energy_conservation(
    original_velocities: np.ndarray,
    mirrored_velocities: np.ndarray,
    masses: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """Validate that kinetic energy is preserved under reflection.

    Args:
        original_velocities: Original velocity trajectory (N, 3)
        mirrored_velocities: Mirrored velocity trajectory (N, 3)
        masses: Mass at each time step (N,), optional

    Returns:
        Dictionary with validation results
    """
    if masses is None:
        masses = np.ones(len(original_velocities))

    # Compute kinetic energy at each timestep
    original_ke = 0.5 * masses * np.sum(original_velocities**2, axis=1)
    mirrored_ke = 0.5 * masses * np.sum(mirrored_velocities**2, axis=1)

    # Total energy should be preserved
    total_original = np.sum(original_ke)
    total_mirrored = np.sum(mirrored_ke)

    energy_preserved = np.isclose(total_original, total_mirrored, rtol=1e-10)

    return {
        "valid": energy_preserved,
        "original_total_ke": float(total_original),
        "mirrored_total_ke": float(total_mirrored),
        "max_difference": float(np.max(np.abs(original_ke - mirrored_ke))),
    }


class HandednessConverter:
    """Converter for left/right handed model visualization.

    Provides a high-level interface for converting between
    right-handed and left-handed representations.
    """

    def __init__(self, source_handedness: Handedness = Handedness.RIGHT_HANDED) -> None:
        """Initialize the handedness converter.

        Args:
            source_handedness: The handedness of the source model
        """
        self.source_handedness = source_handedness
        self.transform = create_mirror_transform()

    def convert_to(
        self,
        target_handedness: Handedness,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        orientations: np.ndarray | None = None,
        angular_velocities: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Convert trajectory to target handedness.

        Args:
            target_handedness: Desired output handedness
            positions: Position trajectory (N, 3)
            velocities: Velocity trajectory (N, 3), optional
            orientations: Rotation matrices (N, 3, 3), optional
            angular_velocities: Angular velocities (N, 3), optional

        Returns:
            Dictionary with converted trajectories
        """
        if target_handedness == self.source_handedness:
            # No conversion needed
            result = {"positions": positions.copy()}
            if velocities is not None:
                result["velocities"] = velocities.copy()
            if orientations is not None:
                result["orientations"] = orientations.copy()
            if angular_velocities is not None:
                result["angular_velocities"] = angular_velocities.copy()
            return result

        # Need to mirror
        return mirror_trajectory(
            positions, velocities, orientations, angular_velocities
        )

    def is_conversion_needed(self, target_handedness: Handedness) -> bool:
        """Check if conversion is needed for target handedness.

        Args:
            target_handedness: Desired output handedness

        Returns:
            True if mirroring is required
        """
        return target_handedness != self.source_handedness
