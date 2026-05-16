"""Hand-path plane computation for shallowing analysis (Phase 1, epic #5422).

Computes the lead hand 3D trajectory from simulation frames and fits a
best-fit plane using Singular Value Decomposition (SVD).

Design by Contract:
    - extract_lead_hand_trajectory: frames must be non-empty, each frame must
      expose a joint_positions mapping with a recognised lead-hand key.
    - compute_hand_path_plane: trajectory must have >= 3 rows; returned normal
      lies in the upward hemisphere (normal[2] >= 0 by convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)

# Candidate keys searched in priority order when resolving the lead hand marker.
_LEAD_HAND_KEYS: tuple[str, ...] = ("lead_hand", "left_wrist", "wrist_L")


@dataclass
class Plane3D:
    """Best-fit plane through a set of 3-D points.

    Attributes:
        normal: Unit normal vector of the plane, shape (3,).  Always lies in
            the upward hemisphere (``normal[2] >= 0``).
        point_on_plane: Centroid of the input points, shape (3,).
        residuals: Mean absolute distance from each input point to the plane.
    """

    normal: np.ndarray  # shape (3,), unit vector
    point_on_plane: np.ndarray  # shape (3,), centroid of input points
    residuals: float  # mean absolute distance of input points from fitted plane


def _resolve_lead_hand_key(joint_positions: dict[str, Any]) -> str:
    """Return the first recognised lead-hand key present in *joint_positions*.

    Design by Contract:
        Precondition: joint_positions is non-empty (caller must ensure).
        Postcondition: returned key is in joint_positions.

    Args:
        joint_positions: Mapping of marker/joint names to 3-D positions.

    Returns:
        The matched key string.

    Raises:
        ValueError: If none of the candidate keys are found.
    """
    for candidate in _LEAD_HAND_KEYS:
        if candidate in joint_positions:
            _LOG.debug("Resolved lead hand key to %r", candidate)
            return candidate
    available = list(joint_positions.keys())
    raise ValueError(
        f"Lead hand marker not found in frame. "
        f"Tried {list(_LEAD_HAND_KEYS)!r}. "
        f"Available: {available}"
    )


def extract_lead_hand_trajectory(sim_frames: list[Any]) -> np.ndarray:
    """Extract lead hand 3-D positions from a sequence of simulation frames.

    Each frame is expected to expose a ``joint_positions`` attribute that
    is a mapping from marker/joint name strings to 3-element position
    arrays.  The function searches for the lead-hand marker in priority
    order: ``"lead_hand"`` > ``"left_wrist"`` > ``"wrist_L"``.

    Design by Contract:
        Preconditions:
            - ``sim_frames`` must not be empty.
            - Each frame must have ``joint_positions`` with at least one of
              the recognised lead-hand keys.
        Postconditions:
            - Returns an array of shape ``(N, 3)`` where
              ``N == len(sim_frames)``.

    Args:
        sim_frames: Ordered list of simulation frame objects.

    Returns:
        NumPy array of shape ``(N, 3)`` containing lead-hand world positions.

    Raises:
        ValueError: If ``sim_frames`` is empty.
        ValueError: If the lead hand marker is not found in a frame.
    """
    if len(sim_frames) == 0:
        raise ValueError("sim_frames must not be empty")

    # Resolve the key from the first frame; apply consistently to all frames.
    first_frame = sim_frames[0]
    key = _resolve_lead_hand_key(first_frame.joint_positions)

    positions: list[np.ndarray] = []
    for frame in sim_frames:
        pos = np.asarray(frame.joint_positions[key], dtype=float)
        positions.append(pos)

    trajectory = np.stack(positions, axis=0)
    _LOG.debug(
        "Extracted lead hand trajectory: %d frames, key=%r", len(sim_frames), key
    )
    return trajectory


def compute_hand_path_plane(trajectory: np.ndarray) -> Plane3D:
    """Compute the SVD best-fit plane through a 3-D trajectory.

    Uses Singular Value Decomposition on the mean-centred trajectory.  The
    plane normal corresponds to the singular vector with the *smallest*
    singular value (i.e. the direction of minimum variance).

    Convention: the normal is flipped if necessary so that ``normal[2] >= 0``
    (upward hemisphere).

    Design by Contract:
        Preconditions:
            - ``trajectory`` must have shape ``(N, 3)`` with ``N >= 3``.
        Postconditions:
            - ``returned.normal`` is a unit vector with ``normal[2] >= 0``.
            - ``returned.point_on_plane`` equals ``trajectory.mean(axis=0)``.
            - ``returned.residuals >= 0``.

    Args:
        trajectory: Array of shape ``(N, 3)`` containing 3-D positions.

    Returns:
        :class:`Plane3D` with the fitted normal, centroid, and mean residual.

    Raises:
        ValueError: If ``trajectory`` has fewer than 3 rows.
    """
    n_points = len(trajectory)
    if n_points < 3:
        raise ValueError(
            f"compute_hand_path_plane requires >= 3 points, got {n_points}"
        )

    centroid: np.ndarray = trajectory.mean(axis=0)
    centered: np.ndarray = trajectory - centroid

    # Full SVD — Vt has shape (3, 3); last row = direction of minimum variance.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    normal: np.ndarray = vt[-1].copy()

    # Upward-hemisphere convention: ensure normal[2] >= 0.
    if normal[2] < 0:
        normal = -normal

    # Normalise for safety (SVD already produces unit vectors but float errors
    # can accumulate if callers pre-process the data).
    norm_magnitude = float(np.linalg.norm(normal))
    if norm_magnitude > 0:
        normal = normal / norm_magnitude

    residuals = float(np.abs(centered @ normal).mean())

    _LOG.debug("Hand-path plane fitted: normal=%s, residuals=%.4g", normal, residuals)
    return Plane3D(
        normal=normal,
        point_on_plane=centroid,
        residuals=residuals,
    )
