"""Calibration of fixed link geometry, club dimensions, and marker offsets (TB-03 #10588).

Calibrates positive bounded link lengths (arm/upper segment, shaft/club) from declared
observation windows, reports rigidity/variance diagnostics, and freezes unidentifiable
mass and inertia parameters to physical priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DEFAULT_ARM_CENTER_OF_MASS_RATIO,
    DEFAULT_ARM_LENGTH_M,
    DEFAULT_ARM_MASS_KG,
    DEFAULT_CLUBHEAD_MASS_KG,
    DEFAULT_SHAFT_COM_RATIO,
    DEFAULT_SHAFT_LENGTH_M,
    DEFAULT_SHAFT_MASS_KG,
)

logger = logging.getLogger(__name__)

DEFAULT_ARM_BOUNDS_M = (0.3, 1.2)
DEFAULT_CLUB_BOUNDS_M = (0.5, 1.8)


@dataclass(frozen=True)
class CalibratedGeometry:
    """Fixed geometric parameters for reduced-order double pendulum models.

    Attributes:
        arm_length_m: Effective upper segment length (hub to grip/wrist) [m].
        club_length_m: Effective lower segment length (grip to clubhead) [m].
        arm_length_std_m: Standard deviation of arm length across valid frames [m].
        club_length_std_m: Standard deviation of club length across valid frames [m].
        shaft_mass_kg: Shaft + grip mass [kg] (frozen prior).
        clubhead_mass_kg: Clubhead mass [kg] (frozen prior).
        arm_mass_kg: Upper segment mass [kg] (frozen prior).
        shaft_com_ratio: Shaft COM location from grip as ratio in (0, 1].
        arm_com_ratio: Arm COM location from shoulder as ratio in (0, 1].
        is_mass_frozen: True indicating mass is not estimated from kinematics.
        is_inertia_frozen: True indicating inertia is not estimated from kinematics.
        marker_offsets: Calibrated 3D displacement vectors from landmarks to markers.
    """

    arm_length_m: float
    club_length_m: float
    arm_length_std_m: float = 0.0
    club_length_std_m: float = 0.0
    shaft_mass_kg: float = DEFAULT_SHAFT_MASS_KG
    clubhead_mass_kg: float = DEFAULT_CLUBHEAD_MASS_KG
    arm_mass_kg: float = DEFAULT_ARM_MASS_KG
    shaft_com_ratio: float = DEFAULT_SHAFT_COM_RATIO
    arm_com_ratio: float = DEFAULT_ARM_CENTER_OF_MASS_RATIO
    is_mass_frozen: bool = True
    is_inertia_frozen: bool = True
    marker_offsets: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def _extract_finite_pairwise_distances(
    p1: np.ndarray, p2: np.ndarray, name: str
) -> np.ndarray:
    """Compute Euclidean distances between two trajectory arrays for finite frames."""
    arr1 = np.asarray(p1, dtype=float)
    arr2 = np.asarray(p2, dtype=float)
    if arr1.shape != arr2.shape or arr1.shape[-1] != 3:
        raise ValueError(
            f"Trajectory shapes must match and have 3 coordinates, got {arr1.shape} and {arr2.shape}"
        )
    diff = arr1 - arr2
    valid = np.isfinite(diff).all(axis=-1)
    if not np.any(valid):
        raise ValueError(f"No finite frames found for {name} distance calculation")
    dists = np.linalg.norm(diff[valid], axis=-1)
    return dists


def calibrate_link_geometry(
    hub_traj: np.ndarray,
    grip_traj: np.ndarray,
    head_traj: np.ndarray,
    *,
    arm_bounds: tuple[float, float] = DEFAULT_ARM_BOUNDS_M,
    club_bounds: tuple[float, float] = DEFAULT_CLUB_BOUNDS_M,
    shaft_mass_kg: float = DEFAULT_SHAFT_MASS_KG,
    clubhead_mass_kg: float = DEFAULT_CLUBHEAD_MASS_KG,
    arm_mass_kg: float = DEFAULT_ARM_MASS_KG,
    marker_offsets: dict[str, tuple[float, float, float]] | None = None,
) -> CalibratedGeometry:
    """Calibrate positive bounded fixed link lengths from trajectory observations."""
    arm_dists = _extract_finite_pairwise_distances(grip_traj, hub_traj, "arm")
    club_dists = _extract_finite_pairwise_distances(head_traj, grip_traj, "club")

    arm_mean = float(np.mean(arm_dists))
    arm_std = float(np.std(arm_dists))
    club_mean = float(np.mean(club_dists))
    club_std = float(np.std(club_dists))

    # Validate bounds
    if not (arm_bounds[0] <= arm_mean <= arm_bounds[1]):
        raise ValueError(
            f"Arm length {arm_mean:.4f} m outside physiological bounds {arm_bounds}"
        )
    if not (club_bounds[0] <= club_mean <= club_bounds[1]):
        raise ValueError(
            f"Club length {club_mean:.4f} m outside physical bounds {club_bounds}"
        )

    return CalibratedGeometry(
        arm_length_m=arm_mean,
        club_length_m=club_mean,
        arm_length_std_m=arm_std,
        club_length_std_m=club_std,
        shaft_mass_kg=shaft_mass_kg,
        clubhead_mass_kg=clubhead_mass_kg,
        arm_mass_kg=arm_mass_kg,
        is_mass_frozen=True,
        is_inertia_frozen=True,
        marker_offsets=marker_offsets or {},
    )
