"""3D-to-2D target projection and calibrated swing plane estimation (TB-03 #10588).

Projects 3D MultiSourceTarget or ClubTarget onto a rigid calibrated swing plane,
preserving geometric projection residuals and rigid frame transformations.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
import numpy as np

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import MultiSourceTarget


@dataclass(frozen=True)
class GeometricProjectionResidual:
    """Geometric projection error diagnostics for a fitted plane."""

    rmse: float
    max_deviation: float
    signed_deviations: np.ndarray

    def __post_init__(self) -> None:
        if self.rmse < 0.0:
            raise ValueError("RMSE must be non-negative")
        if self.max_deviation < 0.0:
            raise ValueError("max_deviation must be non-negative")


@dataclass(frozen=True)
class CalibratedSwingPlane:
    """Rigid 3D-to-2D swing plane definition with explicit SE(3) transform."""

    origin: np.ndarray
    basis: np.ndarray  # Shape (3, 3) where columns are [u, v, n] in SO(3)
    transform_world_to_plane: np.ndarray  # Shape (4, 4)
    transform_plane_to_world: np.ndarray  # Shape (4, 4)
    inclination_deg: float
    azimuth_deg: float
    residual: GeometricProjectionResidual

    @property
    def u_axis(self) -> np.ndarray:
        """First in-plane orthogonal axis."""
        return self.basis[:, 0]

    @property
    def v_axis(self) -> np.ndarray:
        """Second in-plane orthogonal axis."""
        return self.basis[:, 1]

    @property
    def normal(self) -> np.ndarray:
        """Out-of-plane normal vector n = u x v."""
        return self.basis[:, 2]

    def project_points_to_plane(self, pts: np.ndarray) -> np.ndarray:
        """Transform 3D points in world coordinates into plane coordinates [u, v, n]."""
        pts_arr = np.asarray(pts, dtype=float)
        diff = pts_arr - self.origin
        # Plane coords: [x_plane, y_plane, z_plane] = diff @ basis
        return diff @ self.basis

    def reconstruct_points_from_plane(self, plane_pts: np.ndarray) -> np.ndarray:
        """Reconstruct 3D world coordinates from plane coordinates [u, v, n]."""
        coords = np.asarray(plane_pts, dtype=float)
        # Reconstruct: origin + coords @ basis.T
        return self.origin + coords @ self.basis.T

    def transform_vector_to_plane(self, vec: np.ndarray) -> np.ndarray:
        """Transform a 3D direction/vector (e.g. gravity) into the plane frame."""
        v_arr = np.asarray(vec, dtype=float)
        return self.basis.T @ v_arr


def _validate_points_and_weights(
    points: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and filter point cloud and weights, checking finiteness and sample count."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    valid_mask = np.isfinite(pts).all(axis=1)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.shape != (len(pts),):
            raise ValueError("Weights must match number of points")
        valid_mask &= np.isfinite(w_arr) & (w_arr > 0)
        pts_valid = pts[valid_mask]
        w_valid = w_arr[valid_mask]
    else:
        pts_valid = pts[valid_mask]
        w_valid = np.ones(len(pts_valid), dtype=float)

    if len(pts_valid) < 3:
        raise ValueError(
            f"At least 3 valid non-collinear points required, got {len(pts_valid)}"
        )
    w_norm = w_valid / np.sum(w_valid)
    return pts_valid, w_norm


def estimate_swing_plane(
    points: np.ndarray,
    weights: np.ndarray | None = None,
) -> CalibratedSwingPlane:
    """Estimate one rigid swing plane and basis from valid weighted observations.

    Handles degeneracy (collinear points) and reflection (enforces det = +1 SO(3)).
    """
    pts_valid, w_norm = _validate_points_and_weights(points, weights)
    centroid = np.sum(pts_valid * w_norm[:, np.newaxis], axis=0)
    centered = pts_valid - centroid
    weighted_centered = centered * np.sqrt(w_norm[:, np.newaxis])

    if np.linalg.matrix_rank(weighted_centered) < 2:
        raise ValueError("Degenerate point cloud: points are collinear (rank < 2)")

    _, _, vh = np.linalg.svd(weighted_centered, full_matrices=False)
    u = vh[0] / np.linalg.norm(vh[0])
    v = vh[1] / np.linalg.norm(vh[1])
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)

    basis = np.column_stack([u, v, n])
    if np.linalg.det(basis) < 0:
        # Proper right-handed rotation matrix: flip v and n if reflected
        v = -v
        n = np.cross(u, v)
        basis = np.column_stack([u, v, n])

    # Enforce SO(3) determinant tolerance
    det = float(np.linalg.det(basis))
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"Failed to construct valid SO(3) basis, det={det}")

    # Build SE(3) transforms
    rot = basis.T  # world to plane rotation
    trans = -rot @ centroid
    t_w2p = np.eye(4)
    t_w2p[:3, :3] = rot
    t_w2p[:3, 3] = trans

    t_p2w = np.eye(4)
    t_p2w[:3, :3] = basis
    t_p2w[:3, 3] = centroid

    # Compute residuals across input points
    deviations = np.dot(pts_valid - centroid, n)
    rmse = float(np.sqrt(np.mean(deviations**2)))
    max_dev = float(np.max(np.abs(deviations)))
    res = GeometricProjectionResidual(
        rmse=rmse,
        max_deviation=max_dev,
        signed_deviations=deviations,
    )

    # Compute inclination from vertical
    vert_dot = abs(float(n[1])) if abs(n[1]) > abs(n[2]) else abs(float(n[2]))
    inclination_deg = float(math.degrees(math.acos(np.clip(vert_dot, 0.0, 1.0))))
    azimuth_deg = float(math.degrees(math.atan2(n[0], n[2])))

    return CalibratedSwingPlane(
        origin=centroid,
        basis=basis,
        transform_world_to_plane=t_w2p,
        transform_plane_to_world=t_p2w,
        inclination_deg=inclination_deg,
        azimuth_deg=azimuth_deg,
        residual=res,
    )


def project_to_calibrated_plane(
    target: MultiSourceTarget | ClubTarget,
    plane: CalibratedSwingPlane,
) -> ClubTarget:
    """Project a 3D target onto the calibrated swing plane."""
    club = target.club if isinstance(target, MultiSourceTarget) else target
    if club is None:
        raise ValueError("target.club must be set")

    def _proj(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2 or arr.shape[-1] != 3:
            return arr
        # Transform to plane, zero the out-of-plane component, and transform back
        plane_coords = plane.project_points_to_plane(arr)
        plane_coords[:, 2] = 0.0
        return plane.reconstruct_points_from_plane(plane_coords)

    return dataclasses.replace(
        club,
        butt=_proj(club.butt),
        clubhead=_proj(club.clubhead),
    )


def project_to_2d(target: MultiSourceTarget | ClubTarget) -> ClubTarget:
    """Project a 3D target onto an estimated rigid swing plane."""
    club = target.club if isinstance(target, MultiSourceTarget) else target
    if club is None:
        raise ValueError("target.club must be set")

    # Joint observation of clubhead and butt to determine the plane
    joint_pts = np.vstack([club.butt, club.clubhead])
    plane = estimate_swing_plane(joint_pts)
    return project_to_calibrated_plane(club, plane)
