"""Rigid swing plane calibration and coordinate transformations (TB-03 #10588).

Estimates one single rigid plane, basis (u, v, n) and origin for a declared capture
window from weighted observations. Handles degeneracy and reflection explicitly.
Guarantees exact round-trip unprojection and transforms world gravity into the plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DegeneratePlaneError(ValueError):
    """Raised when point cloud is collinear, insufficient, or rank < 2."""


@dataclass(frozen=True)
class PlaneGravity:
    """Gravity vector decomposed into swing plane coordinates."""

    g_u: float
    g_v: float
    g_normal: float
    inclination_deg: float


@dataclass(frozen=True)
class PlaneFitDiagnostics:
    """Geometric diagnostics for a calibrated swing plane."""

    rmse_m: float
    max_residual_m: float
    sample_count: int
    singular_values: tuple[float, float, float]
    per_marker_rmse_m: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RigidSwingPlane:
    """Rigid 3D swing plane with orthonormal basis (u, v, n) and origin P0.

    Attributes:
        origin: 3D origin point (e.g. shoulder/hub or weighted centroid) [m].
        normal: Unit normal vector n (perpendicular to plane).
        u_axis: Unit in-plane horizontal-ish axis.
        v_axis: Unit in-plane vertical-ish axis.
        inclination_deg: Tilt angle of plane relative to vertical [degrees].
    """

    origin: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    inclination_deg: float = 0.0

    def __post_init__(self) -> None:
        """Validate shapes and orthonormality."""
        orig = np.asarray(self.origin, dtype=float)
        n = np.asarray(self.normal, dtype=float)
        u = np.asarray(self.u_axis, dtype=float)
        v = np.asarray(self.v_axis, dtype=float)
        if orig.shape != (3,) or n.shape != (3,) or u.shape != (3,) or v.shape != (3,):
            raise ValueError("All vectors must have shape (3,)")
        # Check right-handedness: u x v = n
        cross_uv = np.cross(u, v)
        if not np.allclose(cross_uv, n, atol=1e-4):
            raise ValueError(
                f"Basis must be right-handed orthonormal (u x v == n), got dot={float(np.dot(cross_uv, n)):.4f}"
            )

    def project_point(self, p_3d: np.ndarray) -> tuple[np.ndarray, float]:
        """Project a single 3D point onto the plane.

        Returns:
            Tuple of ((u, v) in-plane coordinates, out-of-plane residual w).
        """
        pts = np.asarray(p_3d, dtype=float)
        delta = pts - self.origin
        u = float(np.dot(delta, self.u_axis))
        v = float(np.dot(delta, self.v_axis))
        w = float(np.dot(delta, self.normal))
        return np.array([u, v]), w

    def project_points(self, pts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project an array of (N, 3) 3D points onto the plane.

        Returns:
            Tuple of ((N, 2) in-plane coordinates, (N,) out-of-plane residuals).
        """
        arr = np.asarray(pts_3d, dtype=float)
        if arr.ndim == 1:
            p2d, w_single = self.project_point(arr)
            return p2d[np.newaxis, :], np.array([w_single], dtype=float)
        delta = arr - self.origin
        u = np.dot(delta, self.u_axis)
        v = np.dot(delta, self.v_axis)
        w = np.asarray(np.dot(delta, self.normal), dtype=float)
        return np.column_stack([u, v]), w

    def unproject_point(self, p_2d: np.ndarray, w: float = 0.0) -> np.ndarray:
        """Map 2D in-plane point (u, v) and out-of-plane residual w back to 3D world."""
        pt = np.asarray(p_2d, dtype=float)
        return self.origin + pt[0] * self.u_axis + pt[1] * self.v_axis + w * self.normal

    def unproject_points(
        self, pts_2d: np.ndarray, w: np.ndarray | float | None = None
    ) -> np.ndarray:
        """Map (N, 2) in-plane points back to (N, 3) 3D coordinates."""
        arr = np.asarray(pts_2d, dtype=float)
        if arr.ndim == 1:
            w_val = float(w) if w is not None and not isinstance(w, np.ndarray) else 0.0
            return self.unproject_point(arr, w_val)
        n = arr.shape[0]
        if w is None:
            w_arr = np.zeros(n, dtype=float)
        elif isinstance(w, (int, float)):
            w_arr = np.full(n, float(w))
        else:
            w_arr = np.asarray(w, dtype=float)
        return (
            self.origin[np.newaxis, :]
            + arr[:, 0:1] * self.u_axis[np.newaxis, :]
            + arr[:, 1:2] * self.v_axis[np.newaxis, :]
            + w_arr[:, np.newaxis] * self.normal[np.newaxis, :]
        )

    def transform_gravity(self, gravity_world: np.ndarray) -> PlaneGravity:
        """Decompose 3D world gravity into in-plane (u, v) and out-of-plane components."""
        g = np.asarray(gravity_world, dtype=float)
        gu = float(np.dot(g, self.u_axis))
        gv = float(np.dot(g, self.v_axis))
        gn = float(np.dot(g, self.normal))
        # Inclination: angle between plane normal and vertical
        g_mag = float(np.linalg.norm(g))
        if g_mag > 1e-6:
            cos_tilt = abs(gn) / g_mag
            inc_deg = math.degrees(math.acos(min(max(cos_tilt, 0.0), 1.0)))
        else:
            inc_deg = 0.0
        return PlaneGravity(
            g_u=gu,
            g_v=gv,
            g_normal=gn,
            inclination_deg=inc_deg,
        )


def _compute_plane_axes(
    normal: np.ndarray, world_up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute right-handed in-plane basis vectors (u, v) from normal and world up."""
    # Project world up onto the plane to form v_axis
    v_cand = world_up - float(np.dot(world_up, normal)) * normal
    v_norm = float(np.linalg.norm(v_cand))
    if v_norm < 1e-4:
        # Normal is parallel to world_up, fallback to world forward [0, 0, 1]
        fallback = np.array([0.0, 0.0, 1.0])
        v_cand = fallback - float(np.dot(fallback, normal)) * normal
        v_norm = float(np.linalg.norm(v_cand))
    v = v_cand / v_norm

    # Form u = v x normal so that u x v = normal (right-handed)
    # Note: (v x n) x v = v x (n x (-v)) = n (since u x v = (v x n) x v = n)
    u = np.cross(v, normal)
    u = u / float(np.linalg.norm(u))
    # Verify u x v == normal
    if np.dot(np.cross(u, v), normal) < 0:
        u = -u
    return u, v


def _filter_valid_samples(
    points: np.ndarray, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Filter out non-finite rows and zero-weight samples."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise DegeneratePlaneError(
            f"Points array must have shape (N, 3), got {pts.shape}"
        )
    if weights is None:
        w = np.ones(pts.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (pts.shape[0],):
            raise ValueError("Weights must match number of points")
    finite_mask = np.isfinite(pts).all(axis=-1) & np.isfinite(w) & (w > 0.0)
    valid_pts = pts[finite_mask]
    valid_w = w[finite_mask]
    if len(valid_pts) < 3:
        raise DegeneratePlaneError(
            f"At least 3 non-collinear points required; found {len(valid_pts)} valid"
        )
    return valid_pts, valid_w


def fit_rigid_swing_plane(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    world_up: np.ndarray | None = None,
    reference_normal: np.ndarray | None = None,
    marker_names: list[str] | None = None,
) -> tuple[RigidSwingPlane, PlaneFitDiagnostics]:
    """Fit a single rigid swing plane from weighted 3D points using SVD."""
    valid_pts, valid_w = _filter_valid_samples(points, weights)
    up = (
        np.array([0.0, 1.0, 0.0], dtype=float)
        if world_up is None
        else np.asarray(world_up, dtype=float)
    )

    # Weighted centroid
    total_w = float(np.sum(valid_w))
    origin = np.sum(valid_pts * valid_w[:, np.newaxis], axis=0) / total_w

    # Centered weighted coordinates
    centered = valid_pts - origin
    weighted_centered = centered * np.sqrt(valid_w)[:, np.newaxis]

    # SVD
    _, s, vh = np.linalg.svd(weighted_centered, full_matrices=False)
    s_vals = (
        (float(s[0]), float(s[1]), float(s[2]))
        if len(s) >= 3
        else (float(s[0]), float(s[1]), 0.0)
    )

    # Degeneracy check: rank < 2
    if len(s) < 2 or s[1] < 1e-4 * (s[0] + 1e-9) or s[0] < 1e-6:
        raise DegeneratePlaneError(
            f"Point cloud is collinear or rank < 2; singular values={s_vals}"
        )

    normal = vh[2, :].copy()
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        raise DegeneratePlaneError(
            "Failed to compute normal vector from singular vectors"
        )
    normal /= normal_norm

    # Reflection / sign consistency: align with reference_normal if provided
    if reference_normal is not None:
        ref_n = np.asarray(reference_normal, dtype=float)
        if np.dot(normal, ref_n) < 0.0:
            normal = -normal

    # Construct right-handed in-plane axes
    u_axis, v_axis = _compute_plane_axes(normal, up)

    # Calculate residuals
    residuals = np.abs(np.dot(centered, normal))
    rmse = float(np.sqrt(np.sum(valid_w * residuals**2) / total_w))
    max_res = float(np.max(residuals))

    # Per-marker residuals if names provided
    per_marker: dict[str, float] = {}
    if marker_names and len(marker_names) == len(points):
        # Compute subset residuals
        for name in set(marker_names):
            mask = [
                i
                for i, n in enumerate(marker_names)
                if n == name and np.isfinite(points[i]).all()
            ]
            if mask:
                sub_res = np.abs(np.dot(points[mask] - origin, normal))
                per_marker[name] = float(np.sqrt(np.mean(sub_res**2)))

    # Compute tilt angle from vertical
    cos_tilt = abs(float(np.dot(normal, up)))
    inc_deg = math.degrees(math.acos(min(max(cos_tilt, 0.0), 1.0)))

    plane = RigidSwingPlane(
        origin=origin,
        normal=normal,
        u_axis=u_axis,
        v_axis=v_axis,
        inclination_deg=inc_deg,
    )
    diag = PlaneFitDiagnostics(
        rmse_m=rmse,
        max_residual_m=max_res,
        sample_count=len(valid_pts),
        singular_values=s_vals,
        per_marker_rmse_m=per_marker,
    )
    return plane, diag
