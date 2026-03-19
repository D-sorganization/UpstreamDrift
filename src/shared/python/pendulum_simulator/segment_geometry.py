"""
Geometry utilities for 3D segment rendering.

Provides cross-section generators for different segment representations
(line, cylinder, ellipsoid, tapered) and projection/depth-sorting utilities.

Design by Contract
------------------
- All cross-section functions return (N, 2) arrays of polygon vertices.
- project_3d_to_2d returns shape (2,) and optionally a depth scalar.
- depth_sort_segments sorts from far to near (painter's algorithm).

DRY
---
Cross-section generators share a common interface for QPainter rendering.
"""

from __future__ import annotations

import enum
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SegmentStyle(enum.Enum):
    """Visual style for rendering a pendulum segment."""

    LINE = "line"
    CYLINDER = "cylinder"
    ELLIPSOID = "ellipsoid"
    TAPERED = "tapered"


# ---------------------------------------------------------------------------
# Cross-section generators
# ---------------------------------------------------------------------------


def cylinder_cross_section(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Generate a rectangular cross-section polygon for a constant-radius cylinder.

    Returns 4 corner points of the rectangle (the 2D projection of a cylinder).

    Parameters
    ----------
    start : np.ndarray, shape (2,) — proximal end
    end : np.ndarray, shape (2,) — distal end
    radius : float — cylinder radius

    Returns
    -------
    np.ndarray, shape (4, 2) — corner vertices in CCW order

    Design by Contract
    ------------------
    Pre:  radius > 0
    Post: output shape is (4, 2)
    """
    assert radius > 0, f"radius must be positive, got {radius}"

    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-12:
        # Degenerate segment: return a tiny square
        return np.array(
            [
                start - radius,
                start + np.array([radius, 0]),
                start + radius,
                start - np.array([radius, 0]),
            ]
        )

    # Unit normal perpendicular to segment direction
    d_hat = direction / length
    normal = np.array([-d_hat[1], d_hat[0]])  # 90° rotation

    offset = normal * radius
    corners = np.array(
        [
            start + offset,  # top-left
            end + offset,  # top-right
            end - offset,  # bottom-right
            start - offset,  # bottom-left
        ]
    )

    assert corners.shape == (4, 2)
    return corners


def ellipsoid_cross_section(
    centre: np.ndarray,
    semi_a: float,
    semi_b: float,
    angle: float = 0.0,
    n_points: int = 32,
) -> np.ndarray:
    """Generate polygon vertices for an ellipse (2D projection of ellipsoid).

    Parameters
    ----------
    centre : np.ndarray, shape (2,)
    semi_a : float — semi-axis along the segment direction
    semi_b : float — semi-axis perpendicular to segment
    angle : float — rotation angle (radians, CCW from x-axis)
    n_points : int — number of polygon vertices

    Returns
    -------
    np.ndarray, shape (n_points, 2)

    Design by Contract
    ------------------
    Pre:  semi_a > 0, semi_b > 0, n_points >= 3
    Post: output shape is (n_points, 2)
    """
    assert semi_a > 0, f"semi_a must be positive, got {semi_a}"
    assert semi_b > 0, f"semi_b must be positive, got {semi_b}"
    assert n_points >= 3, f"n_points must be >= 3, got {n_points}"

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Unrotated ellipse
    x = semi_a * np.cos(t)
    y = semi_b * np.sin(t)

    # Rotation
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = cos_a * x - sin_a * y + centre[0]
    y_rot = sin_a * x + cos_a * y + centre[1]

    pts = np.column_stack([x_rot, y_rot])
    assert pts.shape == (n_points, 2)
    return pts


def tapered_cylinder_cross_section(
    start: np.ndarray,
    end: np.ndarray,
    radius_start: float,
    radius_end: float,
) -> np.ndarray:
    """Generate a trapezoidal cross-section for a tapered cylinder.

    Parameters
    ----------
    start : np.ndarray, shape (2,) — proximal (thicker) end
    end : np.ndarray, shape (2,) — distal (thinner) end
    radius_start : float — radius at start
    radius_end : float — radius at end

    Returns
    -------
    np.ndarray, shape (4, 2) — trapezoid vertices

    Design by Contract
    ------------------
    Pre:  radius_start > 0, radius_end > 0
    Post: output shape is (4, 2)
    """
    assert radius_start > 0 and radius_end > 0

    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return cylinder_cross_section(start, end, radius_start)

    d_hat = direction / length
    normal = np.array([-d_hat[1], d_hat[0]])

    corners = np.array(
        [
            start + normal * radius_start,
            end + normal * radius_end,
            end - normal * radius_end,
            start - normal * radius_start,
        ]
    )

    assert corners.shape == (4, 2)
    return corners


# ---------------------------------------------------------------------------
# 3D projection
# ---------------------------------------------------------------------------


def project_3d_to_2d(
    point_3d: np.ndarray,
    tilt: float = 0.0,
    azimuth: float = 0.0,
    return_depth: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Project a 3D point to 2D using isometric-style projection.

    Parameters
    ----------
    point_3d : np.ndarray, shape (3,) — (x, y, z)
    tilt : float — tilt angle in radians (rotation about x-axis)
    azimuth : float — azimuth angle in radians (rotation about y-axis)
    return_depth : bool — if True, also return the depth value

    Returns
    -------
    np.ndarray, shape (2,) — projected (x, y)
    or tuple (np.ndarray shape (2,), float depth) if return_depth=True
    """
    assert point_3d is not None, "point_3d must be provided"
    x, y, z = point_3d

    # Apply azimuth rotation (about y-axis)
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    x1 = ca * x + sa * z
    z1 = -sa * x + ca * z

    # Apply tilt rotation (about x-axis)
    ct, st = np.cos(tilt), np.sin(tilt)
    y1 = ct * y - st * z1
    z2 = st * y + ct * z1

    projected = np.array([x1, y1])

    if return_depth:
        return projected, float(z2)
    return projected


# ---------------------------------------------------------------------------
# Depth sorting (painter's algorithm)
# ---------------------------------------------------------------------------


def depth_sort_segments(
    segments: list[dict],
) -> list[dict]:
    """Sort segments from far to near for correct occlusion rendering.

    Each segment dict must have a 'depth' key (float).
    Segments with larger depth are drawn first (further away).

    Parameters
    ----------
    segments : list of dicts with 'depth' key

    Returns
    -------
    list of dicts sorted by depth (descending = far to near)
    """
    return sorted(segments, key=lambda s: s["depth"], reverse=True)


# ---------------------------------------------------------------------------
# Segment cross-section auto-radius
# ---------------------------------------------------------------------------


def auto_radius_from_mass(mass: float, length: float, scale: float = 0.02) -> float:
    """Compute a reasonable visual radius from segment mass and length.

    radius = scale * sqrt(mass / length) — heuristic for visual appeal.

    Parameters
    ----------
    mass : float — segment mass (kg)
    length : float — segment length (m)
    scale : float — visual scaling factor

    Returns
    -------
    float — radius in metres

    Design by Contract
    ------------------
    Pre:  mass > 0, length > 0, scale > 0
    Post: result > 0
    """
    assert mass > 0 and length > 0 and scale > 0
    r = scale * np.sqrt(mass / length)
    assert r > 0
    return float(r)
