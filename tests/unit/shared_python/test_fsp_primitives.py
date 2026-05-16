"""Python-level tests for the FSP Rust extension (issue #5502).

These tests are skipped when the Rust extension has not been compiled
(``maturin develop`` not yet run), so CI passes even in pure-Python
environments.

The module is exposed as ``upstream_physics`` after ``maturin develop``
inside ``rust_core/upstream-physics/``.
"""

import pytest

try:
    import upstream_physics as fsp_mod

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")

# ---------------------------------------------------------------------------
# calculate_fsp
# ---------------------------------------------------------------------------


def test_horizontal_plane_slope() -> None:
    """Points on z=0 → slope should be ≈ 0°."""
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    plane = fsp_mod.calculate_fsp(points)
    assert abs(fsp_mod.fsp_slope_deg(plane)) < 0.01


def test_calculate_fsp_returns_plane_object() -> None:
    """calculate_fsp returns an object with normal and centroid attributes."""
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    plane = fsp_mod.calculate_fsp(points)
    assert hasattr(plane, "normal")
    assert hasattr(plane, "centroid")
    assert len(plane.normal) == 3
    assert len(plane.centroid) == 3


def test_normal_is_unit_vector() -> None:
    """The normal returned by calculate_fsp must be a unit vector."""
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    plane = fsp_mod.calculate_fsp(points)
    n = plane.normal
    length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
    assert abs(length - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# fsp_slope_deg
# ---------------------------------------------------------------------------


def test_vertical_plane_slope() -> None:
    """Points on x=0 (vertical plane) → slope ≈ 90°."""
    points = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    plane = fsp_mod.calculate_fsp(points)
    assert abs(fsp_mod.fsp_slope_deg(plane) - 90.0) < 0.01


# ---------------------------------------------------------------------------
# point_to_fsp_distance
# ---------------------------------------------------------------------------


def test_point_on_plane_distance() -> None:
    """A point that lies on the z=0 plane should have distance ≈ 0."""
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    plane = fsp_mod.calculate_fsp(points)
    dist = fsp_mod.point_to_fsp_distance([0.5, 0.5, 0.0], plane)
    assert abs(dist) < 1e-6


def test_point_above_plane_distance() -> None:
    """Point at (0,0,1) above z=0 plane should have |distance| ≈ 1."""
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    plane = fsp_mod.calculate_fsp(points)
    dist = fsp_mod.point_to_fsp_distance([0.0, 0.0, 1.0], plane)
    assert abs(abs(dist) - 1.0) < 1e-6


def test_distance_sign_convention() -> None:
    """Points on opposite sides of the plane should have opposite-sign distances."""
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    plane = fsp_mod.calculate_fsp(points)
    above = fsp_mod.point_to_fsp_distance([0.0, 0.0, 1.0], plane)
    below = fsp_mod.point_to_fsp_distance([0.0, 0.0, -1.0], plane)
    assert above * below < 0.0
    assert abs(abs(above) - abs(below)) < 1e-6


# ---------------------------------------------------------------------------
# fsp_direction_deg
# ---------------------------------------------------------------------------


def test_direction_along_x_axis() -> None:
    """Target along +X on horizontal plane → azimuth ≈ 0°."""
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    plane = fsp_mod.calculate_fsp(points)
    direction = fsp_mod.fsp_direction_deg(plane, [1.0, 0.0, 0.0])
    assert abs(direction) < 0.01


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_insufficient_points_error() -> None:
    """Fewer than 3 points must raise ValueError (InsufficientPoints mapped to PyValueError)."""
    with pytest.raises(ValueError):
        fsp_mod.calculate_fsp([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_single_point_error() -> None:
    """A single point must raise ValueError (InsufficientPoints mapped to PyValueError)."""
    with pytest.raises(ValueError):
        fsp_mod.calculate_fsp([[0.0, 0.0, 0.0]])


def test_colinear_points_error() -> None:
    """Collinear points must raise ValueError (DegeneratePoints mapped to PyValueError)."""
    with pytest.raises(ValueError):
        fsp_mod.calculate_fsp([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_colinear_diagonal_error() -> None:
    """Collinear points along a diagonal must raise ValueError."""
    with pytest.raises(ValueError):
        fsp_mod.calculate_fsp([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
