"""
Analytical inertia formulas for primitive shapes.

This module provides closed-form solutions for computing inertia
tensors of common geometric primitives.
"""

from __future__ import annotations

import math
from typing import Any


def box_inertia(
    mass: float,
    size_x: float,
    size_y: float,
    size_z: float,
) -> dict[str, float]:
    """
    Compute inertia tensor for a solid box (cuboid).

    The box is centered at the origin with sides parallel to axes.

    Args:
        mass: Mass in kg
        size_x: Full dimension along X axis (m)
        size_y: Full dimension along Y axis (m)
        size_z: Full dimension along Z axis (m)

    Returns:
        Dict with ixx, iyy, izz, ixy, ixz, iyz
    """
    ixx = (mass / 12.0) * (size_y**2 + size_z**2)
    iyy = (mass / 12.0) * (size_x**2 + size_z**2)
    izz = (mass / 12.0) * (size_x**2 + size_y**2)

    return {
        "ixx": ixx,
        "iyy": iyy,
        "izz": izz,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyz": 0.0,
    }


def cylinder_inertia(
    mass: float,
    radius: float,
    length: float,
    axis: str = "z",
) -> dict[str, float]:
    """
    Compute inertia tensor for a solid cylinder.

    The cylinder is centered at the origin.

    Args:
        mass: Mass in kg
        radius: Radius in meters
        length: Length (height) in meters
        axis: Cylinder axis ('x', 'y', or 'z')

    Returns:
        Dict with ixx, iyy, izz, ixy, ixz, iyz
    """
    # Moment of inertia about the cylinder axis
    i_axial = 0.5 * mass * radius**2

    # Moment of inertia about perpendicular axes
    i_perp = (mass / 12.0) * (3.0 * radius**2 + length**2)

    if axis == "x":
        return {
            "ixx": i_axial,
            "iyy": i_perp,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    elif axis == "y":
        return {
            "ixx": i_perp,
            "iyy": i_axial,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    else:  # z (default)
        return {
            "ixx": i_perp,
            "iyy": i_perp,
            "izz": i_axial,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }


def sphere_inertia(mass: float, radius: float) -> dict[str, float]:
    """
    Compute inertia tensor for a solid sphere.

    The sphere is centered at the origin.

    Args:
        mass: Mass in kg
        radius: Radius in meters

    Returns:
        Dict with ixx, iyy, izz, ixy, ixz, iyz
    """
    i = (2.0 / 5.0) * mass * radius**2

    return {
        "ixx": i,
        "iyy": i,
        "izz": i,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyz": 0.0,
    }


def capsule_inertia(
    mass: float,
    radius: float,
    length: float,
    axis: str = "z",
) -> dict[str, float]:
    """
    Compute inertia tensor for a solid capsule.

    A capsule is a cylinder with hemispherical caps. The total length
    includes both hemispheres.

    Args:
        mass: Total mass in kg
        radius: Radius of cylinder and hemispheres (m)
        length: Length of cylindrical portion (m), not including caps
        axis: Capsule axis ('x', 'y', or 'z')

    Returns:
        Dict with ixx, iyy, izz, ixy, ixz, iyz
    """
    # Volume calculations
    v_cyl = math.pi * radius**2 * length
    v_sphere = (4.0 / 3.0) * math.pi * radius**3
    v_total = v_cyl + v_sphere

    if v_total == 0:
        return sphere_inertia(mass, radius)

    # Mass distribution
    m_cyl = mass * v_cyl / v_total
    m_sphere = mass * v_sphere / v_total

    # Cylinder contribution
    i_cyl_axial = 0.5 * m_cyl * radius**2
    i_cyl_perp = (m_cyl / 12.0) * (3.0 * radius**2 + length**2)

    # Sphere contribution (two hemispheres)
    # Inertia of full sphere about center
    i_sphere_center = (2.0 / 5.0) * m_sphere * radius**2

    # Each hemisphere is offset from center
    # Using parallel axis theorem
    # Distance from capsule center to hemisphere center
    hemisphere_offset = length / 2.0 + (3.0 / 8.0) * radius

    # Perpendicular axis (parallel axis theorem)
    i_sphere_perp = i_sphere_center + 0.5 * m_sphere * hemisphere_offset**2

    # Total inertia
    i_axial = i_cyl_axial + i_sphere_center
    i_perp = i_cyl_perp + i_sphere_perp

    if axis == "x":
        return {
            "ixx": i_axial,
            "iyy": i_perp,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    elif axis == "y":
        return {
            "ixx": i_perp,
            "iyy": i_axial,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    else:  # z (default)
        return {
            "ixx": i_perp,
            "iyy": i_perp,
            "izz": i_axial,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }


def ellipsoid_inertia(
    mass: float,
    a: float,
    b: float,
    c: float,
) -> dict[str, float]:
    """
    Compute inertia tensor for a solid ellipsoid.

    The ellipsoid is centered at the origin with semi-axes a, b, c
    along the x, y, z axes respectively.

    Args:
        mass: Mass in kg
        a: Semi-axis along X (m)
        b: Semi-axis along Y (m)
        c: Semi-axis along Z (m)

    Returns:
        Dict with ixx, iyy, izz, ixy, ixz, iyz
    """
    ixx = (mass / 5.0) * (b**2 + c**2)
    iyy = (mass / 5.0) * (a**2 + c**2)
    izz = (mass / 5.0) * (a**2 + b**2)

    return {
        "ixx": ixx,
        "iyy": iyy,
        "izz": izz,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyz": 0.0,
    }


def hollow_cylinder_inertia(
    mass: float,
    inner_radius: float,
    outer_radius: float,
    length: float,
    axis: str = "z",
) -> dict[str, float]:
    """
    Compute inertia tensor for a hollow cylinder (tube).

    Args:
        mass: Mass in kg
        inner_radius: Inner radius (m)
        outer_radius: Outer radius (m)
        length: Length (m)
        axis: Cylinder axis ('x', 'y', or 'z')

    Returns:
        Dict with ixx, iyy, izz, ixy, ixz, iyz
    """
    r1_sq = inner_radius**2
    r2_sq = outer_radius**2

    i_axial = 0.5 * mass * (r1_sq + r2_sq)
    i_perp = (mass / 12.0) * (3.0 * (r1_sq + r2_sq) + length**2)

    if axis == "x":
        return {
            "ixx": i_axial,
            "iyy": i_perp,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    elif axis == "y":
        return {
            "ixx": i_perp,
            "iyy": i_axial,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    else:  # z (default)
        return {
            "ixx": i_perp,
            "iyy": i_perp,
            "izz": i_axial,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }


def cone_inertia(
    mass: float,
    radius: float,
    height: float,
    axis: str = "z",
) -> dict[str, float]:
    """
    Compute inertia tensor for a solid cone.

    The cone has its apex at the origin and extends along the positive axis.

    Args:
        mass: Mass in kg
        radius: Base radius (m)
        height: Height (m)
        axis: Cone axis ('x', 'y', or 'z')

    Returns:
        Dict with ixx, iyy, izz, ixy, ixz, iyz
    """
    # Inertia about apex
    i_axial = (3.0 / 10.0) * mass * radius**2
    i_perp = mass * ((3.0 / 20.0) * radius**2 + (3.0 / 5.0) * height**2)

    if axis == "x":
        return {
            "ixx": i_axial,
            "iyy": i_perp,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    elif axis == "y":
        return {
            "ixx": i_perp,
            "iyy": i_axial,
            "izz": i_perp,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
    else:  # z (default)
        return {
            "ixx": i_perp,
            "iyy": i_perp,
            "izz": i_axial,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }


def parallel_axis(
    inertia: dict[str, float],
    mass: float,
    offset: tuple[float, float, float],
) -> dict[str, float]:
    """
    Apply parallel axis theorem to shift inertia tensor.

    Args:
        inertia: Original inertia about COM
        mass: Mass in kg
        offset: (dx, dy, dz) offset from COM to new point

    Returns:
        Inertia about new point
    """
    dx, dy, dz = offset
    d_sq = dx**2 + dy**2 + dz**2

    return {
        "ixx": inertia["ixx"] + mass * (dy**2 + dz**2),
        "iyy": inertia["iyy"] + mass * (dx**2 + dz**2),
        "izz": inertia["izz"] + mass * (dx**2 + dy**2),
        "ixy": inertia.get("ixy", 0.0) - mass * dx * dy,
        "ixz": inertia.get("ixz", 0.0) - mass * dx * dz,
        "iyz": inertia.get("iyz", 0.0) - mass * dy * dz,
    }


def combine_inertias(
    inertias: list[tuple[dict[str, float], float, tuple[float, float, float]]],
) -> dict[str, float]:
    """
    Combine multiple inertias at their respective positions.

    Each element is (inertia_at_local_com, mass, position_of_local_com).
    Returns combined inertia about the combined COM.

    Args:
        inertias: List of (inertia_dict, mass, position) tuples

    Returns:
        Combined inertia about combined COM
    """
    if not inertias:
        return {"ixx": 0.0, "iyy": 0.0, "izz": 0.0, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}

    # Compute combined COM
    total_mass = sum(m for _, m, _ in inertias)
    if total_mass == 0:
        return {"ixx": 0.0, "iyy": 0.0, "izz": 0.0, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}

    com_x = sum(m * p[0] for _, m, p in inertias) / total_mass
    com_y = sum(m * p[1] for _, m, p in inertias) / total_mass
    com_z = sum(m * p[2] for _, m, p in inertias) / total_mass

    # Sum inertias about combined COM using parallel axis theorem
    result = {"ixx": 0.0, "iyy": 0.0, "izz": 0.0, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}

    for inertia, mass, pos in inertias:
        # Offset from this COM to combined COM
        offset = (pos[0] - com_x, pos[1] - com_y, pos[2] - com_z)
        shifted = parallel_axis(inertia, mass, offset)

        for key in result:
            result[key] += shifted[key]

    return result
