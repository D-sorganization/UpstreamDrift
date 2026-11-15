"""Matplotlib preview helpers with equal-aspect shaded meshes."""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.library import MATERIALS


def skin_tone_rgb(material_name: str) -> tuple[float, float, float]:
    """Resolve a library skin material to linear RGB floats."""
    material = MATERIALS.get(material_name)
    if material is None:
        return (0.85, 0.70, 0.58)
    return tuple(material.base_color[:3])


def plot_shaded_mesh(
    ax,
    mesh: Mesh,
    *,
    color: tuple[float, float, float],
    alpha: float = 0.95,
    light: tuple[float, float, float] = (0.4, 0.2, 0.9),
) -> None:
    """Gouraud-style flat facet shading with equal data scaling per axis."""
    verts = mesh.vertices
    faces = mesh.faces
    centres = verts[faces].mean(axis=1)
    normals = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]],
        verts[faces[:, 2]] - verts[faces[:, 0]],
    )
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    light_vec = np.asarray(light, dtype=float)
    light_vec /= np.linalg.norm(light_vec)
    shade = 0.35 + 0.65 * np.clip(normals @ light_vec, 0.0, 1.0)
    face_colors = np.array(
        [np.array(color) * float(s) for s in shade],
        dtype=float,
    )
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polys = verts[faces]
    collection = Poly3DCollection(
        polys,
        facecolors=face_colors,
        edgecolors=(0.25, 0.18, 0.14, 0.15),
        linewidths=0.2,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    span = verts.max(axis=0) - verts.min(axis=0)
    centre = verts.mean(axis=0)
    radius = float(span.max()) * 0.55
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        ax.set_aspect("equal")


def add_grip_cylinder_wire(
    ax,
    grip_radius_m: float,
    *,
    z_range: tuple[float, float],
    color: tuple[float, float, float] = (0.25, 0.25, 0.3),
) -> None:
    """Draw the grip cylinder edge for context."""
    theta = np.linspace(0.0, 2.0 * math.pi, 48)
    z = np.linspace(z_range[0], z_range[1], 2)
    tg, zg = np.meshgrid(theta, z)
    x = grip_radius_m * np.cos(tg)
    y = grip_radius_m * np.sin(tg)
    ax.plot_wireframe(x, y, zg, color=color, linewidth=0.4, alpha=0.5)


__all__ = ["add_grip_cylinder_wire", "plot_shaded_mesh", "skin_tone_rgb"]
