"""Matplotlib preview: key/fill lighting and 70% framing."""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.library import MATERIALS

DEFAULT_KEY_LIGHT = (0.55, 0.25, 0.79)
DEFAULT_FILL_LIGHT = (-0.35, 0.20, 0.62)


def skin_tone_rgb(material_name: str) -> tuple[float, float, float]:
    material = MATERIALS.get(material_name)
    if material is None:
        return (0.85, 0.70, 0.58)
    return tuple(material.base_color[:3])


def _shade_faces(
    normals: np.ndarray,
    *,
    key: np.ndarray,
    fill: np.ndarray,
    ambient: float = 0.22,
) -> np.ndarray:
    key = key / np.linalg.norm(key)
    fill = fill / np.linalg.norm(fill)
    key_term = np.clip(normals @ key, 0.0, 1.0)
    fill_term = np.clip(normals @ fill, 0.0, 1.0)
    return ambient + 0.62 * key_term + 0.28 * fill_term


def plot_shaded_mesh(
    ax,
    mesh: Mesh,
    *,
    color: tuple[float, float, float],
    alpha: float = 1.0,
    key_light: tuple[float, float, float] = DEFAULT_KEY_LIGHT,
    fill_light: tuple[float, float, float] = DEFAULT_FILL_LIGHT,
    frame_fill: float = 0.70,
    auto_frame: bool = True,
) -> None:
    verts = mesh.vertices
    faces = mesh.faces
    normals = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]],
        verts[faces[:, 2]] - verts[faces[:, 0]],
    )
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    shade = _shade_faces(
        normals,
        key=np.asarray(key_light, float),
        fill=np.asarray(fill_light, float),
    )
    face_colors = np.clip(
        np.array([np.array(color) * float(s) for s in shade], dtype=float),
        0.0,
        1.0,
    )
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polys = verts[faces]
    collection = Poly3DCollection(
        polys,
        facecolors=face_colors,
        edgecolors=(0.0, 0.0, 0.0, 0.0),
        linewidths=0.0,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    if auto_frame:
        frame_hand_viewport(ax, verts, fill_fraction=frame_fill)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        ax.set_aspect("equal")


def frame_hand_viewport(
    ax,
    vertices: np.ndarray,
    *,
    fill_fraction: float = 0.70,
) -> None:
    """Centre the hand and span ~``fill_fraction`` of the viewport."""
    if not (0.2 < fill_fraction <= 1.0):
        raise ValueError("fill_fraction must be in (0.2, 1]")
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    centre = 0.5 * (vmin + vmax)
    span = float(np.max(vmax - vmin))
    half = span / (2.0 * fill_fraction)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)


def view_rotation(name: str) -> np.ndarray:
    """Palm, dorsal (back), or side rotation matrices."""
    if name == "palm":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=float,
        )
    if name == "back":
        return np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=float,
        )
    if name == "side":
        return np.eye(3)
    raise ValueError(f"unknown view {name!r}")


def transform_mesh(mesh: Mesh, rotation: np.ndarray) -> Mesh:
    return Mesh(mesh.vertices @ rotation.T, mesh.faces.copy())


def grip_shaft_mesh(
    grip_radius_m: float,
    *,
    z_range: tuple[float, float],
    sides: int = 28,
) -> Mesh:
    """Simple tapered grip tube for contextual renders (club visible in frame)."""
    import math

    z0, z1 = z_range
    length = z1 - z0
    if length <= 1e-6:
        length = 0.12
        z0, z1 = -0.06, 0.06
    theta = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    rings = []
    for z in (z0, z1):
        rings.append(
            np.column_stack(
                [
                    grip_radius_m * np.cos(theta),
                    grip_radius_m * np.sin(theta),
                    np.full_like(theta, z),
                ]
            )
        )
    verts = np.vstack(rings)
    faces: list[tuple[int, int, int]] = []
    for j in range(sides):
        j2 = (j + 1) % sides
        a, b = j, j2
        c, d = sides + j2, sides + j
        faces += [(a, b, c), (a, c, d)]
    return Mesh(verts, np.asarray(faces, dtype=np.int64))


def add_grip_shaft_solid(
    ax,
    grip_radius_m: float,
    *,
    z_range: tuple[float, float],
    rotation: np.ndarray | None = None,
    color: tuple[float, float, float] = (0.22, 0.24, 0.28),
) -> None:
    grip = grip_shaft_mesh(grip_radius_m, z_range=z_range)
    if rotation is not None:
        grip = transform_mesh(grip, rotation)
    plot_shaded_mesh(
        ax,
        grip,
        color=color,
        alpha=0.92,
        auto_frame=False,
        key_light=(0.5, 0.15, 0.85),
        fill_light=(-0.4, 0.2, 0.5),
    )


def add_grip_cylinder_wire(
    ax,
    grip_radius_m: float,
    *,
    z_range: tuple[float, float],
    rotation: np.ndarray | None = None,
    color: tuple[float, float, float] = (0.35, 0.38, 0.42),
) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 64)
    z = np.linspace(z_range[0], z_range[1], 2)
    pts = []
    for zi in z:
        ring = np.column_stack(
            [
                grip_radius_m * np.cos(theta),
                grip_radius_m * np.sin(theta),
                np.full_like(theta, zi),
            ]
        )
        if rotation is not None:
            ring = ring @ rotation.T
        pts.append(ring)
    for ring in pts:
        ax.plot(
            ring[:, 0], ring[:, 1], ring[:, 2], color=color, linewidth=0.8, alpha=0.65
        )


__all__ = [
    "add_grip_cylinder_wire",
    "add_grip_shaft_solid",
    "grip_shaft_mesh",
    "frame_hand_viewport",
    "plot_shaded_mesh",
    "skin_tone_rgb",
    "transform_mesh",
    "view_rotation",
]
