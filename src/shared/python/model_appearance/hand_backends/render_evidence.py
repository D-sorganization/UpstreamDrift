"""Headless lit renders for hand-backend evidence (1024 px, club + bones)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh


@dataclass(frozen=True)
class LitRenderSettings:
    image_px: int = 1024
    frame_fill: float = 0.7
    bone_rgb: tuple[float, float, float] = (0.86, 0.82, 0.76)
    grip_rgb: tuple[float, float, float] = (0.35, 0.35, 0.38)
    key_light: tuple[float, float, float] = (0.4, 0.35, 1.0)
    fill_light: tuple[float, float, float] = (-0.6, 0.2, 0.4)
    ambient: float = 0.22
    key_weight: float = 0.62
    fill_weight: float = 0.28


def _triangle_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return n / norm


def _face_colors(
    normals: np.ndarray,
    base_rgb: tuple[float, float, float],
    settings: LitRenderSettings,
) -> np.ndarray:
    key = np.asarray(settings.key_light, dtype=float)
    key /= np.linalg.norm(key)
    fill = np.asarray(settings.fill_light, dtype=float)
    fill /= np.linalg.norm(fill)
    lambert = (
        settings.ambient
        + settings.key_weight * np.clip(normals @ key, 0.0, 1.0)
        + settings.fill_weight * np.clip(normals @ fill, 0.0, 1.0)
    )
    base = np.asarray(base_rgb, dtype=float)
    return np.clip(lambert[:, None] * base[None, :], 0.0, 1.0)


def _camera_angles(view: str) -> tuple[float, float]:
    return {
        "front": (15.0, -70.0),
        "side": (10.0, 20.0),
        "top": (88.0, -90.0),
        "grip_oblique": (22.0, -48.0),
    }.get(view, (15.0, -60.0))


def render_scene_png(
    path,
    meshes: dict[str, Mesh],
    *,
    view: str,
    settings: LitRenderSettings | None = None,
) -> None:
    """Render bone + grip meshes to ``path`` (PNG)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cfg = settings or LitRenderSettings()
    if not meshes:
        raise ValueError("meshes must not be empty")

    all_verts = np.vstack([m.vertices for m in meshes.values()])
    centre = all_verts.mean(axis=0)
    extent = float(np.ptp(all_verts, axis=0).max())
    # Tight orthographic limits so subject fills ~frame_fill of the square axes box.
    half = max(0.5 * extent / (cfg.frame_fill * 0.82), 0.024)

    dpi = cfg.image_px
    inches = 1.0
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    for label, mesh in meshes.items():
        normals = _triangle_normals(mesh.vertices, mesh.faces)
        rgb = cfg.grip_rgb if label == "grip" else cfg.bone_rgb
        colors = _face_colors(normals, rgb, cfg)
        tri_verts = mesh.vertices[mesh.faces]
        poly = Poly3DCollection(tri_verts, linewidths=0.04, edgecolors=(0.45, 0.42, 0.4, 0.35))
        poly.set_facecolor(colors)
        poly.set_alpha(1.0 if label != "grip" else 0.92)
        ax.add_collection3d(poly)

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    elev, azim = _camera_angles(view)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_box_aspect([1, 1, 1])
    ax.dist = 7.0
    ax.axis("off")
    fig.savefig(path, dpi=dpi, facecolor="#f8f8fa", pad_inches=0.0)
    plt.close(fig)


__all__ = ["LitRenderSettings", "render_scene_png"]