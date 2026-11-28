"""Matplotlib fallback engine of the hand renderer.

Used only when no MuJoCo GL context can be created. It shares the camera rig
(so the hand is still auto-framed) and paints depth-sorted triangles shaded
from smooth vertex normals with a wrapped Lambert term. It has no depth
buffer, no shadows and no specular highlight: a frame tagged
``engine=matplotlib`` is not quality-bar evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure
from src.shared.python.model_appearance.hand_backends.render_rig import (
    CameraRig,
    ScenePart,
    vertex_normals,
)
from src.shared.python.model_appearance.hand_backends.render_settings import (
    HandRenderSettings,
)


def _shade(
    normals_cam: NDArray[np.float64],
    rgb: NDArray[np.float64],
    settings: HandRenderSettings,
) -> NDArray[np.float64]:
    """Wrapped Lambert colour per face from camera-space normals."""
    wrap = settings.wrap
    tint = settings.sss_strength * np.asarray(settings.sss_tint)
    light_sum = np.tile(np.asarray(settings.ambient_rgb), (len(normals_cam), 1))
    for light in settings.lights.values():
        direction = np.asarray(light.direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        cosine = normals_cam @ direction
        wrapped = np.clip((cosine + wrap) / (1.0 + wrap), 0.0, 1.0)
        scatter = wrapped - np.clip(cosine, 0.0, 1.0)
        colour = np.asarray(light.color) * light.intensity
        light_sum += wrapped[:, None] * colour + scatter[:, None] * tint
    return np.clip(rgb * light_sum, 0.0, 1.0)


def render_matplotlib(
    parts: Sequence[ScenePart],
    rig: CameraRig,
    settings: HandRenderSettings,
) -> NDArray[np.uint8]:
    """Painter's-algorithm frame, ``uint8`` RGB of ``frame_px`` square."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    axes = np.stack([rig.right, rig.up, rig.back], axis=1)
    polygons, colours, depths = [], [], []
    for part in parts:
        mesh = part.mesh
        normals = vertex_normals(
            mesh, weld_tolerance_m=settings.normal_weld_tolerance_m
        )
        tri = mesh.vertices[mesh.faces]
        geometric = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        centroid = tri.mean(axis=1)
        facing = np.einsum("ij,ij->i", geometric, rig.eye - centroid) > 0.0
        if part.kind == "skin":
            face_normal = normals[mesh.faces].mean(axis=1)
        else:
            face_normal = geometric
        length = np.linalg.norm(face_normal, axis=1, keepdims=True)
        face_normal = face_normal / np.where(length > 0.0, length, 1.0)
        shaded = _shade(face_normal @ axes, np.asarray(part.rgba[:3]), settings)
        flat = rig.project(tri.reshape(-1, 3)).reshape(-1, 3, 2)
        depth = rig.to_camera(centroid)[:, 2]
        polygons.append(flat[facing])
        colours.append(shaded[facing])
        depths.append(depth[facing])
    order = np.argsort(np.concatenate(depths), kind="stable")
    polygons_all = np.concatenate(polygons)[order]
    colours_all = np.concatenate(colours)[order]

    inches = settings.frame_px / settings.matplotlib_dpi
    figure = Figure(figsize=(inches, inches), dpi=settings.matplotlib_dpi)
    canvas = FigureCanvasAgg(figure)
    figure.patch.set_facecolor(settings.background_rgba[:3])
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    axis.add_collection(
        PolyCollection(
            polygons_all,
            facecolors=colours_all,
            edgecolors=colours_all,
            linewidths=settings.matplotlib_edge_width,
            antialiaseds=True,
        )
    )
    canvas.draw()
    image = np.ascontiguousarray(
        np.asarray(canvas.buffer_rgba())[..., :3], dtype=np.uint8
    )
    ensure(
        image.shape == (settings.frame_px, settings.frame_px, 3),
        "the matplotlib frame is frame_px square RGB",
    )
    return image
