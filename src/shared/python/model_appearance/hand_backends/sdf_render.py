"""Z-buffer Gouraud previews for the SDF hand.

Matplotlib's 3D polygons tear coincident skin, so this rasterizer keeps a
depth buffer and shades with smoothed vertex normals under camera-relative key
and fill lights. Orthographic framing fits hand vertices only to
``frame_fill`` while rendering the club grip, butt cap, and shaft stub inside
the frame.

If ``hand_backends/render.py`` grows ``render_grip_views``, that shared
renderer replaces :func:`render_standard_views`.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.grip_spec.types import GripSpec, HandGripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.contract import HandJointAngles
from src.shared.python.model_appearance.hand_backends.sdf_hand import (
    HandVisualResult,
    build_hand_visual,
)
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    SdfHandSettings,
    load_sdf_hand_settings,
    repo_root,
)

_VIEWS = ("dorsal", "oblique", "butt")
_ROLLS = (-20.0, 0.0, 20.0)
_CREASE_MIN_DOT = 0.35


def frame_half_extent(span: float, fill: float) -> float:
    """Orthographic half-extent mapping ``span`` onto ``fill`` of the frame."""
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"span must be positive, got {span!r}")
    if not np.isfinite(fill) or not 0.0 < fill <= 1.0:
        raise ValueError(f"fill must be in (0, 1], got {fill!r}")
    return float(span) / (2.0 * float(fill))


def skin_srgb(material_id: str) -> tuple[float, float, float]:
    """Look up a skin tone in ``appearance_defaults.json`` by id."""
    path = repo_root() / "src" / "config" / "appearance_defaults.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["skin_palette"]:
        if entry["id"] == material_id:
            red, green, blue = entry["srgb"]
            return (float(red), float(green), float(blue))
    raise KeyError(f"unknown skin material {material_id!r}")


def default_evidence_dir() -> Path:
    """Directory for SDF hand evidence images and the NOTICE file."""
    return (
        repo_root()
        / "docs"
        / "development"
        / "full_body_models"
        / "evidence"
        / "sdf_hand"
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("cannot normalise a near-zero camera or light vector")
    return arr / norm


def camera_basis(
    camera_dir: np.ndarray, up_hint: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(right, up, view)`` with ``view`` pointing from target to camera."""
    view = _unit(camera_dir)
    up = _unit(up_hint)
    if abs(float(np.dot(up, view))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _unit(np.asarray(np.cross(up, view), dtype=np.float64))
    cam_up = _unit(np.asarray(np.cross(view, right), dtype=np.float64))
    return right, cam_up, view


def project_hand(
    hand_vertices: np.ndarray,
    camera_dir: np.ndarray,
    up_hint: np.ndarray,
    fill: float,
    frame_px: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project hand vertices so the longer span covers ``fill`` of the frame."""
    if frame_px < 8:
        raise ValueError(f"frame_px must be >= 8, got {frame_px}")
    origin = (hand_vertices.min(axis=0) + hand_vertices.max(axis=0)) * 0.5
    right, cam_up, view = camera_basis(camera_dir, up_hint)
    rel = hand_vertices - origin
    x = rel @ right
    y = rel @ cam_up
    depth = -(rel @ view)
    x_mid = float((x.min() + x.max()) * 0.5)
    y_mid = float((y.min() + y.max()) * 0.5)
    span = float(max(x.max() - x.min(), y.max() - y.min()))
    half = frame_half_extent(span, fill)
    pixels = _to_pixels(x, y, x_mid, y_mid, half, frame_px)
    return pixels, depth, half


def _to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    x_mid: float,
    y_mid: float,
    half: float,
    frame_px: int,
) -> np.ndarray:
    px = ((x - x_mid) / half + 1.0) * 0.5 * (frame_px - 1)
    py = (1.0 - ((y - y_mid) / half + 1.0) * 0.5) * (frame_px - 1)
    return np.column_stack((px, py))


def rasterize_triangles(
    pixels: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    frame_px: int,
    background: np.ndarray,
) -> np.ndarray:
    """Paint triangles into an RGB image. Closer depths win."""
    image = np.broadcast_to(
        np.asarray(background, dtype=np.float64), (frame_px, frame_px, 3)
    ).copy()
    zbuffer = np.full((frame_px, frame_px), np.inf, dtype=np.float64)
    if colors.ndim == 1:
        colours = np.repeat(
            np.asarray(colors, dtype=np.float64)[None, :], len(faces), axis=0
        )
    else:
        colours = np.asarray(colors, dtype=np.float64)
    _raster_into(image, zbuffer, pixels, depth, faces, colours)
    return image


def _corner_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edges_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    weighted = np.asarray(np.cross(edges_a, edges_b), dtype=np.float64)
    lengths = np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12)
    face_normals = weighted / lengths
    vertex_sum = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_sum, faces[:, corner], weighted)
    v_norms = np.maximum(np.linalg.norm(vertex_sum, axis=1, keepdims=True), 1e-12)
    smooth_v = vertex_sum / v_norms
    out = np.empty((len(faces), 3, 3), dtype=np.float64)
    for corner in range(3):
        cand = smooth_v[faces[:, corner]]
        dots = np.sum(cand * face_normals, axis=1, keepdims=True)
        out[:, corner, :] = np.where(dots >= _CREASE_MIN_DOT, cand, face_normals)
    return out


def _shade_gouraud(
    vertices: np.ndarray,
    faces: np.ndarray,
    base_rgb: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    view: np.ndarray,
    settings: SdfHandSettings,
) -> np.ndarray:
    corners = _corner_normals(vertices, faces)
    k_cam = np.asarray(settings.key_light, dtype=np.float64)
    f_cam = np.asarray(settings.fill_light, dtype=np.float64)
    world_key = _unit(k_cam[0] * right + k_cam[1] * cam_up + k_cam[2] * view)
    world_fill = _unit(f_cam[0] * right + f_cam[1] * cam_up + f_cam[2] * view)
    dot_key = corners @ world_key
    dot_fill = corners @ world_fill
    if len(faces) <= 2:
        lambert_key = np.abs(dot_key)
        lambert_fill = np.abs(dot_fill)
    else:
        lambert_key = np.maximum(0.0, dot_key)
        lambert_fill = np.maximum(0.0, dot_fill)
    lit = lambert_key + settings.fill_weight * lambert_fill
    intensity = settings.ambient + (1.0 - settings.ambient) * lit / (
        1.0 + settings.fill_weight
    )
    intensity_c = np.clip(intensity, 0.0, 1.0)
    return np.clip(base_rgb[None, None, :] * intensity_c[:, :, None], 0.0, 1.0)


def cylinder_mesh(z0: float, z1: float, radius: float, *, segments: int = 24) -> Mesh:
    """Closed cylinder along z, including caps, for the grip and shaft."""
    if z1 < z0:
        z0, z1 = z1, z0
    if radius <= 0.0 or segments < 6:
        raise ValueError("cylinder needs a positive radius and at least 6 segments")
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(segments)))
    ring0 = circle * radius + np.array([0.0, 0.0, z0])
    ring1 = circle * radius + np.array([0.0, 0.0, z1])
    cap0 = np.array([[0.0, 0.0, z0]])
    cap1 = np.array([[0.0, 0.0, z1]])
    vertices = np.vstack((ring0, ring1, cap0, cap1))
    top = 2 * segments
    bottom = top + 1
    faces: list[tuple[int, int, int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((i, j, segments + j))
        faces.append((i, segments + j, segments + i))
        faces.append((top, j, i))
        faces.append((bottom, segments + i, segments + j))
    return Mesh(vertices, np.asarray(faces, dtype=np.int64))


def render_meshes(
    meshes: list[tuple[Mesh, np.ndarray]],
    hand_vertices: np.ndarray,
    camera_dir: np.ndarray,
    up_hint: np.ndarray,
    settings: SdfHandSettings,
    frame_px: int,
) -> np.ndarray:
    """Shade ``meshes`` smoothly in one camera framed on ``hand_vertices``."""
    origin = (hand_vertices.min(axis=0) + hand_vertices.max(axis=0)) * 0.5
    right, cam_up, view = camera_basis(camera_dir, up_hint)
    rel = hand_vertices - origin
    x = rel @ right
    y = rel @ cam_up
    x_mid = float((x.min() + x.max()) * 0.5)
    y_mid = float((y.min() + y.max()) * 0.5)
    span = float(max(x.max() - x.min(), y.max() - y.min()))
    half = frame_half_extent(span, settings.frame_fill)
    image = np.broadcast_to(
        np.asarray(settings.background_rgb, dtype=np.float64), (frame_px, frame_px, 3)
    ).copy()
    zbuffer = np.full((frame_px, frame_px), np.inf, dtype=np.float64)
    for mesh, rgb in meshes:
        offset = mesh.vertices - origin
        xy = _to_pixels(offset @ right, offset @ cam_up, x_mid, y_mid, half, frame_px)
        zd = -(offset @ view)
        colours = _shade_gouraud(
            mesh.vertices,
            mesh.faces,
            np.asarray(rgb, dtype=np.float64),
            right,
            cam_up,
            view,
            settings,
        )
        _raster_into(image, zbuffer, xy, zd, mesh.faces, colours)
    return image


def _raster_into(
    image: np.ndarray,
    zbuffer: np.ndarray,
    pixels: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> None:
    frame_px = image.shape[0]
    gouraud = colors.ndim == 3
    for index, face in enumerate(faces):
        tri = pixels[face]
        z = depth[face]
        min_x = max(int(np.floor(np.nanmin(tri[:, 0]))), 0)
        max_x = min(int(np.ceil(np.nanmax(tri[:, 0]))), frame_px - 1)
        min_y = max(int(np.floor(np.nanmin(tri[:, 1]))), 0)
        max_y = min(int(np.ceil(np.nanmax(tri[:, 1]))), frame_px - 1)
        if min_x > max_x or min_y > max_y:
            continue
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        area = (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0]) + (
            tri[2, 0] - tri[1, 0]
        ) * (tri[0, 1] - tri[2, 1])
        if abs(float(area)) < 1e-8:
            continue
        w0 = (
            (tri[1, 1] - tri[2, 1]) * (gx - tri[2, 0])
            + (tri[2, 0] - tri[1, 0]) * (gy - tri[2, 1])
        ) / area
        w1 = (
            (tri[2, 1] - tri[0, 1]) * (gx - tri[2, 0])
            + (tri[0, 0] - tri[2, 0]) * (gy - tri[2, 1])
        ) / area
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        zd = w0 * z[0] + w1 * z[1] + w2 * z[2]
        window = zbuffer[min_y : max_y + 1, min_x : max_x + 1]
        closer = mask & (zd < window)
        if not np.any(closer):
            continue
        rows, cols = np.nonzero(closer)
        if gouraud:
            c_face = colors[index]
            tri_rgb = (
                w0[closer, None] * c_face[0]
                + w1[closer, None] * c_face[1]
                + w2[closer, None] * c_face[2]
            )
            image[min_y + rows, min_x + cols] = tri_rgb
        else:
            image[min_y + rows, min_x + cols] = colors[index]
        zbuffer[min_y + rows, min_x + cols] = zd[closer]


def _camera_for(view: str, dorsal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shaft = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    side = _unit(np.asarray(np.cross(dorsal, shaft), dtype=np.float64))
    if view == "dorsal":
        return dorsal, shaft
    if view == "oblique":
        return _unit(dorsal * 0.75 + side * 0.35 - shaft * 0.45), shaft
    if view == "butt":
        return -shaft, dorsal
    raise ValueError(f"unknown view {view!r}")


def _save_png(path: Path, image: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def _shared_renderer():
    try:
        render = importlib.import_module(
            "src.shared.python.model_appearance.hand_backends.render"
        )
    except ImportError:
        return None
    return getattr(render, "render_grip_views", None)


def render_standard_views(
    output_dir: Path | None = None,
    *,
    frame_px: int | None = None,
    resolution: int | None = None,
    settings: SdfHandSettings | None = None,
) -> list[Path]:
    """Write lead, trail, and both hands for three views and three shaft rolls."""
    shared = _shared_renderer()
    if shared is not None:
        try:
            written = shared(
                output_dir,
                frame_px=frame_px,
                resolution=resolution,
                settings=settings,
            )
        except TypeError:
            written = None
        if written is not None:
            return [Path(path) for path in written]
    loaded = load_sdf_hand_settings() if settings is None else settings
    pixels = loaded.frame_px if frame_px is None else int(frame_px)
    samples = loaded.resolution if resolution is None else int(resolution)
    destination = default_evidence_dir() if output_dir is None else Path(output_dir)
    skin = np.asarray(skin_srgb(loaded.skin_material), dtype=np.float64)
    paths: list[Path] = []
    for roll in _ROLLS:
        lead_spec = GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=roll))
        trail_spec = GripSpec.legacy(trail=HandGripSpec(shaft_roll_deg=roll))
        lead = build_hand_visual(
            lead_spec,
            "lead",
            HandJointAngles.from_grip_spec(lead_spec, "lead"),
            settings=loaded,
            resolution=samples,
        )
        trail = build_hand_visual(
            trail_spec,
            "trail",
            HandJointAngles.from_grip_spec(trail_spec, "trail"),
            settings=loaded,
            resolution=samples,
        )
        for role, hands in (
            ("lead", (lead,)),
            ("trail", (trail,)),
            ("both", (lead, trail)),
        ):
            frame_vertices, drawables = _scene(hands, loaded, skin)
            dorsal = np.asarray(hands[0].skeleton.dorsal_hint, dtype=np.float64)
            for view in _VIEWS:
                camera, up = _camera_for(view, dorsal)
                image = render_meshes(
                    drawables, frame_vertices, camera, up, loaded, pixels
                )
                name = f"{role}_{view}_roll{roll:+04.0f}.png"
                path = destination / name
                _save_png(path, image)
                paths.append(path)
    return paths


def _scene(
    hands: tuple[HandVisualResult, ...],
    settings: SdfHandSettings,
    skin: np.ndarray,
) -> tuple[np.ndarray, list[tuple[Mesh, np.ndarray]]]:
    chunks = []
    drawables: list[tuple[Mesh, np.ndarray]] = []
    # Trail sits one native spacing (+z) toward the clubhead in both-hands views.
    spacing = GripSpec.legacy().hand_spacing_m
    for index, hand in enumerate(hands):
        shift = np.array([0.0, 0.0, spacing if index == 1 else 0.0], dtype=np.float64)
        vertices = hand.mesh.vertices + shift
        chunks.append(vertices)
        drawables.append((Mesh(vertices, hand.mesh.faces), skin))
    frame = np.vstack(chunks)
    z0 = float(frame[:, 2].min())
    z1 = float(frame[:, 2].max())
    grip_z0 = z0 - settings.butt_extension_m
    grip_z1 = z1 + 0.014
    extra = (z1 - z0) * settings.shaft_extension_fraction
    grip = cylinder_mesh(grip_z0, grip_z1, hands[0].grip_radius_m, segments=36)
    butt_cap = cylinder_mesh(
        grip_z0 - 0.006, grip_z0, hands[0].grip_radius_m * 1.07, segments=36
    )
    shaft = cylinder_mesh(
        grip_z1, grip_z1 + max(extra, 0.045), settings.shaft_radius_m, segments=24
    )
    grip_rgb = np.asarray(settings.grip_rgb, dtype=np.float64)
    drawables.append((grip, grip_rgb))
    drawables.append((butt_cap, grip_rgb * 0.85))
    drawables.append((shaft, np.asarray(settings.shaft_rgb, dtype=np.float64)))
    return frame, drawables


__all__ = [
    "camera_basis",
    "cylinder_mesh",
    "default_evidence_dir",
    "frame_half_extent",
    "project_hand",
    "rasterize_triangles",
    "render_meshes",
    "render_standard_views",
    "skin_srgb",
]
