"""Matplotlib preview: key/fill lighting and 70% framing."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh, concat_meshes
from src.shared.python.model_appearance.library import MATERIALS

Array = NDArray[np.float64]

DEFAULT_KEY_LIGHT = (0.55, 0.25, 0.79)
DEFAULT_FILL_LIGHT = (-0.35, 0.20, 0.62)


def _camera_basis(
    fwd: Sequence[float], up_hint: Sequence[float] = (0.0, 0.0, 1.0)
) -> Array:
    fwd_arr = np.asarray(fwd, dtype=np.float64)
    fwd_u = fwd_arr / max(float(np.linalg.norm(fwd_arr)), 1e-12)
    up_arr = np.asarray(up_hint, dtype=np.float64)
    right = np.cross(fwd_u, up_arr)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, fwd_u)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    cam_z = -fwd_u
    return np.vstack([right, up, cam_z]).astype(np.float64)


def skin_tone_rgb(material_name: str) -> tuple[float, float, float]:
    material = MATERIALS.get(material_name)
    if material is None:
        return (0.85, 0.70, 0.58)
    base = material.base_color
    return (float(base[0]), float(base[1]), float(base[2]))


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
    """3/4-tilted palm, dorsal (back), side, or isometric camera basis."""
    if name == "palm":
        return _camera_basis((-0.86, -0.30, -0.41))
    if name == "back":
        return _camera_basis((0.84, 0.32, -0.44))
    if name == "side":
        return _camera_basis((-0.30, 0.88, -0.37))
    if name == "iso":
        return _camera_basis((-0.58, 0.62, -0.52))
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


def _tapered_cylinder_mesh(
    z0: float,
    z1: float,
    r0: float,
    r1: float,
    *,
    sides: int = 32,
    cap_bottom: bool = False,
    cap_top: bool = False,
) -> Mesh:
    theta = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    r_bottom = np.column_stack([r0 * cos_t, r0 * sin_t, np.full_like(theta, z0)])
    r_top = np.column_stack([r1 * cos_t, r1 * sin_t, np.full_like(theta, z1)])
    verts_list: list[Array] = [r_bottom, r_top]
    faces: list[tuple[int, int, int]] = []
    for j in range(sides):
        j2 = (j + 1) % sides
        a, b = j, j2
        c, d = sides + j2, sides + j
        faces.extend([(a, b, c), (a, c, d)])
    if cap_bottom:
        c_idx = 2 * sides
        verts_list.append(np.array([[0.0, 0.0, z0]], dtype=np.float64))
        for j in range(sides):
            faces.append((c_idx, (j + 1) % sides, j))
    if cap_top:
        c_idx = sum(len(v) for v in verts_list)
        verts_list.append(np.array([[0.0, 0.0, z1]], dtype=np.float64))
        for j in range(sides):
            faces.append((c_idx, sides + j, sides + (j + 1) % sides))
    return Mesh(np.vstack(verts_list), np.asarray(faces, dtype=np.int64))


def club_context_drawables(
    grip_radius_m: float,
    *,
    z_min: float = -0.18,
    z_max: float = 0.42,
    grip_end_z: float = 0.175,
    grip_rgb: tuple[float, float, float] = (0.17, 0.18, 0.21),
    shaft_rgb: tuple[float, float, float] = (0.64, 0.67, 0.72),
) -> list[tuple[Mesh, tuple[float, float, float], float]]:
    """Build butt cap, uniform contact-radius grip, collar, and steel club shaft."""
    butt_top = min(-0.075, z_min + 0.03)
    butt_cap = _tapered_cylinder_mesh(
        z_min,
        butt_top,
        1.08 * grip_radius_m,
        grip_radius_m,
        cap_bottom=True,
        cap_top=False,
    )
    grip_body = _tapered_cylinder_mesh(
        butt_top,
        grip_end_z,
        grip_radius_m,
        grip_radius_m,
        cap_bottom=False,
        cap_top=False,
    )
    collar = _tapered_cylinder_mesh(
        grip_end_z,
        grip_end_z + 0.025,
        grip_radius_m,
        0.0062,
        cap_bottom=False,
        cap_top=False,
    )
    steel = _tapered_cylinder_mesh(
        grip_end_z + 0.025,
        max(z_max, grip_end_z + 0.10),
        0.0062,
        0.0052,
        cap_bottom=False,
        cap_top=True,
    )
    grip_mesh = concat_meshes(butt_cap, grip_body, collar)
    return [(grip_mesh, grip_rgb, 0.06), (steel, shaft_rgb, 0.26)]


def _welded_vertex_normals(mesh: Mesh, weld_tol_m: float = 1.5e-4) -> Array:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    quant = np.rint(verts / weld_tol_m).astype(np.int64)
    _, inv = np.unique(quant, axis=0, return_inverse=True)
    n_weld = int(inv.max()) + 1
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    wn = np.zeros((n_weld, 3), dtype=np.float64)
    wf = inv[faces]
    for c in range(3):
        np.add.at(wn, wf[:, c], fn)
    norms = np.linalg.norm(wn, axis=1, keepdims=True)
    wn = np.where(norms > 1e-12, wn / np.maximum(norms, 1e-12), [0.0, 0.0, 1.0])
    return wn[inv]


def _rasterize_ssaa(
    sx: Array,
    sy: Array,
    sz: Array,
    faces: NDArray[np.int64],
    tri_mask: NDArray[np.bool_],
    w_ss: int,
    h_ss: int,
) -> NDArray[np.uint64]:
    active = np.flatnonzero(tri_mask)
    best = np.zeros(w_ss * h_ss, dtype=np.uint64)
    if len(active) == 0:
        return best
    z_min = float(sz.min()) - 1e-4
    z_max = float(sz.max()) + 1e-4
    z_scale = 4294967290.0 / max(z_max - z_min, 1e-6)
    f_act = faces[active]
    p0x, p0y = sx[f_act[:, 0]], sy[f_act[:, 0]]
    p1x, p1y = sx[f_act[:, 1]], sy[f_act[:, 1]]
    p2x, p2y = sx[f_act[:, 2]], sy[f_act[:, 2]]
    tz0, tz1, tz2 = sz[f_act[:, 0]], sz[f_act[:, 1]], sz[f_act[:, 2]]
    x0i = np.maximum(
        0, np.floor(np.minimum(np.minimum(p0x, p1x), p2x)).astype(np.int64)
    )
    x1i = np.minimum(
        w_ss - 1, np.ceil(np.maximum(np.maximum(p0x, p1x), p2x)).astype(np.int64)
    )
    y0i = np.maximum(
        0, np.floor(np.minimum(np.minimum(p0y, p1y), p2y)).astype(np.int64)
    )
    y1i = np.minimum(
        h_ss - 1, np.ceil(np.maximum(np.maximum(p0y, p1y), p2y)).astype(np.int64)
    )
    bw = np.maximum(0, x1i - x0i + 1)
    bh = np.maximum(0, y1i - y0i + 1)
    area_bb = bw * bh
    area2 = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
    v_idx = np.flatnonzero((area_bb > 0) & (np.abs(area2) > 1e-6))
    if len(v_idx) == 0:
        return best
    batch_start = 0
    max_cand = 1_600_000
    cum = np.cumsum(area_bb[v_idx])
    while batch_start < len(v_idx):
        base_c = 0 if batch_start == 0 else int(cum[batch_start - 1])
        end = max(
            int(np.searchsorted(cum, base_c + max_cand, side="right")), batch_start + 1
        )
        sl = v_idx[batch_start:end]
        counts = area_bb[sl]
        bws = bw[sl]
        local_idx = np.repeat(sl, counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        rel = np.arange(int(counts.sum()), dtype=np.int64) - starts
        bws_rep = np.repeat(bws, counts)
        px = np.repeat(x0i[sl], counts) + (rel % bws_rep)
        py = np.repeat(y0i[sl], counts) + (rel // bws_rep)
        cx, cy = px.astype(np.float64) + 0.5, py.astype(np.float64) + 0.5
        t0x, t0y = p0x[local_idx], p0y[local_idx]
        t1x, t1y = p1x[local_idx], p1y[local_idx]
        t2x, t2y = p2x[local_idx], p2y[local_idx]
        inv_a = 1.0 / area2[local_idx]
        w0 = ((t1x - cx) * (t2y - cy) - (t1y - cy) * (t2x - cx)) * inv_a
        w1 = ((t2x - cx) * (t0y - cy) - (t2y - cy) * (t0x - cx)) * inv_a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if np.any(inside):
            hi = local_idx[inside]
            hz = w0[inside] * tz0[hi] + w1[inside] * tz1[hi] + w2[inside] * tz2[hi]
            hpx = py[inside] * w_ss + px[inside]
            zq = np.clip((hz - z_min) * z_scale, 1.0, 4294967294.0).astype(np.uint64)
            packed = (zq << np.uint64(32)) | (
                active[hi].astype(np.uint64) + np.uint64(1)
            )
            np.maximum.at(best, hpx, packed)
        batch_start = end
    return best


def render_shaded_scene_rgb(
    drawables: Sequence[
        tuple[Mesh, tuple[float, float, float]]
        | tuple[Mesh, tuple[float, float, float], float]
    ],
    *,
    focus_vertices: Array,
    view_rot: Array,
    width_px: int = 1024,
    height_px: int = 1024,
    target_fill: float = 0.70,
    bg_rgb: tuple[float, float, float] = (0.955, 0.96, 0.97),
) -> NDArray[np.uint8]:
    """Render meshes with welded smooth normals and 2x2 SSAA Z-buffer at target_fill."""
    rot = np.asarray(view_rot, dtype=np.float64)
    focus_cam = np.asarray(focus_vertices, dtype=np.float64) @ rot.T
    u_min, u_max = float(focus_cam[:, 0].min()), float(focus_cam[:, 0].max())
    v_min, v_max = float(focus_cam[:, 1].min()), float(focus_cam[:, 1].max())
    cu, cv = 0.5 * (u_min + u_max), 0.5 * (v_min + v_max)
    span_u = max(u_max - u_min, 1e-4)
    span_v = max(v_max - v_min, 1e-4)
    aspect = float(width_px) / float(max(height_px, 1))
    world_h = max(span_v, span_u / max(aspect, 1e-6)) / target_fill
    world_w = world_h * aspect

    all_v, all_n, all_f, all_col, all_spec = [], [], [], [], []
    base = 0
    for item in drawables:
        if len(item) == 2:
            mesh, col = item
            spec_p = 0.09
        else:
            mesh, col, spec_p = item
        all_v.append(mesh.vertices)
        all_n.append(_welded_vertex_normals(mesh))
        all_f.append(mesh.faces + base)
        nf = len(mesh.faces)
        all_col.append(np.tile(col, (nf, 1)))
        all_spec.append(np.full(nf, spec_p, dtype=np.float64))
        base += len(mesh.vertices)

    verts_cam = np.vstack(all_v) @ rot.T
    vnrm_cam = np.vstack(all_n) @ rot.T
    faces = np.vstack(all_f)
    t_col = np.vstack(all_col)
    t_spec = np.concatenate(all_spec)

    ss = 2
    w_ss, h_ss = width_px * ss, height_px * ss
    sx = ((verts_cam[:, 0] - cu) / world_w + 0.5) * w_ss
    sy = (0.5 - (verts_cam[:, 1] - cv) / world_h) * h_ss
    sz = verts_cam[:, 2]
    in_fr = (
        (np.maximum(np.maximum(sx[faces[:, 0]], sx[faces[:, 1]]), sx[faces[:, 2]]) >= 0)
        & (
            np.minimum(np.minimum(sx[faces[:, 0]], sx[faces[:, 1]]), sx[faces[:, 2]])
            < w_ss
        )
        & (
            np.maximum(np.maximum(sy[faces[:, 0]], sy[faces[:, 1]]), sy[faces[:, 2]])
            >= 0
        )
        & (
            np.minimum(np.minimum(sy[faces[:, 0]], sy[faces[:, 1]]), sy[faces[:, 2]])
            < h_ss
        )
    )
    best = _rasterize_ssaa(sx, sy, sz, faces, in_fr, w_ss, h_ss)

    img = np.empty((h_ss * w_ss, 3), dtype=np.float64)
    img[:] = bg_rgb
    hit_pix = np.flatnonzero(best != 0)
    if len(hit_pix) > 0:
        tri_id = (best[hit_pix] & np.uint64(0xFFFFFFFF)).astype(np.int64) - 1
        cx = (hit_pix % w_ss).astype(np.float64) + 0.5
        cy = (hit_pix // w_ss).astype(np.float64) + 0.5
        fh = faces[tri_id]
        x0, y0 = sx[fh[:, 0]], sy[fh[:, 0]]
        x1, y1 = sx[fh[:, 1]], sy[fh[:, 1]]
        x2, y2 = sx[fh[:, 2]], sy[fh[:, 2]]
        den = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        inv_d = np.where(
            np.abs(den) > 1e-8, 1.0 / np.where(np.abs(den) > 1e-8, den, 1.0), 0.0
        )
        w0 = np.clip(((x1 - cx) * (y2 - cy) - (y1 - cy) * (x2 - cx)) * inv_d, 0.0, 1.0)
        w1 = np.clip(((x2 - cx) * (y0 - cy) - (y2 - cy) * (x0 - cx)) * inv_d, 0.0, 1.0)
        w2 = np.clip(1.0 - w0 - w1, 0.0, 1.0)
        n = (
            w0[:, None] * vnrm_cam[fh[:, 0]]
            + w1[:, None] * vnrm_cam[fh[:, 1]]
            + w2[:, None] * vnrm_cam[fh[:, 2]]
        )
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n = np.where((n[:, 2] < 0.0)[:, None], -n, n)
        l_key = np.array([0.45, 0.55, 0.70], dtype=np.float64)
        l_key /= float(np.linalg.norm(l_key))
        l_fill = np.array([-0.65, -0.25, 0.68], dtype=np.float64)
        l_fill /= float(np.linalg.norm(l_fill))
        cam_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        h_vec = (l_key + cam_dir) / float(np.linalg.norm(l_key + cam_dir))
        ndotl_k = np.maximum(n @ l_key, 0.0)
        ndotl_f = np.maximum(n @ l_fill, 0.0)
        rim = np.power(1.0 - np.maximum(n[:, 2], 0.0), 2.4)
        spec_i = np.power(np.maximum(n @ h_vec, 0.0), 34.0) * t_spec[tri_id]
        shade = 0.27 + 0.65 * ndotl_k + 0.24 * ndotl_f + 0.08 * rim
        img[hit_pix] = np.clip(
            t_col[tri_id] * shade[:, None] + spec_i[:, None], 0.0, 1.0
        )

    ds = img.reshape(height_px, ss, width_px, ss, 3).mean(axis=(1, 3))
    return np.clip(np.rint(ds * 255.0), 0.0, 255.0).astype(np.uint8)


__all__ = [
    "add_grip_cylinder_wire",
    "add_grip_shaft_solid",
    "club_context_drawables",
    "frame_hand_viewport",
    "grip_shaft_mesh",
    "plot_shaded_mesh",
    "render_shaded_scene_rgb",
    "skin_tone_rgb",
    "transform_mesh",
    "view_rotation",
]
