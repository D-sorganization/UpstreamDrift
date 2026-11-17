"""Vectorized orthographic Gouraud Z-buffer evidence renderer (1024 px, club + bones)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.shared.python.model_appearance.geometry import Mesh

Array = NDArray[np.float64]

_CLUB_KEYS: frozenset[str] = frozenset(
    {"grip", "shaft", "hosel", "ferrule", "butt_cap", "head"}
)


@dataclass(frozen=True)
class LitRenderSettings:
    image_px: int = 1024
    frame_fill: float = 0.70
    bone_rgb: tuple[float, float, float] = (0.86, 0.82, 0.76)
    grip_rgb: tuple[float, float, float] = (0.22, 0.23, 0.25)
    shaft_rgb: tuple[float, float, float] = (0.72, 0.75, 0.79)
    butt_cap_rgb: tuple[float, float, float] = (0.14, 0.14, 0.16)
    background_rgb: tuple[float, float, float] = (0.965, 0.968, 0.976)
    key_light: tuple[float, float, float] = (0.45, 0.65, 0.85)
    fill_light: tuple[float, float, float] = (-0.6, -0.25, 0.55)
    ambient: float = 0.26
    key_weight: float = 0.58
    fill_weight: float = 0.24
    specular_weight: float = 0.14
    specular_power: float = 28.0
    crease_cos: float = 0.45


def _unit(v: Array) -> Array:
    norm = float(np.linalg.norm(v))
    return v / max(norm, 1e-12)


def _camera_basis(view: str) -> tuple[Array, Array, Array]:
    if view == "front":
        fwd = _unit(np.array([0.34, -0.92, 0.26], dtype=np.float64))
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif view == "side":
        fwd = _unit(np.array([0.94, 0.30, 0.18], dtype=np.float64))
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif view == "top":
        fwd = _unit(np.array([0.04, 0.99, 0.08], dtype=np.float64))
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif view == "grip_oblique":
        fwd = _unit(np.array([0.66, 0.64, 0.39], dtype=np.float64))
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        fwd = _unit(np.array([0.50, -0.78, 0.36], dtype=np.float64))
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if abs(float(np.dot(fwd, up_hint))) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = _unit(np.asarray(np.cross(up_hint, fwd), dtype=np.float64))
    cam_up = _unit(np.asarray(np.cross(fwd, right), dtype=np.float64))
    return right, cam_up, fwd


def _corner_normals(verts: Array, faces: np.ndarray, crease_cos: float) -> Array:
    e0 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    weighted = np.asarray(np.cross(e0, e1), dtype=np.float64)
    norms = np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12)
    f_norm = weighted / norms
    v_sum = np.zeros_like(verts, dtype=np.float64)
    for c in range(3):
        np.add.at(v_sum, faces[:, c], weighted)
    v_norms = np.maximum(np.linalg.norm(v_sum, axis=1, keepdims=True), 1e-12)
    smooth = v_sum / v_norms
    out = np.empty((len(faces), 3, 3), dtype=np.float64)
    for c in range(3):
        cand = smooth[faces[:, c]]
        dots = np.sum(cand * f_norm, axis=1, keepdims=True)
        out[:, c, :] = np.where(dots >= crease_cos, cand, f_norm)
    return out


def _shade_corners(
    verts: Array,
    faces: np.ndarray,
    base_rgb: tuple[float, float, float],
    right: Array,
    cam_up: Array,
    fwd: Array,
    cfg: LitRenderSettings,
) -> Array:
    corners = _corner_normals(verts, faces, cfg.crease_cos)
    k = cfg.key_light
    fl = cfg.fill_light
    wk = _unit(k[0] * right + k[1] * cam_up + k[2] * fwd)
    wf = _unit(fl[0] * right + fl[1] * cam_up + fl[2] * fwd)
    half_v = _unit(wk + fwd)
    dk = np.maximum(0.0, corners @ wk)
    df = np.maximum(0.0, corners @ wf)
    spec = cfg.specular_weight * (
        np.maximum(0.0, corners @ half_v) ** cfg.specular_power
    )
    diff = cfg.ambient + cfg.key_weight * dk + cfg.fill_weight * df
    base = np.asarray(base_rgb, dtype=np.float64)
    return np.clip(base[None, None, :] * diff[:, :, None] + spec[:, :, None], 0.0, 1.0)


def _rasterize_batches(
    sx: Array,
    sy: Array,
    sz: Array,
    faces: NDArray[np.int64],
    px: int,
) -> NDArray[np.uint64]:
    """Vectorized orthographic Z-buffer rasterizer returning packed depth|tri_id."""
    p0x, p0y = sx[faces[:, 0]], sy[faces[:, 0]]
    p1x, p1y = sx[faces[:, 1]], sy[faces[:, 1]]
    p2x, p2y = sx[faces[:, 2]], sy[faces[:, 2]]
    z0, z1, z2 = sz[faces[:, 0]], sz[faces[:, 1]], sz[faces[:, 2]]

    x0i = np.maximum(
        0, np.floor(np.minimum(np.minimum(p0x, p1x), p2x)).astype(np.int64)
    )
    x1i = np.minimum(
        px - 1, np.ceil(np.maximum(np.maximum(p0x, p1x), p2x)).astype(np.int64)
    )
    y0i = np.maximum(
        0, np.floor(np.minimum(np.minimum(p0y, p1y), p2y)).astype(np.int64)
    )
    y1i = np.minimum(
        px - 1, np.ceil(np.maximum(np.maximum(p0y, p1y), p2y)).astype(np.int64)
    )
    bw = np.maximum(0, x1i - x0i + 1)
    bh = np.maximum(0, y1i - y0i + 1)
    area_bb = bw * bh
    area2 = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
    valid = (x1i >= x0i) & (y1i >= y0i) & (area_bb > 0) & (np.abs(area2) > 1e-6)
    v_idx = np.flatnonzero(valid)

    best = np.zeros(px * px, dtype=np.uint64)
    if len(v_idx) == 0:
        return best

    z_min = float(sz.min()) - 1e-4
    z_max = float(sz.max()) + 1e-4
    z_scale = 4294967290.0 / max(z_max - z_min, 1e-6)

    batch_start = 0
    max_cand = 1_600_000
    cum = np.cumsum(area_bb[v_idx])
    while batch_start < len(v_idx):
        base_c = 0 if batch_start == 0 else int(cum[batch_start - 1])
        end = int(np.searchsorted(cum, base_c + max_cand, side="right"))
        end = max(end, batch_start + 1)
        sl = v_idx[batch_start:end]
        counts = area_bb[sl]
        bws = bw[sl]
        local_idx = np.repeat(sl, counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        rel = np.arange(int(counts.sum()), dtype=np.int64) - starts
        bws_rep = np.repeat(bws, counts)
        gpx = np.repeat(x0i[sl], counts) + (rel % bws_rep)
        gpy = np.repeat(y0i[sl], counts) + (rel // bws_rep)
        cx, cy = gpx.astype(np.float64) + 0.5, gpy.astype(np.float64) + 0.5

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
            hz = w0[inside] * z0[hi] + w1[inside] * z1[hi] + w2[inside] * z2[hi]
            hpx = gpy[inside] * px + gpx[inside]
            zq = np.clip((hz - z_min) * z_scale, 1.0, 4294967294.0).astype(np.uint64)
            packed = (zq << np.uint64(32)) | (hi.astype(np.uint64) + np.uint64(1))
            np.maximum.at(best, hpx, packed)
        batch_start = end
    return best


def _color_for_label(label: str, cfg: LitRenderSettings) -> tuple[float, float, float]:
    if label == "grip":
        return cfg.grip_rgb
    if label in ("shaft", "hosel"):
        return cfg.shaft_rgb
    if label in ("butt_cap", "ferrule"):
        return cfg.butt_cap_rgb
    return cfg.bone_rgb


def render_scene_png(
    path: str | Path,
    meshes: dict[str, Mesh],
    *,
    view: str,
    settings: LitRenderSettings | None = None,
) -> None:
    """Render club + hand bone meshes at ``image_px`` with hand-only framing."""
    cfg = settings or LitRenderSettings()
    if not meshes:
        raise ValueError("meshes must not be empty")

    hand_meshes = {k: m for k, m in meshes.items() if k not in _CLUB_KEYS}
    fit_verts = np.vstack([m.vertices for m in (hand_meshes or meshes).values()])

    right, cam_up, fwd = _camera_basis(view)
    proj_x = fit_verts @ right
    proj_y = fit_verts @ cam_up
    span = max(
        float(proj_x.max() - proj_x.min()),
        float(proj_y.max() - proj_y.min()),
        1e-4,
    )
    cx = 0.5 * (float(proj_x.min()) + float(proj_x.max()))
    cy = 0.5 * (float(proj_y.min()) + float(proj_y.max()))

    px = cfg.image_px
    scale = (cfg.frame_fill * float(px)) / span
    half = 0.5 * float(px)

    all_v: list[Array] = []
    all_f: list[NDArray[np.int64]] = []
    all_rgb: list[Array] = []
    base = 0
    for label, mesh in meshes.items():
        all_v.append(mesh.vertices)
        faces_i = np.asarray(mesh.faces, dtype=np.int64)
        all_f.append(faces_i + base)
        all_rgb.append(
            _shade_corners(
                mesh.vertices,
                faces_i,
                _color_for_label(label, cfg),
                right,
                cam_up,
                fwd,
                cfg,
            )
        )
        base += len(mesh.vertices)

    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    corner_rgb = np.vstack(all_rgb)

    sx = (verts @ right - cx) * scale + half
    sy = half - (verts @ cam_up - cy) * scale
    sz = verts @ fwd

    best = _rasterize_batches(sx, sy, sz, faces, px)
    img = np.empty((px * px, 3), dtype=np.float64)
    img[:, :] = np.asarray(cfg.background_rgb, dtype=np.float64)
    hit_pix = np.flatnonzero(best != 0)
    if len(hit_pix) > 0:
        tri_id = (best[hit_pix] & np.uint64(0xFFFFFFFF)).astype(np.int64) - 1
        hx = (hit_pix % px).astype(np.float64) + 0.5
        hy = (hit_pix // px).astype(np.float64) + 0.5
        f_h = faces[tri_id]
        x0, y0 = sx[f_h[:, 0]], sy[f_h[:, 0]]
        x1, y1 = sx[f_h[:, 1]], sy[f_h[:, 1]]
        x2, y2 = sx[f_h[:, 2]], sy[f_h[:, 2]]
        den = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        w0 = np.clip(((x1 - hx) * (y2 - hy) - (y1 - hy) * (x2 - hx)) / den, 0.0, 1.0)
        w1 = np.clip(((x2 - hx) * (y0 - hy) - (y2 - hy) * (x0 - hx)) / den, 0.0, 1.0)
        w2 = np.clip(1.0 - w0 - w1, 0.0, 1.0)
        cf = corner_rgb[tri_id]
        img[hit_pix] = np.clip(
            w0[:, None] * cf[:, 0] + w1[:, None] * cf[:, 1] + w2[:, None] * cf[:, 2],
            0.0,
            1.0,
        )

    rgb_u8 = np.clip(np.round(img.reshape(px, px, 3) * 255.0), 0.0, 255.0).astype(
        np.uint8
    )
    Image.fromarray(rgb_u8, mode="RGB").save(str(path), format="PNG")


__all__ = ["LitRenderSettings", "render_scene_png"]
