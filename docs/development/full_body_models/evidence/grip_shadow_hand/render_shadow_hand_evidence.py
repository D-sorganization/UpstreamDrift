"""Render 1024x1024 smooth Z-buffer PNG evidence for the Shadow hand backend."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    assembly_meshes,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.shadow import (
    build_hand_visual,
    finger_pad_contact_gaps,
    measure_shadow_anthropometry,
    penetration_metrics,
    solve_shadow_hand_qpos,
    visual_settings,
)

Array = NDArray[np.float64]
_OUT_DIR = Path(__file__).resolve().parent
_VIEWS = ("iso", "butt", "face")
_ROLLS = (0.0, 20.0, -20.0)
_DARK_SKIN_RGB = (0.36, 0.24, 0.17)
_LARGE_HAND_SCALE = 1.15


def _apply_xf(mesh: Mesh, xf: Array) -> Mesh:
    return Mesh(mesh.vertices @ xf[:3, :3].T + xf[:3, 3], mesh.faces)


def _welded_vertex_normals(mesh: Mesh) -> Array:
    """Compute smooth area-weighted normals across spatially welded vertices."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    _, inv = np.unique(np.round(verts, 6), axis=0, return_inverse=True)
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


def _camera_basis(view_name: str, roll0_xf: Array) -> tuple[Array, Array, Array]:
    top = roll0_xf[:3, 0]
    target = roll0_xf[:3, 1]
    shaft = roll0_xf[:3, 2]
    if view_name == "iso":
        fwd = -(0.50 * top + 0.68 * target - 0.54 * shaft)
        up_h = top
    elif view_name == "butt":
        fwd = 0.98 * shaft - 0.14 * top - 0.10 * target
        up_h = top
    else:
        fwd = -(0.22 * top - 0.96 * target + 0.15 * shaft)
        up_h = top
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_h)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return right, up, -fwd


def _rasterize_batches(
    sx: Array,
    sy: Array,
    sz: Array,
    faces: NDArray[np.int64],
    tri_mask: NDArray[np.bool_],
    w_ss: int,
    h_ss: int,
) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
    active = np.flatnonzero(tri_mask)
    best = np.zeros(w_ss * h_ss, dtype=np.uint64)
    if len(active) == 0:
        return best, np.empty(0, dtype=np.float64)
    z_min = float(sz.min()) - 1e-4
    z_max = float(sz.max()) + 1e-4
    z_scale = 4294967290.0 / max(z_max - z_min, 1e-6)

    f_act = faces[active]
    p0x, p0y = sx[f_act[:, 0]], sy[f_act[:, 0]]
    p1x, p1y = sx[f_act[:, 1]], sy[f_act[:, 1]]
    p2x, p2y = sx[f_act[:, 2]], sy[f_act[:, 2]]
    z0, z1, z2 = sz[f_act[:, 0]], sz[f_act[:, 1]], sz[f_act[:, 2]]

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
    valid = (area_bb > 0) & (np.abs(area2) > 1e-6)
    v_idx = np.flatnonzero(valid)
    if len(v_idx) == 0:
        return best, np.empty(0, dtype=np.float64)

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
            hz = w0[inside] * z0[hi] + w1[inside] * z1[hi] + w2[inside] * z2[hi]
            hpx = py[inside] * w_ss + px[inside]
            zq = np.clip((hz - z_min) * z_scale, 1.0, 4294967294.0).astype(np.uint64)
            packed = (zq << np.uint64(32)) | (
                active[hi].astype(np.uint64) + np.uint64(1)
            )
            np.maximum.at(best, hpx, packed)
        batch_start = end
    return best, area2


def render_scene(
    parts: list[tuple[Mesh, tuple[float, float, float], float]],
    hand_verts: Array,
    basis: tuple[Array, Array, Array],
    *,
    out_px: int,
    fill: float,
) -> Image.Image:
    """Render parts framed tightly to hand_verts with smooth Phong shading."""
    right, up, cam_z = basis
    ss = 2
    w_ss = h_ss = out_px * ss

    h_u = hand_verts @ right
    h_v = hand_verts @ up
    cu = 0.5 * (float(h_u.min()) + float(h_u.max()))
    cv = 0.5 * (float(h_v.min()) + float(h_v.max()))
    span = max(float(h_u.max() - h_u.min()), float(h_v.max() - h_v.min()), 1e-4)
    half = 0.5 * span / fill

    all_v, all_n, all_f, all_col, all_spec = [], [], [], [], []
    base = 0
    for mesh, col, spec_p in parts:
        all_v.append(mesh.vertices)
        all_n.append(_welded_vertex_normals(mesh))
        all_f.append(mesh.faces + base)
        n_f = len(mesh.faces)
        all_col.append(np.tile(col, (n_f, 1)))
        all_spec.append(np.full(n_f, spec_p, dtype=np.float64))
        base += len(mesh.vertices)

    verts = np.vstack(all_v)
    vnrm = np.vstack(all_n)
    faces = np.vstack(all_f)
    t_col = np.vstack(all_col)
    t_spec = np.concatenate(all_spec)

    sx = ((verts @ right - cu) / (2.0 * half) + 0.5) * w_ss
    sy = (0.5 - (verts @ up - cv) / (2.0 * half)) * h_ss
    sz = verts @ cam_z
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
    best, _ = _rasterize_batches(sx, sy, sz, faces, in_fr, w_ss, h_ss)

    img = np.full((h_ss * w_ss, 3), 0.96, dtype=np.float64)
    hit_pix = np.flatnonzero(best != 0)
    if len(hit_pix) > 0:
        tri_id = (best[hit_pix] & np.uint64(0xFFFFFFFF)).astype(np.int64) - 1
        cx = (hit_pix % w_ss).astype(np.float64) + 0.5
        cy = (hit_pix // w_ss).astype(np.float64) + 0.5
        f_h = faces[tri_id]
        x0, y0 = sx[f_h[:, 0]], sy[f_h[:, 0]]
        x1, y1 = sx[f_h[:, 1]], sy[f_h[:, 1]]
        x2, y2 = sx[f_h[:, 2]], sy[f_h[:, 2]]
        den = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        w0 = np.clip(((x1 - cx) * (y2 - cy) - (y1 - cy) * (x2 - cx)) / den, 0.0, 1.0)
        w1 = np.clip(((x2 - cx) * (y0 - cy) - (y2 - cy) * (x0 - cx)) / den, 0.0, 1.0)
        w2 = np.clip(1.0 - w0 - w1, 0.0, 1.0)
        n = (
            w0[:, None] * vnrm[f_h[:, 0]]
            + w1[:, None] * vnrm[f_h[:, 1]]
            + w2[:, None] * vnrm[f_h[:, 2]]
        )
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        flip = (n @ cam_z) < 0.0
        n = np.where(flip[:, None], -n, n)

        l_key = right * 0.45 + up * 0.55 + cam_z * 0.70
        l_key /= np.linalg.norm(l_key)
        l_fill = -right * 0.65 - up * 0.25 + cam_z * 0.68
        l_fill /= np.linalg.norm(l_fill)
        h_vec = l_key + cam_z
        h_vec /= np.linalg.norm(h_vec)

        ndotl_k = np.maximum(n @ l_key, 0.0)
        ndotl_f = np.maximum(n @ l_fill, 0.0)
        rim = np.power(1.0 - np.maximum(n @ cam_z, 0.0), 2.4)
        spec_i = np.power(np.maximum(n @ h_vec, 0.0), 36.0) * t_spec[tri_id]
        shade = 0.26 + 0.66 * ndotl_k + 0.24 * ndotl_f + 0.08 * rim
        rgb = t_col[tri_id] * shade[:, None] + spec_i[:, None]
        img[hit_pix] = np.clip(rgb, 0.0, 1.0)

    pil = Image.fromarray((img.reshape(h_ss, w_ss, 3) * 255.0).astype(np.uint8))
    return pil.resize((out_px, out_px), Image.Resampling.BILINEAR)


def _write_evidence_receipt(
    cfg: Any,
    base: GripSpec,
    records: list[dict[str, object]],
) -> None:
    s_lead = solve_shadow_hand_qpos(
        Hand.LEAD, base.lead, grip_spec=base, grip_radius_m=cfg.grip_radius_nominal_m
    )
    s_trail = solve_shadow_hand_qpos(
        Hand.TRAIL, base.trail, grip_spec=base, grip_radius_m=cfg.grip_radius_nominal_m
    )
    receipt = {
        "frame_px": cfg.render_frame_px,
        "frame_fill": cfg.render_frame_fill,
        "anthropometry": {
            "lead": measure_shadow_anthropometry(base, "lead"),
            "trail": measure_shadow_anthropometry(base, "trail"),
        },
        "lead_pad_gaps_m": finger_pad_contact_gaps(
            Hand.LEAD, s_lead, grip_radius_m=cfg.grip_radius_nominal_m
        ),
        "trail_pad_gaps_m": finger_pad_contact_gaps(
            Hand.TRAIL, s_trail, grip_radius_m=cfg.grip_radius_nominal_m
        ),
        "lead_penetration": penetration_metrics(
            Hand.LEAD,
            s_lead.qpos,
            mount_matrix=s_lead.mount_matrix,
            grip_radius_m=cfg.grip_radius_nominal_m,
        ),
        "trail_penetration": penetration_metrics(
            Hand.TRAIL,
            s_trail.qpos,
            mount_matrix=s_trail.mount_matrix,
            grip_radius_m=cfg.grip_radius_nominal_m,
        ),
        "renders": records,
    }
    (_OUT_DIR / "render_stats.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger(__name__).info(
        "Rendered %d images at %dpx.", len(records), cfg.render_frame_px
    )


def main() -> None:
    cfg = visual_settings()
    base = GripSpec.legacy()
    club = ClubAssembly(
        head_alias="iron_7", length_m=0.94, grip_length_m=0.27, shaft_radius_m=0.0062
    )
    club_meshes = assembly_meshes(club)
    base_pose = base.to_grip_pose()
    roll0_xf = grip_hand_frame(club, Hand.LEAD, base_pose)
    lead_rgb = (cfg.skin_tone_rgba[0], cfg.skin_tone_rgba[1], cfg.skin_tone_rgba[2])
    trail_rgb = (lead_rgb[0] * 0.92, lead_rgb[1] * 0.89, lead_rgb[2] * 0.86)
    records: list[dict[str, object]] = []

    for roll in _ROLLS:
        spec = replace(
            base,
            lead=replace(base.lead, shaft_roll_deg=roll),
            trail=replace(base.trail, shaft_roll_deg=roll),
        )
        rtag = f"roll_{int(roll)}p" if roll >= 0 else f"roll_{abs(int(roll))}m"

        # The backend applies the shaft roll (ADR-0057 A2), so place each hand
        # with the unrolled frame.
        radius = cfg.grip_radius_nominal_m
        lead_local = build_hand_visual(spec, "lead", grip_radius_m=radius).mesh
        trail_local = build_hand_visual(spec, "trail", grip_radius_m=radius).mesh
        lead_mesh = _apply_xf(lead_local, grip_hand_frame(club, Hand.LEAD, base_pose))
        trail_mesh = _apply_xf(
            trail_local, grip_hand_frame(club, Hand.TRAIL, base_pose)
        )

        static_parts = [
            (club_meshes["grip"], (0.17, 0.17, 0.20), 0.10),
            (club_meshes["shaft"], (0.72, 0.75, 0.80), 0.35),
        ]
        modes = {
            "lead": ([(lead_mesh, lead_rgb, 0.08), *static_parts], lead_mesh.vertices),
            "trail": (
                [(trail_mesh, trail_rgb, 0.08), *static_parts],
                trail_mesh.vertices,
            ),
            "both": (
                [
                    (lead_mesh, lead_rgb, 0.08),
                    (trail_mesh, trail_rgb, 0.08),
                    *static_parts,
                ],
                np.vstack([lead_mesh.vertices, trail_mesh.vertices]),
            ),
        }
        for mode, (parts, hverts) in modes.items():
            for view in _VIEWS:
                basis = _camera_basis(view, roll0_xf)
                im = render_scene(
                    parts,
                    hverts,
                    basis,
                    out_px=cfg.render_frame_px,
                    fill=cfg.render_frame_fill,
                )
                fname = f"grip_shadow_{mode}_{view}_{rtag}.png"
                im.save(_OUT_DIR / fname)
                records.append(
                    {"file": fname, "mode": mode, "view": view, "roll_deg": roll}
                )

    records.extend(
        _render_variants(cfg, base, club, club_meshes, roll0_xf, lead_rgb, trail_rgb)
    )
    _write_evidence_receipt(cfg, base, records)


def _render_variants(
    cfg: Any,
    base: GripSpec,
    club: ClubAssembly,
    club_meshes: dict[str, Mesh],
    roll0_xf: Array,
    lead_rgb: tuple[float, float, float],
    trail_rgb: tuple[float, float, float],
) -> list[dict[str, object]]:
    """Both hands at roll 0 with a chosen skin colour and with a larger hand."""
    size = base.hand_size
    large = replace(
        base,
        hand_size=replace(
            size,
            length_m=size.length_m * _LARGE_HAND_SCALE,
            breadth_m=size.breadth_m * _LARGE_HAND_SCALE,
        ),
    )
    pose = base.to_grip_pose()
    radius = cfg.grip_radius_nominal_m
    variants = {
        "skin_dark": (base, _DARK_SKIN_RGB, _DARK_SKIN_RGB),
        "size_large": (large, None, None),
    }
    records: list[dict[str, object]] = []
    for tag, (spec, lead_skin, trail_skin) in variants.items():
        lead = build_hand_visual(spec, "lead", grip_radius_m=radius, skin_rgb=lead_skin)
        trail = build_hand_visual(
            spec, "trail", grip_radius_m=radius, skin_rgb=trail_skin
        )
        lead_mesh = _apply_xf(lead.mesh, grip_hand_frame(club, Hand.LEAD, pose))
        trail_mesh = _apply_xf(trail.mesh, grip_hand_frame(club, Hand.TRAIL, pose))
        parts = [
            (lead_mesh, lead.skin_rgb if lead_skin else lead_rgb, 0.08),
            (trail_mesh, trail.skin_rgb if trail_skin else trail_rgb, 0.08),
            (club_meshes["grip"], (0.17, 0.17, 0.20), 0.10),
            (club_meshes["shaft"], (0.72, 0.75, 0.80), 0.35),
        ]
        hverts = np.vstack([lead_mesh.vertices, trail_mesh.vertices])
        for view in ("iso", "face"):
            im = render_scene(
                parts,
                hverts,
                _camera_basis(view, roll0_xf),
                out_px=cfg.render_frame_px,
                fill=cfg.render_frame_fill,
            )
            fname = f"grip_shadow_both_{view}_{tag}.png"
            im.save(_OUT_DIR / fname)
            records.append(
                {"file": fname, "mode": "both", "view": view, "variant": tag}
            )
    return records


if __name__ == "__main__":
    main()
