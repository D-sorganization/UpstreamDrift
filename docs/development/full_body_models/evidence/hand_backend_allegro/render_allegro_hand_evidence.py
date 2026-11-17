"""Gouraud Z-buffer evidence renderer for the Wonik Allegro visual backend (#15152)."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh, concat_meshes
from src.shared.python.model_appearance.hand_backends.allegro import (
    extract_allegro_hand_parts,
    finger_pad_contact_gaps,
    measure_allegro_anthropometry,
    penetration_metrics,
)
from src.shared.python.model_appearance.hand_backends.allegro_settings import (
    _CONFIG_PATH,
    AllegroHandVisualSettings,
    visual_settings,
)

_MODES = ("lead", "trail", "both")
_VIEWS = ("side", "top", "butt")
_ROLLS = (-20.0, 0.0, 20.0)


@dataclasses.dataclass(frozen=True)
class _RenderTheme:
    frame_px: int
    frame_fill: float
    both_frame_fill: float
    club_overhang_m: float
    grip_rgba: tuple[float, float, float, float]
    butt_cap_rgba: tuple[float, float, float, float]
    shaft_rgba: tuple[float, float, float, float]
    background_rgb: np.ndarray
    crease_cos: float
    ambient: float
    key_dir: np.ndarray
    key_weight: float
    fill_dir: np.ndarray
    fill_weight: float
    spec_weight: float
    spec_power: float


def _load_theme() -> _RenderTheme:
    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))["rendering"]
    return _RenderTheme(
        frame_px=int(raw["frame_px"]),
        frame_fill=float(raw["frame_fill"]),
        both_frame_fill=float(raw["both_frame_fill"]),
        club_overhang_m=float(raw["club_axial_overhang_m"]),
        grip_rgba=tuple(float(v) for v in raw["grip_rubber_rgba"]),  # type: ignore[arg-type]
        butt_cap_rgba=tuple(float(v) for v in raw["butt_cap_rgba"]),  # type: ignore[arg-type]
        shaft_rgba=tuple(float(v) for v in raw["shaft_steel_rgba"]),  # type: ignore[arg-type]
        background_rgb=np.asarray(raw["background_rgb"], dtype=np.float64) / 255.0,
        crease_cos=float(raw["crease_cos_threshold"]),
        ambient=float(raw["ambient_light"]),
        key_dir=np.asarray(raw["key_light_dir"], dtype=np.float64),
        key_weight=float(raw["key_light_weight"]),
        fill_dir=np.asarray(raw["fill_light_dir"], dtype=np.float64),
        fill_weight=float(raw["fill_light_weight"]),
        spec_weight=float(raw["specular_weight"]),
        spec_power=float(raw["specular_power"]),
    )


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def _camera_basis(view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if view == "side":
        cam_dir = np.array([1.0, -0.35, -0.18], dtype=np.float64)
        up_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    elif view == "top":
        cam_dir = np.array([0.2, 1.0, -0.15], dtype=np.float64)
        up_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    elif view == "butt":
        cam_dir = np.array([0.12, 0.12, -1.0], dtype=np.float64)
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"unknown view {view!r}")
    fwd = _unit(cam_dir)
    up = _unit(up_hint)
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = _unit(np.asarray(np.cross(up, fwd), dtype=np.float64))
    cam_up = _unit(np.asarray(np.cross(fwd, right), dtype=np.float64))
    return right, cam_up, fwd


def _cylinder_mesh(z0: float, z1: float, radius: float, segments: int = 28) -> Mesh:
    lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circ = np.column_stack((np.cos(ang), np.sin(ang), np.zeros(segments)))
    r0 = circ * radius + np.array([0.0, 0.0, lo])
    r1 = circ * radius + np.array([0.0, 0.0, hi])
    c0 = np.array([[0.0, 0.0, lo]])
    c1 = np.array([[0.0, 0.0, hi]])
    verts = np.vstack((r0, r1, c0, c1))
    top_i, bot_i = 2 * segments, 2 * segments + 1
    faces: list[tuple[int, int, int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(
            [
                (i, j, segments + j),
                (i, segments + j, segments + i),
                (top_i, j, i),
                (bot_i, segments + i, segments + j),
            ]
        )
    return Mesh(verts, np.asarray(faces, dtype=np.int64))


def _corner_normals(
    vertices: np.ndarray, faces: np.ndarray, crease_cos: float
) -> np.ndarray:
    e0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    weighted = np.asarray(np.cross(e0, e1), dtype=np.float64)
    norms = np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12)
    f_norm = weighted / norms
    v_sum = np.zeros_like(vertices, dtype=np.float64)
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
    vertices: np.ndarray,
    faces: np.ndarray,
    base_rgb: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    fwd: np.ndarray,
    theme: _RenderTheme,
) -> np.ndarray:
    corners = _corner_normals(vertices, faces, theme.crease_cos)
    k = theme.key_dir
    fl = theme.fill_dir
    wk = _unit(k[0] * right + k[1] * cam_up + k[2] * fwd)
    wf = _unit(fl[0] * right + fl[1] * cam_up + fl[2] * fwd)
    half_vec = _unit(wk + fwd)
    dk = np.maximum(0.0, corners @ wk)
    df = np.maximum(0.0, corners @ wf)
    spec = theme.spec_weight * (np.maximum(0.0, corners @ half_vec) ** theme.spec_power)
    diff = theme.ambient + theme.key_weight * dk + theme.fill_weight * df
    return np.clip(
        base_rgb[None, None, :] * diff[:, :, None] + spec[:, :, None],
        0.0,
        1.0,
    )


def _raster_into(
    image: np.ndarray,
    zbuffer: np.ndarray,
    px_xy: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    corner_rgb: np.ndarray,
) -> None:
    frame_px = image.shape[0]
    p0 = px_xy[faces[:, 0]]
    p1 = px_xy[faces[:, 1]]
    p2 = px_xy[faces[:, 2]]
    min_xs = np.maximum(
        np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(int), 0
    )
    max_xs = np.minimum(
        np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(int),
        frame_px - 1,
    )
    min_ys = np.maximum(
        np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(int), 0
    )
    max_ys = np.minimum(
        np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(int),
        frame_px - 1,
    )
    areas = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (
        p0[:, 1] - p2[:, 1]
    )
    active = np.nonzero(
        (max_xs >= min_xs) & (max_ys >= min_ys) & (np.abs(areas) >= 1e-8)
    )[0]
    for idx in active:
        x0, x1 = int(min_xs[idx]), int(max_xs[idx])
        y0, y1 = int(min_ys[idx]), int(max_ys[idx])
        tri = px_xy[faces[idx]]
        zs = depth[faces[idx]]
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        area = float(areas[idx])
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
        zd = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        closer = mask & (zd < zbuffer[y0 : y1 + 1, x0 : x1 + 1])
        if not np.any(closer):
            continue
        rows, cols = np.nonzero(closer)
        c_face = corner_rgb[idx]
        image[y0 + rows, x0 + cols] = (
            w0[closer, None] * c_face[0]
            + w1[closer, None] * c_face[1]
            + w2[closer, None] * c_face[2]
        )
        zbuffer[y0 + rows, x0 + cols] = zd[closer]


def compute_hand_fill_fraction(
    hand_bounds_xy: tuple[float, float, float, float],
    viewport_bounds_xy: tuple[float, float, float, float],
) -> float:
    """Return max projected hand span divided by square viewport span."""
    hx0, hx1, hy0, hy1 = hand_bounds_xy
    vx0, vx1, vy0, vy1 = viewport_bounds_xy
    v_span = max(vx1 - vx0, vy1 - vy0)
    if v_span <= 0.0:
        raise ValueError("viewport span must be positive")
    return float(max(hx1 - hx0, hy1 - hy0) / v_span)


def _split_skin_and_tips(
    parts: dict[str, Mesh], z_shift: float, cfg: AllegroHandVisualSettings
) -> list[tuple[Mesh, tuple[float, float, float, float]]]:
    skin_meshes: list[Mesh] = []
    tip_meshes: list[Mesh] = []
    for k, m in parts.items():
        shifted = Mesh(m.vertices + np.array([0.0, 0.0, z_shift]), m.faces)
        if k.endswith("_tip"):
            tip_meshes.append(shifted)
        else:
            skin_meshes.append(shifted)
    out: list[tuple[Mesh, tuple[float, float, float, float]]] = []
    if skin_meshes:
        out.append((concat_meshes(*skin_meshes), cfg.skin_tone_rgba))
    if tip_meshes:
        out.append((concat_meshes(*tip_meshes), cfg.tip_pad_rgba))
    return out


def _club_scene_items(
    hand_verts: np.ndarray, radius: float, theme: _RenderTheme
) -> list[tuple[Mesh, tuple[float, float, float, float]]]:
    """Butt cap, grip and shaft cylinders spanning the hands' z extent."""
    z_min = float(hand_verts[:, 2].min()) - theme.club_overhang_m * 0.45
    z_max = float(hand_verts[:, 2].max()) + theme.club_overhang_m
    butt_z0 = z_min - 0.015
    butt_mesh = _cylinder_mesh(butt_z0, z_min, radius * 1.12)
    grip_mesh = _cylinder_mesh(z_min, z_max * 0.65, radius)
    shaft_mesh = _cylinder_mesh(z_max * 0.65, z_max + 0.12, radius * 0.48)
    return [
        (butt_mesh, theme.butt_cap_rgba),
        (grip_mesh, theme.grip_rgba),
        (shaft_mesh, theme.shaft_rgba),
    ]


def render_single_view(
    grip_spec: GripSpec,
    *,
    mode: str,
    view: str,
    roll_deg: float,
    out_path: Path,
    grip_radius_m: float | None = None,
) -> dict[str, Any]:
    """Render one 1024+ px view with hand-only 70% framing and club shaft visible."""
    cfg = visual_settings()
    theme = _load_theme()
    radius = (
        cfg.grip_radius_nominal_m if grip_radius_m is None else float(grip_radius_m)
    )
    rolled_spec = dataclasses.replace(
        grip_spec,
        lead=dataclasses.replace(grip_spec.lead, shaft_roll_deg=roll_deg),
        trail=dataclasses.replace(grip_spec.trail, shaft_roll_deg=roll_deg),
    )

    hand_groups: list[tuple[Mesh, tuple[float, float, float, float]]] = []
    hand_verts_list: list[np.ndarray] = []
    lead_z = 0.0
    trail_z = 0.0
    if mode in ("lead", "both"):
        parts_l = extract_allegro_hand_parts(rolled_spec, "lead", grip_radius_m=radius)
        g_l = _split_skin_and_tips(parts_l, 0.0, cfg)
        hand_groups.extend(g_l)
        vl = np.vstack([m.vertices for m, _ in g_l])
        hand_verts_list.append(vl)
        lead_z = float(vl[:, 2].mean())
    if mode in ("trail", "both"):
        shift = float(rolled_spec.hand_spacing_m) if mode == "both" else 0.0
        parts_t = extract_allegro_hand_parts(rolled_spec, "trail", grip_radius_m=radius)
        g_t = _split_skin_and_tips(parts_t, shift, cfg)
        hand_groups.extend(g_t)
        vt = np.vstack([m.vertices for m, _ in g_t])
        hand_verts_list.append(vt)
        trail_z = float(vt[:, 2].mean())

    hand_verts = np.vstack(hand_verts_list)
    scene_items = [*_club_scene_items(hand_verts, radius, theme), *hand_groups]

    right, cam_up, fwd = _camera_basis(view)
    origin = 0.5 * (hand_verts.min(axis=0) + hand_verts.max(axis=0))
    rel = hand_verts - origin
    hx = rel @ right
    hy = rel @ cam_up
    x_mid = float(0.5 * (hx.min() + hx.max()))
    y_mid = float(0.5 * (hy.min() + hy.max()))
    h_span = float(max(hx.max() - hx.min(), hy.max() - hy.min()))
    fill = theme.both_frame_fill if mode == "both" else theme.frame_fill
    half = h_span / (2.0 * fill)

    n_px = theme.frame_px
    image = np.broadcast_to(theme.background_rgb, (n_px, n_px, 3)).copy()
    zbuffer = np.full((n_px, n_px), np.inf, dtype=np.float64)
    for mesh, rgba in scene_items:
        off = mesh.vertices - origin
        px = ((off @ right - x_mid) / half + 1.0) * 0.5 * (n_px - 1)
        py = (1.0 - ((off @ cam_up - y_mid) / half + 1.0) * 0.5) * (n_px - 1)
        xy = np.column_stack((px, py))
        zd = -(off @ fwd)
        shaded = _shade_corners(
            mesh.vertices,
            mesh.faces,
            np.asarray(rgba[:3], dtype=np.float64),
            right,
            cam_up,
            fwd,
            theme,
        )
        _raster_into(image, zbuffer, xy, zd, mesh.faces, shaded)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, np.clip(image, 0.0, 1.0))

    hand_bounds = (float(hx.min()), float(hx.max()), float(hy.min()), float(hy.max()))
    view_bounds = (x_mid - half, x_mid + half, y_mid - half, y_mid + half)
    return {
        "file": out_path.name,
        "mode": mode,
        "view": view,
        "roll_deg": float(roll_deg),
        "width_px": n_px,
        "height_px": n_px,
        "hand_projected_bounds": hand_bounds,
        "viewport_bounds": view_bounds,
        "hand_fill_fraction": compute_hand_fill_fraction(hand_bounds, view_bounds),
        "lead_z_mean_m": lead_z,
        "trail_z_mean_m": trail_z,
    }


def build_evidence_receipt(
    renders: list[dict[str, Any]],
    *,
    spec: GripSpec | None = None,
    include_roll_audit: bool = True,
) -> dict[str, Any]:
    """Assemble the evidence receipt with per-roll pad-gap audit metrics."""
    base_spec = GripSpec.legacy() if spec is None else spec
    receipt: dict[str, Any] = {
        "backend": "allegro",
        "render_count": len(renders),
        "anthropometry_lead": measure_allegro_anthropometry(base_spec, "lead"),
        "pad_gaps_lead_m": finger_pad_contact_gaps(base_spec, "lead"),
        "pad_gaps_trail_m": finger_pad_contact_gaps(base_spec, "trail"),
        "penetration_lead": penetration_metrics(base_spec, "lead"),
        "penetration_trail": penetration_metrics(base_spec, "trail"),
        "renders": renders,
    }
    if include_roll_audit:
        by_roll: dict[str, dict[str, dict[str, float]]] = {}
        for roll in _ROLLS:
            roll_key = f"{int(roll):+d}" if roll != 0.0 else "0"
            rolled_spec = dataclasses.replace(
                base_spec,
                lead=dataclasses.replace(base_spec.lead, shaft_roll_deg=roll),
                trail=dataclasses.replace(base_spec.trail, shaft_roll_deg=roll),
            )
            by_roll[roll_key] = {
                "lead": finger_pad_contact_gaps(rolled_spec, "lead"),
                "trail": finger_pad_contact_gaps(rolled_spec, "trail"),
            }
        receipt["pad_gaps_by_roll_m"] = by_roll
    return receipt


def render_evidence_suite(out_dir: Path | None = None) -> dict[str, Any]:
    """Render all 27 (mode x view x roll) Allegro PNGs and write render_stats.json."""
    target_dir = Path(__file__).resolve().parent if out_dir is None else out_dir
    spec = GripSpec.legacy()
    renders: list[dict[str, Any]] = []
    for mode in _MODES:
        for view in _VIEWS:
            for roll in _ROLLS:
                tag = f"roll{int(roll):+04d}".replace("+", "p").replace("-", "m")
                fname = f"allegro_{mode}_{view}_{tag}.png"
                stats = render_single_view(
                    spec,
                    mode=mode,
                    view=view,
                    roll_deg=roll,
                    out_path=target_dir / fname,
                )
                stats["sha256"] = hashlib.sha256(
                    (target_dir / fname).read_bytes()
                ).hexdigest()
                renders.append(stats)

    receipt = build_evidence_receipt(renders, spec=spec, include_roll_audit=True)
    (target_dir / "render_stats.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Allegro hand visual evidence.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    receipt = render_evidence_suite(args.out_dir)
    sys.stdout.write(
        json.dumps({"render_count": receipt["render_count"]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
