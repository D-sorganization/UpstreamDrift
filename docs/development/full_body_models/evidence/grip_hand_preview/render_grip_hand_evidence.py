"""Evidence stills for procedural grip hands (#15141).

Lead and trail hands alone (palm, back, side) and a grip-region crop.

Run from the repo root::

    MPLBACKEND=Agg PYTHONPATH=.:src python3 \\
        docs/development/full_body_models/evidence/grip_hand_preview/render_grip_hand_evidence.py \\
        --out docs/development/full_body_models/evidence/grip_hand_preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.geometry import concat_meshes
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_hands import (
    assembly_meshes_with_hands,
    grip_hand_frame,
    grip_radius_at,
)
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand
from src.shared.python.model_appearance.grip_preview_matplotlib import (
    GripMeshPreviewStyle,
    add_mesh_preview,
    apply_equal_axis_limits,
    axis_half_widths,
    preview_figure_size,
)

_ROOT = Path(__file__).resolve().parents[4]
_VIEWS: dict[str, tuple[float, float]] = {
    "palm": (18.0, -90.0),
    "back": (18.0, 90.0),
    "side": (12.0, 0.0),
}


def _club() -> ClubAssembly:
    return ClubAssembly("iron7", 0.95, 0.25, 0.0065)


def _hand_in_club_frame(hand: Hand, club: ClubAssembly):
    radius = grip_radius_at(club, DEFAULT_GRIP_POSE.distance_below_butt_m(hand))
    baked = build_grip_hand_mesh(hand, grip_radius_m=radius, pose=DEFAULT_GRIP_POSE)
    frame = grip_hand_frame(club, hand, DEFAULT_GRIP_POSE)
    verts = (frame[:3, :3] @ baked.mesh.vertices.T).T + frame[:3, 3]
    return baked.mesh, verts, frame[:3, 3]


def _legacy_limits(ax, points: np.ndarray, centre: np.ndarray) -> None:
    """Pre-fix independent per-axis club limits (distorted, for before stills)."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    pad = 0.06
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect((1, 1, 1))


def _render_hand_view(
    hand: Hand,
    view: str,
    out: Path,
    *,
    legacy_camera: bool,
) -> dict[str, float]:
    club = _club()
    hand_mesh, hand_world, centre = _hand_in_club_frame(hand, club)
    elev, azim = _VIEWS[view]
    fig = plt.figure(figsize=(5.0, 5.0), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    style = GripMeshPreviewStyle()
    from src.shared.python.model_appearance.geometry import Mesh

    mesh = Mesh(hand_world, hand_mesh.faces)
    add_mesh_preview(ax, mesh, style=style)
    if legacy_camera:
        _legacy_limits(ax, hand_world, centre)
    else:
        half = apply_equal_axis_limits(ax, hand_world, centre=centre, style=style)
        fig.set_size_inches(*preview_figure_size(half, dpi=140))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f"{hand.value} hand — {view}", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    x, y, z = axis_half_widths(ax)
    return {
        "half_x_m": x,
        "half_y_m": y,
        "half_z_m": z,
        "equal_aspect": abs(x - y) < 1e-9 and abs(y - z) < 1e-9,
    }


def _render_grip_region(out: Path, *, legacy_camera: bool) -> dict[str, object]:
    club = _club()
    parts = assembly_meshes_with_hands(club, DEFAULT_GRIP_POSE)
    mesh = concat_meshes(parts["lead_hand"], parts["trail_hand"], parts["grip"])
    verts = mesh.vertices
    centre = verts.mean(axis=0)
    fig = plt.figure(figsize=(5.5, 5.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    style = GripMeshPreviewStyle(base_rgb=(0.68, 0.55, 0.46))
    add_mesh_preview(ax, mesh, style=style)
    if legacy_camera:
        _legacy_limits(ax, verts, centre)
    else:
        apply_equal_axis_limits(ax, verts, centre=centre, style=style)
    ax.view_init(elev=15.0, azim=-62.0)
    ax.set_title("Grip region — lead + trail + rubber", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    x, y, z = axis_half_widths(ax)
    return {"half_x_m": x, "half_y_m": y, "half_z_m": z, "equal_aspect": x == y == z}


def render_all(out_dir: Path, *, legacy_camera: bool) -> dict[str, object]:
    tag = "before" if legacy_camera else "after"
    receipt: dict[str, object] = {
        "tag": tag,
        "legacy_camera": legacy_camera,
        "stills": {},
    }
    for hand in (Hand.LEAD, Hand.TRAIL):
        for view in _VIEWS:
            name = f"{tag}_{hand.value}_{view}.png"
            path = out_dir / name
            meta = _render_hand_view(hand, view, path, legacy_camera=legacy_camera)
            receipt["stills"][name] = meta
    grip_name = f"{tag}_grip_region.png"
    receipt["stills"][grip_name] = _render_grip_region(
        out_dir / grip_name, legacy_camera=legacy_camera
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--legacy-camera",
        action="store_true",
        help="Use pre-fix per-axis limits for before stills",
    )
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    receipt = render_all(args.out, legacy_camera=args.legacy_camera)
    if args.write_receipt:
        name = "render_stats_before.json" if args.legacy_camera else "render_stats_after.json"
        (args.out / name).write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    sys.stdout.write(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
