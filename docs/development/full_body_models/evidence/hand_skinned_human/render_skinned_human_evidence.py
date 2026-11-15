#!/usr/bin/env python3
"""Evidence stills for the skinned_human hand visual backend (#15122)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_preview_matplotlib import (
    GripMeshPreviewStyle,
    add_mesh_preview,
    apply_equal_axis_limits,
)
from src.shared.python.model_appearance.hand_backends import build_hand_visual
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    legacy_neutral_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    default_settings,
)

_OUT = Path(__file__).resolve().parent
_GRIP_R = 0.0145
_ROLLS = (-20.0, 0.0, 20.0)
_DPI = 128
_PX = 1024
_HAND_STYLE = GripMeshPreviewStyle(
    base_rgb=default_settings().skin_rgb,
    smooth_normals=True,
    margin_fraction=0.32,
    min_half_width_m=0.032,
)
_SHAFT_STYLE = GripMeshPreviewStyle(
    base_rgb=(0.22, 0.24, 0.28),
    ambient=0.45,
    diffuse=0.4,
    smooth_normals=True,
    margin_fraction=0.32,
    min_half_width_m=0.032,
)


def _grip_cylinder(radius: float, *, segments: int = 64) -> Mesh:
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    z0, z1 = -0.04, 0.16
    verts, faces = [], []
    for z in (z0, z1):
        ring = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z)]
        )
        base = len(verts)
        verts.append(ring)
        if base > 0:
            for i in range(segments):
                j = (i + 1) % segments
                faces.append([base - segments + i, base - segments + j, base + i])
                faces.append([base - segments + j, base + j, base + i])
    return Mesh(np.vstack(verts), np.asarray(faces, dtype=np.int64))


def _render_view(
    meshes: list[tuple[Mesh, GripMeshPreviewStyle]],
    *,
    elev: float,
    azim: float,
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(_PX / _DPI, _PX / _DPI), dpi=_DPI)
    ax = fig.add_subplot(111, projection="3d")
    frame_pts = np.vstack([mesh.vertices for mesh, _ in meshes])
    for mesh, style in meshes:
        add_mesh_preview(ax, mesh, style=style)
    apply_equal_axis_limits(ax, frame_pts, style=_HAND_STYLE)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="#e8e4dc", fontsize=11)
    ax.set_axis_off()
    fig.patch.set_facecolor("#12151a")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shaft = _grip_cylinder(_GRIP_R)
    views = (("butt", 12, -35), ("side", 6, 24), ("top", 82, -48))
    for side in ("lead", "trail"):
        for roll in _ROLLS:
            hand_key = "lead" if side == "lead" else "trail"
            spec = GripSpec.legacy(**{hand_key: HandGripSpec(shaft_roll_deg=roll)})
            angles = legacy_neutral_joint_angles(spec, side)
            hand = build_hand_visual(spec, side, angles, grip_radius_m=_GRIP_R).mesh
            for view_name, elev, azim in views:
                out = args.out / f"{side}_{view_name}_roll_{roll:+.0f}.png"
                _render_view(
                    [(shaft, _SHAFT_STYLE), (hand, _HAND_STYLE)],
                    elev=elev,
                    azim=azim,
                    out_path=out,
                    title=f"{side} hand  roll={roll:+.0f}°  ({view_name})",
                )
    both_spec = GripSpec.legacy(
        lead=HandGripSpec(shaft_roll_deg=0.0),
        trail=HandGripSpec(shaft_roll_deg=0.0),
    )
    lead = build_hand_visual(
        both_spec,
        "lead",
        legacy_neutral_joint_angles(both_spec, "lead"),
        grip_radius_m=_GRIP_R,
    ).mesh
    trail = build_hand_visual(
        both_spec,
        "trail",
        legacy_neutral_joint_angles(both_spec, "trail"),
        grip_radius_m=_GRIP_R,
    ).mesh
    for view_name, elev, azim in views:
        out = args.out / f"both_{view_name}_roll_0.png"
        _render_view(
            [(shaft, _SHAFT_STYLE), (lead, _HAND_STYLE), (trail, _HAND_STYLE)],
            elev=elev,
            azim=azim,
            out_path=out,
            title=f"Both hands  roll=0°  ({view_name})",
        )


if __name__ == "__main__":
    main()