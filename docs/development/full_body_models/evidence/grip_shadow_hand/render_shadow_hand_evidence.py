#!/usr/bin/env python3
"""Render Shadow hand grip evidence at shaft roll 0 and ±20° (#15122)."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    GRIP_BUTT_RADIUS_M,
    assembly_meshes,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.club_shaft_mesh import tapered_tube
from src.shared.python.model_appearance.grip_hands import grip_hand_meshes
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE
from src.shared.python.model_appearance.grip_preview_matplotlib import (
    GripMeshPreviewStyle,
    add_mesh_preview,
    apply_equal_axis_limits,
)

OUT = Path(__file__).resolve().parent
_DPI = 1024
_FIG_IN = 8.0
_CROP_QUANTILE = (0.06, 0.94)


def _crop_focus_points(points: np.ndarray) -> np.ndarray:
    """Drop extreme verts (long proximal segments) so the hand fills the frame."""
    pts = np.asarray(points, dtype=float)
    lo = np.quantile(pts, _CROP_QUANTILE[0], axis=0)
    hi = np.quantile(pts, _CROP_QUANTILE[1], axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    trimmed = pts[keep]
    return trimmed if len(trimmed) > 64 else pts


def _spec_for_roll(roll_deg: float) -> GripSpec:
    if roll_deg == 0.0:
        return GripSpec.legacy()
    return replace(
        GripSpec.legacy(),
        lead=HandGripSpec(shaft_roll_deg=roll_deg),
        trail=HandGripSpec(shaft_roll_deg=-roll_deg),
    )


def _meshes_for_view(
    club: ClubAssembly,
    spec: GripSpec,
    *,
    subset: str,
) -> dict[str, object]:
    pose = spec.to_grip_pose()
    hands = grip_hand_meshes(club, pose, grip_spec=spec)
    hand_verts = []
    for label in hands:
        if subset == "lead" and label != "lead_hand":
            continue
        if subset == "trail" and label != "trail_hand":
            continue
        hand_verts.append(hands[label].vertices)
    if not hand_verts:
        raise ValueError(f"no hands for subset {subset!r}")
    hand_stack = np.vstack(list(hand_verts))
    y_mid = float(hand_stack[:, 1].mean())
    z = club.axis_offset_m
    span = float(np.ptp(hand_stack[:, 1])) + 0.04
    grip_stub = tapered_tube(
        np.array([0.0, y_mid + span / 2, z]),
        np.array([0.0, y_mid - span / 2, z]),
        GRIP_BUTT_RADIUS_M,
        GRIP_BUTT_RADIUS_M,
    )
    meshes: dict[str, Mesh] = {"grip": grip_stub}
    for k, m in hands.items():
        if (
            subset == "both"
            or (subset == "lead" and k == "lead_hand")
            or (subset == "trail" and k == "trail_hand")
        ):
            meshes[k] = m
    return meshes


def _render(
    path: Path,
    *,
    elev: float,
    azim: float,
    title: str,
    roll_deg: float,
    subset: str,
) -> None:
    club = ClubAssembly("iron7", 0.95, 0.25, 0.0065)
    spec = _spec_for_roll(roll_deg)
    meshes = _meshes_for_view(club, spec, subset=subset)
    colours = {
        "grip": (0.22, 0.20, 0.20),
        "lead_hand": (0.78, 0.62, 0.54),
        "trail_hand": (0.72, 0.58, 0.50),
    }
    fig = plt.figure(figsize=(_FIG_IN, _FIG_IN), dpi=_DPI)
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    base = GripMeshPreviewStyle(
        ambient=0.32,
        diffuse=0.62,
        specular=0.08,
        light_direction=(0.4, 0.55, 0.75),
        frame_fill_fraction=0.72,
        edge_rgba=None,
        edge_linewidth=0.0,
    )
    for name, mesh in meshes.items():
        style = replace(base, base_rgb=colours.get(name, base.base_rgb))
        add_mesh_preview(ax, mesh, style=style)
        all_pts.append(mesh.vertices)
    hand_pts = np.vstack(
        [mesh.vertices for name, mesh in meshes.items() if name.endswith("_hand")]
    )
    focus = _crop_focus_points(hand_pts)
    crop_pts = np.vstack([focus, meshes["grip"].vertices])
    centre = crop_pts.mean(axis=0)
    crop_style = replace(base, frame_fill_fraction=0.78)
    apply_equal_axis_limits(ax, crop_pts, centre=centre, style=crop_style)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main() -> None:
    rolls = (0.0, 20.0, -20.0)
    views = (
        ("iso", 25.0, -55.0),
        ("butt", 12.0, 180.0),
        ("face", 8.0, 90.0),
    )
    subsets = ("both", "lead", "trail")
    artifacts: list[str] = []
    for roll in rolls:
        tag = f"roll_{abs(int(roll))}{'p' if roll >= 0 else 'm'}"
        for subset in subsets:
            for vname, elev, azim in views:
                fname = f"grip_shadow_{subset}_{vname}_{tag}.png"
                _render(
                    OUT / fname,
                    elev=elev,
                    azim=azim,
                    title=f"Shadow {subset} {vname} {roll:+.0f}°",
                    roll_deg=roll,
                    subset=subset,
                )
                artifacts.append(fname)
    receipt = {
        "artifacts": sorted(artifacts),
        "shaft_rolls_deg": list(rolls),
        "subsets": list(subsets),
        "dpi": _DPI,
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
