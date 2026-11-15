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
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.grip_hands import assembly_meshes_with_hands
from src.shared.python.model_appearance.grip_preview_matplotlib import (
    GripMeshPreviewStyle,
    add_mesh_preview,
    apply_equal_axis_limits,
)

OUT = Path(__file__).resolve().parent


def _spec_for_roll(roll_deg: float) -> GripSpec:
    if roll_deg == 0.0:
        return GripSpec.legacy()
    return replace(
        GripSpec.legacy(),
        lead=HandGripSpec(shaft_roll_deg=roll_deg),
        trail=HandGripSpec(shaft_roll_deg=-roll_deg),
    )


def _render(path: Path, *, elev: float, azim: float, title: str, roll_deg: float) -> None:
    club = ClubAssembly("iron7", 0.95, 0.25, 0.0065)
    spec = _spec_for_roll(roll_deg)
    meshes = assembly_meshes_with_hands(club, spec.to_grip_pose(), grip_spec=spec)
    colours = {
        "shaft": (0.45, 0.45, 0.5),
        "grip": (0.25, 0.25, 0.28),
        "head": (0.55, 0.55, 0.58),
        "lead_hand": (0.78, 0.62, 0.54),
        "trail_hand": (0.72, 0.58, 0.50),
    }
    fig = plt.figure(figsize=(5.2, 5.2), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    base = GripMeshPreviewStyle()
    for name, mesh in meshes.items():
        style = replace(base, base_rgb=colours.get(name, base.base_rgb))
        add_mesh_preview(ax, mesh, style=style)
        all_pts.append(mesh.vertices)
    apply_equal_axis_limits(ax, np.vstack(all_pts), style=base)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    rolls = (0.0, 20.0, -20.0)
    views = (
        ("iso", 25.0, -55.0),
        ("butt", 12.0, 180.0),
        ("face", 8.0, 90.0),
    )
    artifacts: list[str] = []
    for roll in rolls:
        tag = f"roll_{abs(int(roll))}{'p' if roll >= 0 else 'm'}"
        for vname, elev, azim in views:
            fname = f"grip_shadow_{vname}_{tag}.png"
            _render(
                OUT / fname,
                elev=elev,
                azim=azim,
                title=f"Shadow grip {vname} {roll:+.0f}° shaft roll",
                roll_deg=roll,
            )
            artifacts.append(fname)
    receipt = {"artifacts": sorted(artifacts), "shaft_rolls_deg": list(rolls)}
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()