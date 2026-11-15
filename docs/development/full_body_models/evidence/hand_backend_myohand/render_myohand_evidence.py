#!/usr/bin/env python3
"""Render MyoHand grip visual evidence (lead/trail, shaft roll 0/±20°, three views)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.geometry import concat_meshes
from src.shared.python.model_appearance.hand_backends.myohand import build_hand_visual

OUT = Path(__file__).resolve().parent
ROLLS = (0.0, -20.0, 20.0)
VIEWS = (
    ("front", 0, 0),
    ("side", 0, 90),
    ("top", 90, 0),
)


def _plot_hand(
    ax, verts: np.ndarray, faces: np.ndarray, elev: float, azim: float
) -> None:
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color=(0.82, 0.68, 0.58),
        linewidth=0.05,
        edgecolor=(0.55, 0.45, 0.38),
        shade=True,
        antialiased=True,
    )
    ax.set_box_aspect([1, 1, 1])
    centre = verts.mean(0)
    half = float(np.ptp(verts, axis=0).max()) * 0.55
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")


def main() -> None:
    artifacts: list[str] = []
    base = GripSpec.neutral()

    for side in ("lead", "trail"):
        for roll in ROLLS:
            if side == "lead":
                spec = replace(base, lead=replace(base.lead, shaft_roll_deg=roll))
            else:
                spec = replace(base, trail=replace(base.trail, shaft_roll_deg=roll))
            meshes = build_hand_visual(spec, side)  # type: ignore[arg-type]
            merged = concat_meshes(*meshes.values())
            for view_name, elev, azim in VIEWS:
                fig = plt.figure(figsize=(5, 5), dpi=140)
                ax = fig.add_subplot(111, projection="3d")
                _plot_hand(ax, merged.vertices, merged.faces, elev, azim)
                roll_tag = f"{int(roll):+d}".replace("+", "p").replace("-", "m")
                fname = f"{side}_{view_name}_roll_{roll_tag}.png"
                path = OUT / fname
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                artifacts.append(fname)

    fig = plt.figure(figsize=(8, 5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    for side, tone in (("lead", (0.82, 0.68, 0.58)), ("trail", (0.78, 0.64, 0.55))):
        meshes = build_hand_visual(base, side)  # type: ignore[arg-type]
        merged = concat_meshes(*meshes.values())
        ax.plot_trisurf(
            merged.vertices[:, 0],
            merged.vertices[:, 1],
            merged.vertices[:, 2],
            triangles=merged.faces,
            color=tone,
            linewidth=0.05,
            edgecolor=(0.5, 0.42, 0.36),
            shade=True,
        )
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=15, azim=-55)
    ax.axis("off")
    both_name = "both_hands_grip_oblique_roll_0.png"
    fig.savefig(OUT / both_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    artifacts.append(both_name)

    receipt = {
        "backend": "myohand",
        "spec": "GripSpec.neutral",
        "rolls_deg": list(ROLLS),
        "artifacts": sorted(artifacts),
    }
    (OUT / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
