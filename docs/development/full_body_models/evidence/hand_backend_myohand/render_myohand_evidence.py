#!/usr/bin/env python3
"""MyoHand skeleton reference evidence: bones on club, shaft roll 0 / ±20°."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.club_assembly import ClubAssembly, assembly_meshes
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.myohand import build_hand_visual
from src.shared.python.model_appearance.hand_backends.render_evidence import (
    LitRenderSettings,
    render_scene_png,
)
from src.shared.python.model_appearance.hand_backends.skeleton_export import (
    export_bilateral_myohand_skeleton,
)

OUT = Path(__file__).resolve().parent
VIEWS = ("front", "side", "top", "grip_oblique")
ROLLS = (0.0, -20.0, 20.0)


def _place_hand_on_club(club: ClubAssembly, spec: GripSpec, side: str) -> dict[str, Mesh]:
    hand = Hand.LEAD if side == "lead" else Hand.TRAIL
    pose = spec.to_grip_pose()
    frame = grip_hand_frame(club, hand, pose)
    local = build_hand_visual(spec, side)  # type: ignore[arg-type]
    rot, trans = frame[:3, :3], frame[:3, 3]
    return {
        f"{side}_{name}": Mesh(mesh.vertices @ rot.T + trans, mesh.faces)
        for name, mesh in local.items()
    }


def _scene_meshes(
    club: ClubAssembly,
    spec: GripSpec,
    *,
    sides: tuple[str, ...] = ("lead", "trail"),
) -> dict[str, Mesh]:
    parts = {k: v for k, v in assembly_meshes(club).items() if k == "grip"}
    for side in sides:
        parts.update(_place_hand_on_club(club, spec, side))
    return parts


def main() -> None:
    club = ClubAssembly("iron7", 0.95, 0.25, 0.0065)
    base = GripSpec.neutral()
    lit = LitRenderSettings(image_px=1024)
    artifacts: list[str] = []

    skeleton = export_bilateral_myohand_skeleton(base)
    sk_path = OUT / "myohand_skeleton_grip_spec_neutral.json"
    sk_path.write_text(json.dumps(skeleton, indent=2) + "\n", encoding="utf-8")
    artifacts.append(sk_path.name)

    for side in ("lead", "trail"):
        for roll in ROLLS:
            if side == "lead":
                spec = replace(base, lead=replace(base.lead, shaft_roll_deg=roll))
            else:
                spec = replace(base, trail=replace(base.trail, shaft_roll_deg=roll))
            meshes = _scene_meshes(club, spec, sides=(side,))
            for view in VIEWS:
                roll_tag = f"{int(roll):+d}".replace("+", "p").replace("-", "m")
                fname = f"{side}_{view}_club_roll_{roll_tag}.png"
                render_scene_png(OUT / fname, meshes, view=view, settings=lit)
                artifacts.append(fname)

    for roll in ROLLS:
        spec = replace(
            base,
            lead=replace(base.lead, shaft_roll_deg=roll),
            trail=replace(base.trail, shaft_roll_deg=roll),
        )
        meshes = _scene_meshes(club, spec)
        for view in ("grip_oblique", "front", "side"):
            roll_tag = f"{int(roll):+d}".replace("+", "p").replace("-", "m")
            fname = f"both_{view}_club_roll_{roll_tag}.png"
            render_scene_png(OUT / fname, meshes, view=view, settings=lit)
            artifacts.append(fname)

    receipt = {
        "backend": "myohand_skeleton",
        "research_only": True,
        "asset_note": (
            "myo_sim: STL bone meshes only; class=skin MuJoCo capsules are not a "
            "continuous surface mesh and are excluded"
        ),
        "spec": "GripSpec.neutral",
        "rolls_deg": list(ROLLS),
        "image_px": lit.image_px,
        "artifacts": sorted(artifacts),
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
