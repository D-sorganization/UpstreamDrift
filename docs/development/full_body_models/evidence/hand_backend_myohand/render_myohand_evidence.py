#!/usr/bin/env python3
"""MyoHand skeleton reference evidence: bones on club, shaft roll 0 / ±20°."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    assembly_meshes,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.myohand import (
    build_myohand_bone_meshes,
)
from src.shared.python.model_appearance.hand_backends.render_evidence import (
    LitRenderSettings,
    render_scene_png,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    default_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.skeleton_export import (
    export_bilateral_myohand_skeleton,
)

OUT = Path(__file__).resolve().parent
VIEWS = ("front", "side", "top", "grip_oblique")
ROLLS = (0.0, -20.0, 20.0)


def _zero_roll_spec(spec: GripSpec) -> GripSpec:
    return replace(
        spec,
        lead=replace(spec.lead, shaft_roll_deg=0.0),
        trail=replace(spec.trail, shaft_roll_deg=0.0),
    )


def _place_hand_on_club(
    club: ClubAssembly, spec: GripSpec, side: str
) -> dict[str, Mesh]:
    hand = Hand.LEAD if side == "lead" else Hand.TRAIL
    # Pass zero-roll pose so shaft_roll_deg is applied once inside build_myohand_bone_meshes.
    frame = grip_hand_frame(club, hand, _zero_roll_spec(spec).to_grip_pose())
    local = build_myohand_bone_meshes(spec, side)  # type: ignore[arg-type]
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
    club_meshes = assembly_meshes(club)
    parts: dict[str, Mesh] = {
        k: club_meshes[k] for k in ("grip", "shaft") if k in club_meshes
    }
    for side in sides:
        parts.update(_place_hand_on_club(club, spec, side))
    return parts


def main() -> None:
    club = ClubAssembly("iron7", 0.95, 0.25, 0.0065)
    base = GripSpec.neutral()
    cfg = default_myohand_settings()
    lit = LitRenderSettings(
        image_px=1024,
        frame_fill=cfg.render_frame_fill,
        bone_rgb=cfg.bone_rgb,
        grip_rgb=cfg.grip_rgb,
        shaft_rgb=cfg.shaft_rgb,
        butt_cap_rgb=cfg.butt_cap_rgb,
        background_rgb=cfg.background_rgb,
    )
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
        "frame_fill": lit.frame_fill,
        "artifacts": sorted(artifacts),
    }
    (OUT / "render_stats.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
