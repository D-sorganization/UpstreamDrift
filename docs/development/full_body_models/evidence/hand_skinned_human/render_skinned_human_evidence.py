#!/usr/bin/env python3
"""Evidence stills of the skinned MakeHuman hand on the club (epic #15150).

Lead, trail and both hands, from the palm, back, thumb side and butt end, at
shaft roll 0 and +/-20 degrees. The hand is built with the roll in its grip
spec (the backend rolls it); the camera stays where the unrolled hand puts it.

The frames come from the shared renderer (``hand_backends/render.py``, PR
#15154). Until that lands on this branch, point ``--renderer-root`` (or
``HAND_RENDERER_ROOT``) at a checkout of its branch.

    PYTHONPATH=. MUJOCO_GL=egl python \\
        docs/development/full_body_models/evidence/hand_skinned_human/render_skinned_human_evidence.py \\
        --renderer-root ../hand-renderer-15154
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

import src.shared.python.model_appearance.hand_backends as hand_backends
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
    max_grip_cylinder_penetration_m,
    forearm_spread_deg,
    hand_pair_penetration_m,
    skinning_defects,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    build_grip_hands,
    build_hand_visual,
)

OUT = Path(__file__).resolve().parent
SETTINGS = json.loads((OUT / "evidence_settings.json").read_text(encoding="utf-8"))
_PACKAGE = Path("src/shared/python/model_appearance/hand_backends")


def _load_renderer(root: str | None):
    """Import the shared renderer, from this tree or from ``root``."""
    if not (Path(hand_backends.__file__).parent / "render.py").is_file():
        if not root or not (Path(root) / _PACKAGE / "render.py").is_file():
            raise SystemExit("shared renderer not found: pass --renderer-root")
        hand_backends.__path__.append(str(Path(root).resolve() / _PACKAGE))
    from src.shared.python.model_appearance.hand_backends import render

    return render


def _spec(roll_deg: float) -> GripSpec:
    base = GripSpec.legacy()
    return dataclasses.replace(
        base,
        lead=dataclasses.replace(base.lead, shaft_roll_deg=roll_deg),
        trail=dataclasses.replace(base.trail, shaft_roll_deg=roll_deg),
    )


def _basis(result) -> tuple[np.ndarray, np.ndarray]:
    """Thumb-side and back-of-hand directions of an unrolled result."""
    rotation = result.frame_from_rest[:3, :3]
    return rotation @ np.array([1.0, 0.0, 0.0]), rotation @ np.array([0.0, 0.0, -1.0])


def _roll_tag(roll_deg: float) -> str:
    return f"roll{'m' if roll_deg < 0 else 'p'}{abs(round(roll_deg)):02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer-root", default=os.environ.get("HAND_RENDERER_ROOT"))
    args = parser.parse_args()
    render = _load_renderer(args.renderer_root)
    settings = render.load_hand_render_settings().with_overrides(
        frame_px=int(SETTINGS["frame_px"])
    )
    quality = load_hand_visual_quality()
    radius = float(quality["grip_radius_m"])
    views = tuple(SETTINGS["views"])
    report: dict[str, object] = {"frames": {}, "hands": {}, "pairs": {}}

    def skin(mesh: Mesh, rgb) -> object:
        return render.ScenePart(mesh, (*rgb, 1.0), "skin")

    def shoot(name: str, parts, view: str, basis, side: str) -> None:
        z = np.concatenate([p.mesh.vertices[:, 2] for p in parts])
        club = render.build_club_parts(
            float(z.min()), float(z.max()), settings=settings, grip_radius_m=radius
        )
        frame = render.render_scene(
            [*parts, *club], view, basis=basis, settings=settings, side=side
        )
        render.save_png(frame, OUT / f"{name}.png")
        report["frames"][name] = {
            "engine": frame.engine,
            "fill": round(float(frame.fill_fraction), 3),
        }

    flat = {
        side: build_hand_visual(_spec(0.0), side, grip_radius_m=radius)
        for side in ("lead", "trail")
    }
    for roll in SETTINGS["rolls_deg"]:
        hands = {
            side: build_hand_visual(_spec(float(roll)), side, grip_radius_m=radius)
            for side in ("lead", "trail")
        }
        tag = _roll_tag(float(roll))
        for side, result in hands.items():
            defects = skinning_defects(
                result, min_area_m2=float(quality["min_triangle_area_m2"])
            )
            report["hands"][f"{side}_{tag}"] = {
                "max_penetration_mm": round(
                    1e3 * max_grip_cylinder_penetration_m(result.mesh.vertices, radius),
                    3,
                ),
                "contact_gaps_mm": {
                    k: round(1e3 * v, 2) for k, v in result.contact_gaps_m.items()
                },
                "joint_angles_deg": {
                    k: round(float(np.degrees(v)), 1)
                    for k, v in result.joint_angles.as_mapping().items()
                },
                **defects,
            }
            for view in views:
                shoot(
                    f"skinned_{side}_{view}_{tag}",
                    [skin(result.mesh, result.skin_rgb)],
                    view,
                    _basis(flat[side]),
                    side,
                )
        # Both hands: the trail hand is posed around the grip and the lead hand.
        both = build_grip_hands(_spec(float(roll)), grip_radius_m=radius)
        report["pairs"][tag] = {
            "pair_penetration_mm": round(
                1e3 * hand_pair_penetration_m(both["lead"], both["trail"]), 3
            ),
            "forearm_spread_deg": round(
                forearm_spread_deg(both["lead"], both["trail"]), 1
            ),
            "trail_contact_gaps_mm": {
                k: round(1e3 * v, 2) for k, v in both["trail"].contact_gaps_m.items()
            },
        }
        pair = [skin(r.mesh, r.skin_rgb) for r in both.values()]
        for view in views:
            shoot(
                f"skinned_both_{view}_{tag}", pair, view, _basis(flat["lead"]), "lead"
            )
    (OUT / "render_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"wrote {len(report['frames'])} frames to {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
