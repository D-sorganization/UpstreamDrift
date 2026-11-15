"""Lead, trail, and both-hands 1024px smooth Z-buffer roll evidence."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
    hand_mesh_with_shaft_roll,
)
from src.shared.python.model_appearance.hand_backends.placement import (
    mesh_in_shared_visual_frame,
)
from src.shared.python.model_appearance.hand_backends.preview import (
    club_context_drawables,
    render_shaded_scene_rgb,
    skin_tone_rgb,
    view_rotation,
)
from src.shared.python.model_appearance.hand_backends.registry import build_hand_visual
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
)

_LOG = logging.getLogger(__name__)
ROLLS = (-20.0, 0.0, 20.0)
VIEWS = ("palm", "back", "side")
ROLES: tuple[HandRole, ...] = ("lead", "trail")


def _spec(roll_deg: float) -> GripSpec:
    return GripSpec.legacy(
        lead=HandGripSpec(shaft_roll_deg=roll_deg),
        trail=HandGripSpec(shaft_roll_deg=-0.5 * roll_deg),
    )


def _stamp_badge(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    draw.rectangle([16, 16, 16 + len(text) * 12 + 24, 54], fill=(28, 34, 46))
    draw.text((28, 22), text, fill=(245, 247, 250), font=font)
    return img


def _hand_fill_fraction(
    focus_verts: np.ndarray, rot: np.ndarray, target_fill: float
) -> float:
    cam = focus_verts @ rot.T
    du = float(cam[:, 0].max() - cam[:, 0].min())
    dv = float(cam[:, 1].max() - cam[:, 1].min())
    return round(max(du, dv) / max(max(du, dv) / target_fill, 1e-9), 4)


def render_roll_strip(output_dir: Path, backend_id: str = "procedural_skin") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = DEFAULT_HAND_VISUAL_SETTINGS
    panel_px = max(1024, settings.render_frame_px)
    fill = settings.render_frame_fill
    club = club_context_drawables(DEFAULT_VISUAL_GRIP_RADIUS_M)
    receipts: list[dict[str, Any]] = []

    for roll in ROLLS:
        spec = _spec(roll)
        for role in ROLES:
            angles = HandJointAngles.from_grip_spec(spec, role)
            vis = build_hand_visual(
                backend_id,
                spec,
                role,
                angles,
                grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                settings=settings,
            )
            mesh = hand_mesh_with_shaft_roll(vis.mesh, spec, role)
            color = skin_tone_rgb(settings.skin_material_for_role(role))
            drawables = [(mesh, color, 0.09), *club]

            strip = Image.new("RGB", (panel_px * len(VIEWS), panel_px), (244, 246, 249))
            for idx, view in enumerate(VIEWS):
                rot = view_rotation(view)
                rgb = render_shaded_scene_rgb(
                    drawables,
                    focus_vertices=mesh.vertices,
                    view_rot=rot,
                    width_px=panel_px,
                    height_px=panel_px,
                    target_fill=fill,
                )
                panel = _stamp_badge(
                    Image.fromarray(rgb),
                    f"{backend_id} · {role} · {view} ({roll:+.0f}°)",
                )
                strip.paste(panel, (idx * panel_px, 0))
            out_strip = output_dir / f"{backend_id}_{role}_roll_{roll:+.0f}.png"
            strip.save(out_strip)
            _LOG.info("Wrote %s", out_strip)
            receipts.append(
                {
                    "file": out_strip.name,
                    "width": strip.width,
                    "height": strip.height,
                    "hand_fill": _hand_fill_fraction(
                        mesh.vertices, view_rotation("palm"), fill
                    ),
                    "club_in_frame": True,
                }
            )

            iso_rgb = render_shaded_scene_rgb(
                drawables,
                focus_vertices=mesh.vertices,
                view_rot=view_rotation("iso"),
                width_px=panel_px,
                height_px=panel_px,
                target_fill=fill,
            )
            iso_img = _stamp_badge(
                Image.fromarray(iso_rgb), f"{backend_id} · {role} · iso ({roll:+.0f}°)"
            )
            out_iso = output_dir / f"{backend_id}_{role}_iso_roll_{roll:+.0f}.png"
            iso_img.save(out_iso)
            _LOG.info("Wrote %s", out_iso)
            receipts.append(
                {
                    "file": out_iso.name,
                    "width": iso_img.width,
                    "height": iso_img.height,
                    "hand_fill": _hand_fill_fraction(
                        mesh.vertices, view_rotation("iso"), fill
                    ),
                    "club_in_frame": True,
                }
            )

        both_verts: list[np.ndarray] = []
        both_drawables: list[tuple[Any, tuple[float, float, float], float]] = []
        for role in ROLES:
            angles = HandJointAngles.from_grip_spec(spec, role)
            vis = build_hand_visual(
                backend_id,
                spec,
                role,
                angles,
                grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                settings=settings,
            )
            rolled = hand_mesh_with_shaft_roll(vis.mesh, spec, role)
            shared = mesh_in_shared_visual_frame(rolled, spec, role)
            both_verts.append(shared.vertices)
            both_drawables.append(
                (shared, skin_tone_rgb(settings.skin_material_for_role(role)), 0.09)
            )
        focus_both = np.vstack(both_verts)
        both_drawables.extend(club)

        strip_both = Image.new(
            "RGB", (panel_px * len(VIEWS), panel_px), (244, 246, 249)
        )
        for idx, view in enumerate(VIEWS):
            rot = view_rotation(view)
            rgb = render_shaded_scene_rgb(
                both_drawables,
                focus_vertices=focus_both,
                view_rot=rot,
                width_px=panel_px,
                height_px=panel_px,
                target_fill=fill,
            )
            panel = _stamp_badge(
                Image.fromarray(rgb), f"{backend_id} · both · {view} ({roll:+.0f}°)"
            )
            strip_both.paste(panel, (idx * panel_px, 0))
        out_both = output_dir / f"{backend_id}_both_hands_roll_{roll:+.0f}.png"
        strip_both.save(out_both)
        _LOG.info("Wrote %s", out_both)
        receipts.append(
            {
                "file": out_both.name,
                "width": strip_both.width,
                "height": strip_both.height,
                "hand_fill": _hand_fill_fraction(
                    focus_both, view_rotation("palm"), fill
                ),
                "club_in_frame": True,
            }
        )

        both_iso_rgb = render_shaded_scene_rgb(
            both_drawables,
            focus_vertices=focus_both,
            view_rot=view_rotation("iso"),
            width_px=panel_px,
            height_px=panel_px,
            target_fill=fill,
        )
        both_iso = _stamp_badge(
            Image.fromarray(both_iso_rgb), f"{backend_id} · both · iso ({roll:+.0f}°)"
        )
        out_both_iso = output_dir / f"{backend_id}_both_iso_roll_{roll:+.0f}.png"
        both_iso.save(out_both_iso)
        _LOG.info("Wrote %s", out_both_iso)
        receipts.append(
            {
                "file": out_both_iso.name,
                "width": both_iso.width,
                "height": both_iso.height,
                "hand_fill": _hand_fill_fraction(
                    focus_both, view_rotation("iso"), fill
                ),
                "club_in_frame": True,
            }
        )

    receipt_path = output_dir / f"{backend_id}_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "backend_id": backend_id,
                "panel_px": panel_px,
                "target_fill": fill,
                "frames": receipts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/development/full_body_models/evidence/hand_backends"),
    )
    parser.add_argument("--backend", default="procedural_skin")
    args = parser.parse_args(argv)
    render_roll_strip(args.output_dir, args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
