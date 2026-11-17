"""Lead, trail, and both-hands 1024px smooth Z-buffer roll evidence."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.hand_backends.types import HandSide
from src.shared.python.model_appearance.hand_backends.placement import (
    mesh_in_shared_visual_frame,
)
from src.shared.python.model_appearance.hand_backends.preview import (
    club_context_drawables,
    render_shaded_scene_rgb,
    skin_tone_rgb,
    view_rotation,
)
from src.shared.python.model_appearance.hand_backends.comparison import (
    build_for_comparison,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
)

_LOG = logging.getLogger(__name__)
ROLLS = (-20.0, 0.0, 20.0)
VIEWS = ("palm", "back", "side")
ROLES: tuple[HandSide, ...] = ("lead", "trail")
SIZE_SKIN_VARIANTS = (
    ("small", 0.88, "skin_1"),
    ("base", 1.0, "skin_7"),
    ("large", 1.15, "skin_tan"),
)


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


@dataclass(frozen=True)
class _RollRenderJob:
    """Output location and framing shared by every roll-strip frame."""

    output_dir: Path
    backend_id: str
    panel_px: int
    fill: float


def _render_view(
    job: _RollRenderJob,
    drawables: list[Any],
    focus: np.ndarray,
    view: str,
    label: str,
) -> Image.Image:
    rgb = render_shaded_scene_rgb(
        drawables,
        focus_vertices=focus,
        view_rot=view_rotation(view),
        width_px=job.panel_px,
        height_px=job.panel_px,
        target_fill=job.fill,
    )
    return _stamp_badge(Image.fromarray(rgb), label)


def _save_frame(
    job: _RollRenderJob,
    img: Image.Image,
    name: str,
    focus: np.ndarray,
    fill_view: str,
) -> dict[str, Any]:
    out = job.output_dir / name
    img.save(out)
    _LOG.info("Wrote %s", out)
    return {
        "file": out.name,
        "width": img.width,
        "height": img.height,
        "hand_fill": _hand_fill_fraction(focus, view_rotation(fill_view), job.fill),
        "club_in_frame": True,
    }


def _render_strip_and_iso(
    job: _RollRenderJob,
    drawables: list[Any],
    focus: np.ndarray,
    roll: float,
    tag: str,
    strip_tag: str,
) -> list[dict[str, Any]]:
    """Palm/back/side strip plus an iso frame; returns their receipts."""
    bid = job.backend_id
    strip = Image.new("RGB", (job.panel_px * len(VIEWS), job.panel_px), (244, 246, 249))
    for idx, view in enumerate(VIEWS):
        panel = _render_view(
            job, drawables, focus, view, f"{bid} · {tag} · {view} ({roll:+.0f}°)"
        )
        strip.paste(panel, (idx * job.panel_px, 0))
    strip_receipt = _save_frame(
        job, strip, f"{bid}_{strip_tag}_roll_{roll:+.0f}.png", focus, "palm"
    )
    iso_img = _render_view(
        job, drawables, focus, "iso", f"{bid} · {tag} · iso ({roll:+.0f}°)"
    )
    iso_receipt = _save_frame(
        job, iso_img, f"{bid}_{tag}_iso_roll_{roll:+.0f}.png", focus, "iso"
    )
    return [strip_receipt, iso_receipt]


def render_roll_strip(output_dir: Path, backend_id: str = "procedural_skin") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = DEFAULT_HAND_VISUAL_SETTINGS
    job = _RollRenderJob(
        output_dir=output_dir,
        backend_id=backend_id,
        panel_px=max(1024, settings.render_frame_px),
        fill=settings.render_frame_fill,
    )
    club = club_context_drawables(DEFAULT_VISUAL_GRIP_RADIUS_M)
    receipts: list[dict[str, Any]] = []

    for roll in ROLLS:
        spec = _spec(roll)
        for role in ROLES:
            vis = build_for_comparison(
                backend_id,
                spec,
                role,
                grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                settings=settings,
            )
            mesh = vis.mesh
            color = skin_tone_rgb(settings.skin_material_for_role(role))
            drawables = [(mesh, color, 0.09), *club]
            receipts += _render_strip_and_iso(
                job, drawables, mesh.vertices, roll, role, role
            )

        both_verts: list[np.ndarray] = []
        both_drawables: list[tuple[Any, tuple[float, float, float], float]] = []
        for role in ROLES:
            vis = build_for_comparison(
                backend_id,
                spec,
                role,
                grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                settings=settings,
            )
            shared = mesh_in_shared_visual_frame(vis.mesh, spec, role)
            both_verts.append(shared.vertices)
            both_drawables.append(
                (shared, skin_tone_rgb(settings.skin_material_for_role(role)), 0.09)
            )
        both_drawables.extend(club)
        receipts += _render_strip_and_iso(
            job, both_drawables, np.vstack(both_verts), roll, "both", "both_hands"
        )

    receipt_path = output_dir / f"{backend_id}_render_stats.json"
    receipt_path.write_text(
        json.dumps(
            {
                "backend_id": backend_id,
                "panel_px": job.panel_px,
                "target_fill": job.fill,
                "frames": receipts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def render_size_skin_variants(
    output_dir: Path, backend_id: str = "procedural_skin"
) -> Path:
    """Contact sheet: hand size (0.88x / 1x / 1.15x) and skin tone per row.

    All panels share the large hand's framing so the size change stays visible.
    """
    base = _spec(0.0)
    club = club_context_drawables(DEFAULT_VISUAL_GRIP_RADIUS_M)
    rows: list[tuple[str, str, list[Any]]] = []
    for tag, scale, skin in SIZE_SKIN_VARIANTS:
        size = base.hand_size
        spec = replace(
            base,
            hand_size=replace(
                size, length_m=size.length_m * scale, breadth_m=size.breadth_m * scale
            ),
        )
        meshes = [
            build_for_comparison(
                backend_id,
                spec,
                role,
                grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                settings=DEFAULT_HAND_VISUAL_SETTINGS,
            ).mesh
            for role in ROLES
        ]
        rows.append((tag, skin, meshes))
    panel_px = 512
    sheet = Image.new("RGB", (panel_px * 2 * len(ROLES), panel_px * len(rows)))
    for r, (tag, skin, meshes) in enumerate(rows):
        for m, (role, mesh) in enumerate(zip(ROLES, meshes, strict=True)):
            for v, view in enumerate(("palm", "iso")):
                rgb = render_shaded_scene_rgb(
                    [(mesh, skin_tone_rgb(skin), 0.09), *club],
                    focus_vertices=rows[-1][2][m].vertices,
                    view_rot=view_rotation(view),
                    width_px=panel_px,
                    height_px=panel_px,
                    target_fill=0.6,
                )
                panel = _stamp_badge(
                    Image.fromarray(rgb), f"{tag} · {role} · {view} · {skin}"
                )
                sheet.paste(panel, ((2 * m + v) * panel_px, r * panel_px))
    out = output_dir / f"{backend_id}_size_skin_variants.png"
    sheet.save(out)
    _LOG.info("Wrote %s", out)
    return out


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
    render_size_skin_variants(args.output_dir, args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
