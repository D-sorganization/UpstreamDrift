"""Backend contact sheet: all registered backends, roll 0°, smooth 2x2 SSAA."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.shared.python.grip_spec import GripSpec
from src.shared.python.grip_spec.shaft_roll import resolved_shaft_roll_deg
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.types import HandSide
from src.shared.python.model_appearance.hand_backends.metrics import (
    anthropometric_ratios,
    anthropometric_vs_reference,
    grip_shaft_penetration_fraction,
    inter_finger_penetration_fraction,
    palm_arch_angle_deg,
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
from src.shared.python.model_appearance.hand_backends.procedural_skin import (
    measure_procedural_hand_anthropometry,
)
from src.shared.python.model_appearance.hand_backends.comparison import (
    build_for_comparison,
    comparison_backend_names,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    HandVisualSettings,
)

_LOG = logging.getLogger(__name__)
CELL_PX = 512
HEADER_PX = 34
BANNER_PX = 48
VIEWS = ("palm", "back", "side")
ROW_SPECS: tuple[tuple[str, HandSide | None, str], ...] = (
    ("lead", "lead", "palm"),
    ("lead", "lead", "back"),
    ("lead", "lead", "side"),
    ("trail", "trail", "palm"),
    ("trail", "trail", "back"),
    ("trail", "trail", "side"),
    ("both", None, "palm"),
    ("both", None, "back"),
    ("both", None, "side"),
)


def _metrics_for_backend(
    backend_id: str,
    grip_spec: GripSpec,
    role: HandSide,
    settings: HandVisualSettings,
) -> dict[str, Any]:
    vis = build_for_comparison(
        backend_id,
        grip_spec,
        role,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    rolled = vis.mesh
    hand = Hand.LEAD if role == "lead" else Hand.TRAIL
    reference = build_grip_hand_mesh(
        hand,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        pose=grip_spec.to_grip_pose(),
        params=HandMeshParams(scale=hand_mesh_scale(grip_spec.hand_size.length_m)),
    )
    finger_clouds = [
        (part.finger, part.mesh.vertices)
        for part in reference.components
        if part.finger in FINGER_NAMES and part.name.endswith("_proximal")
    ]
    hand_len_m: float | None = None
    palm_wid_m: float | None = None
    mid_len_m: float | None = None
    extra: dict[str, Any] = {}
    if backend_id == "procedural_skin":
        proc = measure_procedural_hand_anthropometry(
            grip_spec,
            role,
            grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
            settings=settings,
        )
        hand_len_m = float(proc["hand_length_m"])
        palm_wid_m = float(proc["palm_width_m"])
        mid_len_m = float(proc["middle_finger_length_m"])
        extra = {
            "palm_length_m": proc["palm_length_m"],
            "cup_depth_m": proc["cup_depth_m"],
            "max_knuckle_gap_m": proc["max_knuckle_gap_m"],
            "min_inter_finger_gap_m": proc["min_inter_finger_gap_m"],
            "pad_clearances_m": proc["pad_clearances_m"],
        }
    anthro = anthropometric_vs_reference(
        rolled,
        reference=settings.anthropometry,
        hand_length_m=hand_len_m,
        palm_width_m=palm_wid_m,
        middle_finger_length_m=mid_len_m,
    )
    return {
        "backend_id": backend_id,
        "role": role,
        "shaft_roll_deg": resolved_shaft_roll_deg(grip_spec, role),
        "shaft_penetration_fraction": grip_shaft_penetration_fraction(
            rolled,
            grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
            tolerance_m=settings.shaft_penetration_tolerance_m,
        ),
        "finger_penetration_fraction": inter_finger_penetration_fraction(
            finger_clouds,
            tolerance_m=settings.finger_penetration_tolerance_m,
        ),
        "palm_arch_angle_deg": palm_arch_angle_deg(rolled),
        **anthropometric_ratios(rolled),
        **anthro,
        **extra,
    }


def render_contact_sheet(
    output_png: Path,
    output_json: Path,
    *,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> None:
    backends = comparison_backend_names()
    spec = GripSpec.legacy()
    n_rows = len(ROW_SPECS)
    n_cols = len(backends)
    club = club_context_drawables(DEFAULT_VISUAL_GRIP_RADIUS_M)

    cell_h = HEADER_PX + CELL_PX
    canvas = Image.new(
        "RGB", (n_cols * CELL_PX, BANNER_PX + n_rows * cell_h), (242, 244, 247)
    )
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.load_default(size=22)
    font_cell = ImageFont.load_default(size=18)
    title_text = "Hand backends @ shaft roll 0° (" + " | ".join(backends) + ")"
    draw.rectangle([0, 0, canvas.width, BANNER_PX], fill=(28, 34, 46))
    draw.text((16, 12), title_text, fill=(245, 247, 250), font=font_title)

    metrics: list[dict[str, Any]] = []
    for row_idx, (row_label, role, view) in enumerate(ROW_SPECS):
        rot = view_rotation(view)
        y0 = BANNER_PX + row_idx * cell_h
        for col_idx, backend_id in enumerate(backends):
            x0 = col_idx * CELL_PX
            if role is not None:
                vis = build_for_comparison(
                    backend_id,
                    spec,
                    role,
                    grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                    settings=settings,
                )
                mesh = vis.mesh
                metrics.append(_metrics_for_backend(backend_id, spec, role, settings))
                color = skin_tone_rgb(settings.skin_material_for_role(role))
                rgb = render_shaded_scene_rgb(
                    [(mesh, color, 0.09), *club],
                    focus_vertices=mesh.vertices,
                    view_rot=rot,
                    width_px=CELL_PX,
                    height_px=CELL_PX,
                    target_fill=settings.render_frame_fill,
                )
            else:
                meshes = []
                drawables: list[tuple[Any, tuple[float, float, float], float]] = []
                for r in ("lead", "trail"):
                    vis = build_for_comparison(
                        backend_id,
                        spec,
                        r,
                        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                        settings=settings,
                    )
                    mesh = vis.mesh
                    shared = mesh_in_shared_visual_frame(mesh, spec, r)
                    meshes.append(shared.vertices)
                    drawables.append(
                        (
                            shared,
                            skin_tone_rgb(settings.skin_material_for_role(r)),
                            0.09,
                        )
                    )
                drawables.extend(club)
                rgb = render_shaded_scene_rgb(
                    drawables,
                    focus_vertices=np.vstack(meshes),
                    view_rot=rot,
                    width_px=CELL_PX,
                    height_px=CELL_PX,
                    target_fill=settings.render_frame_fill,
                )
            draw.rectangle(
                [x0, y0, x0 + CELL_PX - 1, y0 + HEADER_PX], fill=(228, 232, 238)
            )
            draw.text(
                (x0 + 12, y0 + 7),
                f"{backend_id}  |  {row_label} · {view}",
                fill=(24, 28, 36),
                font=font_cell,
            )
            canvas.paste(Image.fromarray(rgb), (x0, y0 + HEADER_PX))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)
    output_json.write_text(
        json.dumps({"metrics": metrics, "shaft_roll_deg": 0.0}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--png",
        type=Path,
        default=Path(
            "docs/development/full_body_models/evidence/hand_backends/contact_sheet.png"
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(
            "docs/development/full_body_models/evidence/hand_backends/metrics.json"
        ),
    )
    args = parser.parse_args(argv)
    render_contact_sheet(args.png, args.json)
    _LOG.info("Wrote %s and %s", args.png, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
