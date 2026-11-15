"""Backend contact sheet: 2 columns, roll 0°, 1024px legible layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.grip_spec.shaft_roll import resolved_shaft_roll_deg
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
    hand_mesh_with_shaft_roll,
)
from src.shared.python.model_appearance.hand_backends.metrics import (
    anthropometric_ratios,
    anthropometric_vs_reference,
    grip_shaft_penetration_fraction,
    inter_finger_penetration_fraction,
)
from src.shared.python.model_appearance.hand_backends.preview import (
    add_grip_shaft_solid,
    frame_hand_viewport,
    plot_shaded_mesh,
    skin_tone_rgb,
    transform_mesh,
    view_rotation,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual,
    registered_backend_ids,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    DEFAULT_HAND_VISUAL_SETTINGS,
    HandVisualSettings,
)
from src.shared.python.model_appearance.hand_backends.skeleton_extract import (
    finger_polyline,
    middle_finger_length,
)
from src.shared.python.model_appearance.geometry import concat_meshes
from src.shared.python.model_appearance.hand_backends.placement import (
    mesh_in_shared_visual_frame,
)

RENDER_DPI = 128
VIEWS = ("palm", "back", "side")
ROW_SPECS: tuple[tuple[str, str | None, str], ...] = (
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
    role: HandRole,
    angles: HandJointAngles,
    settings: HandVisualSettings,
) -> dict[str, Any]:
    vis = build_hand_visual(
        backend_id,
        grip_spec,
        role,
        angles,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    rolled = hand_mesh_with_shaft_roll(vis.mesh, grip_spec, role)
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
    mid_len = None
    if backend_id == "procedural_skin":
        mid_len = middle_finger_length(finger_polyline(reference, "middle"))
    anthro = anthropometric_vs_reference(
        rolled,
        reference=settings.anthropometry,
        middle_finger_length_m=mid_len,
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
        "palm_arch_angle_deg": vis.palm_arch_angle_deg,
        **anthropometric_ratios(rolled),
        **anthro,
    }


def render_contact_sheet(
    output_png: Path,
    output_json: Path,
    *,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backends = registered_backend_ids()
    if len(backends) > 2:
        backends = backends[:2]
    spec = GripSpec.legacy()
    n_rows = len(ROW_SPECS)
    n_cols = len(backends)
    fig = plt.figure(
        figsize=(6.0 * n_cols, 3.0 * n_rows),
        facecolor="white",
    )
    metrics: list[dict[str, Any]] = []
    cell = 0
    for row_label, role, view in ROW_SPECS:
        rot = view_rotation(view)
        for backend_id in backends:
            cell += 1
            ax = fig.add_subplot(n_rows, n_cols, cell, projection="3d")
            if role is not None:
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
                metrics.append(
                    _metrics_for_backend(backend_id, spec, role, angles, settings)
                )
                color = skin_tone_rgb(settings.skin_material_for_role(role))
                posed = transform_mesh(mesh, rot)
                plot_shaded_mesh(ax, posed, color=color, frame_fill=0.72)
            else:
                meshes = []
                colors = []
                for r in ("lead", "trail"):
                    angles = HandJointAngles.from_grip_spec(spec, r)
                    vis = build_hand_visual(
                        backend_id,
                        spec,
                        r,
                        angles,
                        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
                        settings=settings,
                    )
                    mesh = hand_mesh_with_shaft_roll(vis.mesh, spec, r)
                    shared = mesh_in_shared_visual_frame(mesh, spec, r)
                    meshes.append(transform_mesh(shared, rot))
                    colors.append(skin_tone_rgb(settings.skin_material_for_role(r)))
                for m, col in zip(meshes, colors, strict=True):
                    plot_shaded_mesh(
                        ax, m, color=col, auto_frame=False, frame_fill=0.72
                    )
                combined = concat_meshes(*meshes)
                frame_hand_viewport(ax, combined.vertices, fill_fraction=0.72)
            z0, z1 = (
                float(ax.get_zlim3d()[0]),
                float(ax.get_zlim3d()[1]),
            )
            add_grip_shaft_solid(
                ax,
                DEFAULT_VISUAL_GRIP_RADIUS_M,
                z_range=(z0, z1),
                rotation=rot,
            )
            ax.set_title(
                f"{backend_id}\n{row_label} · {view}",
                fontsize=11,
                linespacing=1.2,
            )
            ax.set_axis_off()
    fig.suptitle(
        "Hand backends @ shaft roll 0°  (capsule | procedural skin)",
        fontsize=14,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.96, bottom=0.02, hspace=0.08, wspace=0.05
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=RENDER_DPI, facecolor="white")
    plt.close(fig)
    output_json.write_text(
        json.dumps({"metrics": metrics, "shaft_roll_deg": 0.0}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
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
    print(f"Wrote {args.png} and {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
