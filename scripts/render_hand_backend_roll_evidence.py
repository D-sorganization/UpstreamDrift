"""Lead/trail single-hand views then dual-hand strips (1024px, dual light)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.geometry import concat_meshes
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    hand_mesh_with_shaft_roll,
)
from src.shared.python.model_appearance.hand_backends.preview import (
    add_grip_shaft_solid,
    frame_hand_viewport,
    grip_shaft_mesh,
    plot_shaded_mesh,
    skin_tone_rgb,
    transform_mesh,
    view_rotation,
)
from src.shared.python.model_appearance.hand_backends.placement import (
    mesh_in_shared_visual_frame,
)
from src.shared.python.model_appearance.hand_backends.registry import build_hand_visual
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
)

ROLLS = (-20.0, 0.0, 20.0)
VIEWS = ("palm", "back", "side")
ROLES = ("lead", "trail")
RENDER_DPI = 128
FIG_INCH = 8.0  # 8 * 128 ~= 1024 px


def _spec(roll_deg: float) -> GripSpec:
    return GripSpec.legacy(
        lead=HandGripSpec(shaft_roll_deg=roll_deg),
        trail=HandGripSpec(shaft_roll_deg=-0.5 * roll_deg),
    )


def _render_panel(fig, slot, mesh, grip_r, view, color, rotation):
    ax = fig.add_subplot(1, 3, slot, projection="3d")
    posed = transform_mesh(mesh, rotation)
    z0, z1 = float(posed.vertices[:, 2].min()), float(posed.vertices[:, 2].max())
    margin = 0.04
    z_range = (z0 - margin, z1 + margin)
    grip = transform_mesh(grip_shaft_mesh(grip_r, z_range=z_range), rotation)
    plot_shaded_mesh(ax, grip, color=(0.22, 0.24, 0.28), auto_frame=False, alpha=0.9)
    plot_shaded_mesh(ax, posed, color=color, auto_frame=False)
    combined = np.vstack([posed.vertices, grip.vertices])
    frame_hand_viewport(ax, combined, fill_fraction=0.70)
    ax.set_title(view, fontsize=12, fontweight="medium")
    ax.set_axis_off()
    return ax


def render_roll_strip(output_dir: Path, backend_id: str = "procedural_skin") -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    settings = DEFAULT_HAND_VISUAL_SETTINGS
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
            fig = plt.figure(figsize=(FIG_INCH * 3, FIG_INCH), facecolor="white")
            for idx, view in enumerate(VIEWS, start=1):
                rot = view_rotation(view)
                _render_panel(
                    fig, idx, mesh, DEFAULT_VISUAL_GRIP_RADIUS_M, view, color, rot
                )
            fig.suptitle(
                f"{backend_id} — {role} hand  roll {roll:+.0f}°",
                fontsize=13,
            )
            fig.tight_layout()
            out = output_dir / f"{backend_id}_{role}_roll_{roll:+.0f}.png"
            fig.savefig(out, dpi=RENDER_DPI, facecolor="white")
            plt.close(fig)
            print(out)

        fig = plt.figure(figsize=(FIG_INCH * 3, FIG_INCH), facecolor="white")
        for idx, view in enumerate(VIEWS, start=1):
            rot = view_rotation(view)
            meshes = []
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
                meshes.append(transform_mesh(shared, rot))
            combined = concat_meshes(*meshes)
            color_lead = skin_tone_rgb(settings.lead_skin_material)
            color_trail = skin_tone_rgb(settings.trail_skin_material)
            ax = fig.add_subplot(1, 3, idx, projection="3d")
            for m, col in zip(meshes, (color_lead, color_trail), strict=True):
                plot_shaded_mesh(
                    ax, m, color=col, frame_fill=0.70, alpha=0.98, auto_frame=False
                )
            from src.shared.python.model_appearance.hand_backends.preview import (
                frame_hand_viewport,
            )

            frame_hand_viewport(ax, combined.vertices, fill_fraction=0.70)
            z0, z1 = float(ax.get_zlim3d()[0]), float(ax.get_zlim3d()[1])
            margin = 0.04
            grip = transform_mesh(
                grip_shaft_mesh(
                    DEFAULT_VISUAL_GRIP_RADIUS_M,
                    z_range=(z0 - margin, z1 + margin),
                ),
                rot,
            )
            plot_shaded_mesh(
                ax, grip, color=(0.22, 0.24, 0.28), auto_frame=False, alpha=0.9
            )
            ax.set_title(f"{view} (lead + trail)", fontsize=12)
            ax.set_axis_off()
        out = output_dir / f"{backend_id}_both_hands_roll_{roll:+.0f}.png"
        fig.suptitle(f"{backend_id} — both hands  roll {roll:+.0f}°", fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=RENDER_DPI, facecolor="white")
        plt.close(fig)
        print(out)


def main(argv: list[str] | None = None) -> int:
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
