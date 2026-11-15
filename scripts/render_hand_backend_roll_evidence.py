"""Per-roll dual-hand grip evidence (±20°, 0°) for PR acceptance."""

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
    add_grip_cylinder_wire,
    plot_shaded_mesh,
    skin_tone_rgb,
)
from src.shared.python.model_appearance.hand_backends.registry import build_hand_visual
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
)

ROLLS = (-20.0, 0.0, 20.0)
VIEWS = (
    ("front", np.eye(3)),
    (
        "side",
        np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=float,
        ),
    ),
    (
        "grip",
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
    ),
)


def _spec(roll_deg: float) -> GripSpec:
    return GripSpec.legacy(
        lead=HandGripSpec(shaft_roll_deg=roll_deg),
        trail=HandGripSpec(shaft_roll_deg=-0.5 * roll_deg),
    )


def render_roll_strip(output_dir: Path, backend_id: str = "procedural_skin") -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    settings = DEFAULT_HAND_VISUAL_SETTINGS
    color = skin_tone_rgb(settings.skin_material)
    for roll in ROLLS:
        spec = _spec(roll)
        fig = plt.figure(figsize=(12, 3.6))
        for idx, (view_name, frame) in enumerate(VIEWS, start=1):
            ax = fig.add_subplot(1, 3, idx, projection="3d")
            meshes = []
            for role in ("lead", "trail"):
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
                from src.shared.python.model_appearance.geometry import Mesh

                meshes.append(
                    Mesh(rolled.vertices @ frame, rolled.faces),
                )
            combined = concat_meshes(*meshes)
            plot_shaded_mesh(ax, combined, color=color)
            z0, z1 = (
                float(combined.vertices[:, 2].min()),
                float(combined.vertices[:, 2].max()),
            )
            add_grip_cylinder_wire(ax, DEFAULT_VISUAL_GRIP_RADIUS_M, z_range=(z0, z1))
            ax.set_title(f"{view_name} roll {roll:+.0f}°")
            ax.set_axis_off()
        out = output_dir / f"{backend_id}_both_hands_roll_{roll:+.0f}.png"
        fig.suptitle(f"{backend_id} — lead+trail")
        fig.tight_layout()
        fig.savefig(out, dpi=160, facecolor="white")
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
