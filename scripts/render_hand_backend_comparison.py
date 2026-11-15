"""Side-by-side hand backend comparison contact sheet (epic #15122)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    build_grip_hand_mesh,
)
from src.shared.python.grip_spec.shaft_roll import resolved_shaft_roll_deg
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_hand_mesh import HandMeshParams
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
    hand_mesh_with_shaft_roll,
)
from src.shared.python.model_appearance.hand_backends.metrics import (
    anthropometric_ratios,
    grip_shaft_penetration_fraction,
    inter_finger_penetration_fraction,
)
from src.shared.python.model_appearance.hand_backends.preview import (
    add_grip_cylinder_wire,
    plot_shaded_mesh,
    skin_tone_rgb,
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

DEFAULT_ROLLS_DEG = (-20.0, 0.0, 20.0)
_VIEW_AXES = (
    ("front", (0, -1, 0)),
    ("side", (1, 0, 0)),
    ("grip", (0, 0, 1)),
)


def _spec_for_roll(roll_deg: float) -> GripSpec:
    return GripSpec.legacy(
        lead=HandGripSpec(shaft_roll_deg=roll_deg),
        trail=HandGripSpec(shaft_roll_deg=-roll_deg * 0.5),
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
    }


def _rotate_view(mesh, view: tuple[float, float, float]):
    from src.shared.python.model_appearance.geometry import Mesh

    direction = np.asarray(view, dtype=float)
    direction /= np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    frame = np.column_stack([right, up, direction])
    return Mesh(mesh.vertices @ frame, mesh.faces)


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
    roles: tuple[HandRole, ...] = ("lead", "trail")
    metrics: list[dict[str, Any]] = []
    n_rows = len(DEFAULT_ROLLS_DEG) * len(roles) * len(_VIEW_AXES)
    n_cols = len(backends)
    fig = plt.figure(figsize=(3.2 * n_cols, 2.4 * n_rows))
    color = skin_tone_rgb(settings.skin_material)
    cell = 0
    for roll in DEFAULT_ROLLS_DEG:
        spec = _spec_for_roll(roll)
        for role in roles:
            angles = HandJointAngles.from_grip_spec(spec, role)
            for view_name, view_axis in _VIEW_AXES:
                for backend_id in backends:
                    cell += 1
                    ax = fig.add_subplot(n_rows, n_cols, cell, projection="3d")
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
                    viewed = _rotate_view(mesh, view_axis)
                    plot_shaded_mesh(ax, viewed, color=color)
                    z0, z1 = (
                        float(mesh.vertices[:, 2].min()),
                        float(mesh.vertices[:, 2].max()),
                    )
                    add_grip_cylinder_wire(
                        ax, DEFAULT_VISUAL_GRIP_RADIUS_M, z_range=(z0, z1)
                    )
                    ax.set_title(
                        f"{backend_id} {role} {view_name} roll={roll:+.0f}°",
                        fontsize=7,
                    )
                    ax.set_axis_off()
    fig.suptitle(
        "Hand visual backends — GripSpec.legacy() iron7 grip",
        fontsize=11,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, facecolor="white")
    plt.close(fig)
    output_json.write_text(
        json.dumps({"metrics": metrics, "rolls_deg": list(DEFAULT_ROLLS_DEG)}, indent=2)
        + "\n",
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
