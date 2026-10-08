"""Annotated address foot diagrams for every engine from its own forward kinematics.

OSV-4 (#11730). For each engine the shared address seed (``hip_rotation_*``
solved by :func:`seed_document_feet`) is applied, the engine's own FK places the
femur, tibia, calcn and toes bodies, and overhead and face-on diagrams are
drawn with the angle of each foot annotated. These are FK diagrams, not engine
renders; the MuJoCo renders come from ``render_foot_progression_stills``.

    python3 -m scripts.render_foot_progression_engine_diagrams --out DIR \
        --left 16.4 --right 4.0 --label driver
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.motion_matching.foot_progression import (
    foot_role,
    model_long_axis,
    progression_angle_deg,
)
from src.shared.python.motion_matching.pipeline.address_feet import (
    MODEL_TARGET_AXIS,
    NATIVE_UP_AXIS,
    FootTargets,
    seed_document_feet,
)
from src.shared.python.motion_matching.pipeline.constants import (
    LEG_SEEDS,
    REPO_ROOT,
    square_forefoot_seeds,
)
from src.shared.python.motion_matching.pipeline.lane import document_seed
from src.shared.python.motion_matching.pipeline.plant import get_plant

SPEC = REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
BODIES = ("femur", "tibia", "calcn", "toes")
COLORS = {"left": "#1f77b4", "right": "#d62728"}


def plant_leg_points(engine: str, targets: FootTargets) -> dict[str, np.ndarray]:
    """Leg body origins (spec world: forward +x, left +y, up +z) for one engine."""
    document = json.loads(SPEC.read_text(encoding="utf-8"))
    plant = get_plant(engine, document)
    kin = plant.create_ik(square_forefoot_seeds(LEG_SEEDS))
    seeded = seed_document_feet(document, kin, targets, document_seed(document, kin))
    q = document_seed(seeded, kin)
    names = [f"{b}_{s}" for b in BODIES for s in "rl"]
    poses = plant.frame_poses({n: (n, (0.0, 0.0, 0.0)) for n in names}, q)
    return {n: np.asarray(poses[n][1], dtype=float) for n in names}


def opensim_leg_points(targets: FootTargets) -> dict[str, np.ndarray]:
    """OpenSim leg body origins mapped to the same display frame."""
    import opensim as osim

    from src.engines.physics_engines.opensim.python.tour_matching.address_feet import (
        apply_toe_out_seed,
    )

    path = (
        REPO_ROOT
        / "src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim"
    )
    q = {
        "pelvis_tilt": -0.15,
        "hip_flexion_r": 0.35,
        "knee_angle_r": 0.35,
        "ankle_angle_r": 0.1,
        "hip_flexion_l": 0.35,
        "knee_angle_l": 0.35,
        "ankle_angle_l": 0.1,
    }
    apply_toe_out_seed(path, q, dict(targets.target_deg))
    model = osim.Model(str(path))
    state = model.initSystem()
    for name, value in q.items():
        model.updCoordinateSet().get(name).setValue(state, value, False)
    model.realizePosition(state)
    out = {}
    for b in BODIES:
        for s in "rl":
            v = model.getBodySet().get(f"{b}_{s}").getPositionInGround(state)
            x, y, z = (v.get(i) for i in range(3))
            out[f"{b}_{s}"] = np.array([x, -z, y])  # forward, left, up
    return out


def foot_angles(
    points: dict[str, np.ndarray], targets: FootTargets
) -> dict[str, float]:
    """Toe-out per foot from calcn -> toes in the display frame."""
    out = {}
    for side, sfx in (("left", "l"), ("right", "r")):
        axis = model_long_axis(
            points[f"calcn_{sfx}"], points[f"toes_{sfx}"], NATIVE_UP_AXIS
        )
        out[side] = progression_angle_deg(
            axis,
            target_axis=MODEL_TARGET_AXIS,
            up=NATIVE_UP_AXIS,
            foot_role=foot_role(side, targets.handedness),
        )
    return out


def draw(
    engines: dict[str, dict[str, np.ndarray]],
    targets: FootTargets,
    label: str,
    out_dir: Path,
) -> list[Path]:
    """One PNG per engine: overhead (left) and face-on (right) leg diagrams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for engine, pts in engines.items():
        angles = foot_angles(pts, targets)
        fig, (top, front) = plt.subplots(1, 2, figsize=(11, 5.2))
        for side, sfx in (("left", "l"), ("right", "r")):
            chain = np.array([pts[f"{b}_{sfx}"] for b in BODIES])
            top.plot(chain[:, 0], chain[:, 1], "-o", color=COLORS[side], ms=3)
            front.plot(chain[:, 1], chain[:, 2], "-o", color=COLORS[side], ms=3)
            heel, toe = pts[f"calcn_{sfx}"], pts[f"toes_{sfx}"]
            axis = model_long_axis(heel, toe, NATIVE_UP_AXIS)
            top.plot(
                [heel[0], heel[0] + 0.35 * axis[0]],
                [heel[1], heel[1] + 0.35 * axis[1]],
                color="k",
                lw=2,
            )
            top.plot([heel[0], heel[0] + 0.35], [heel[1], heel[1]], "--", color="0.5")
            role = foot_role(side, targets.handedness)
            flag = " (default)" if targets.is_default[side] else ""
            top.annotate(
                f"{role} {side}: model {angles[side]:+.1f} deg\ncapture target {targets.target_deg[side]:+.1f}{flag}",
                (heel[0] + 0.05, heel[1] + (0.12 if side == "left" else -0.2)),
                color=COLORS[side],
                fontsize=9,
            )
        top.set_title(f"{engine}: overhead (golfer faces right, target up)")
        top.set_aspect("equal")
        top.set_xlabel("forward (m)")
        top.set_ylabel("toward target (m)")
        front.set_title(f"{engine}: face-on (target to image left)")
        front.set_aspect("equal")
        front.set_xlabel("toward target (m)")
        front.set_ylabel("up (m)")
        front.invert_xaxis()
        fig.suptitle(f"Address foot progression, {label}: FK of the shared seed")
        path = out_dir / f"{label}_{engine}_fk_diagram.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--left", type=float, required=True)
    parser.add_argument("--right", type=float, required=True)
    parser.add_argument("--label", default="driver")
    parser.add_argument("--engines", nargs="*", default=["mujoco", "drake", "opensim"])
    args = parser.parse_args(argv)
    targets = FootTargets(
        {"left": args.left, "right": args.right},
        {"left": False, "right": False},
        "capture",
    )
    engines: dict[str, dict[str, Any]] = {}
    for engine in args.engines:
        if engine == "opensim":
            engines[engine] = opensim_leg_points(targets)
        else:
            engines[engine] = plant_leg_points(engine, targets)
    for path in draw(engines, targets, args.label, args.out):
        print(path)  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
