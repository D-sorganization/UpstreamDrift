"""Render the shared poser's standard views for the visual self-review.

The pictures are kinematic capsules, not skin. The script does not score
them. Open the PNGs and use the checklist in
``docs/development/full_body_models/hand_handoff/shared_posing.md``.

Examples
--------
python3 scripts/render_shared_hand_pose.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.shared.python.grip_spec.types import GripSpec  # noqa: E402
from src.shared.python.model_appearance.hand_backends import (  # noqa: E402
    default_grip,
    pose_grip,
)

_VIEWS = ("face", "side", "oblique")
_ROLLS = (0.0, 20.0, -20.0)
_LEAD = "#e0a888"
_TRAIL = "#c4845a"
_GRIP = "#7d8490"
_FRAME = 0.76
_PIXELS = 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Render shared hand-pose views")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/development/full_body_models/hand_handoff/figures"),
    )
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    grip = default_grip()
    right = {
        roll: pose_grip(GripSpec.legacy(), grip, shaft_roll_deg=roll) for roll in _ROLLS
    }
    left = pose_grip(GripSpec.legacy(handedness="left"), grip, shaft_roll_deg=0.0)
    written: list[Path] = []
    for roll, pose in right.items():
        for view in _VIEWS:
            for which in ("lead", "trail", "both"):
                path = out / f"{which}_{view}_roll_{_tag(roll)}.png"
                _render(
                    path,
                    _hands(pose, which),
                    view,
                    grip.radius_m,
                    _title(which, view, roll),
                )
                written.append(path)
    mirror = out / "both_oblique_left_roll_0.png"
    _render(
        mirror,
        _hands(left, "both"),
        "oblique",
        grip.radius_m,
        "Both Hands, Oblique View, Left-Handed, Shaft Roll 0",
    )
    written.append(mirror)
    _sheet(out / "contact_sheet.png", written)
    print(f"wrote {len(written)} views under {out}")  # noqa: T201 - CLI inventory


def _hands(pose, which: str):
    if which == "lead":
        return [(pose.lead, _LEAD)]
    if which == "trail":
        return [(pose.trail, _TRAIL)]
    return [(pose.lead, _LEAD), (pose.trail, _TRAIL)]


def _render(path: Path, hands, view: str, grip_radius: float, title: str) -> None:
    projected = []
    raw_z: list[float] = []
    for pose, color in hands:
        for segment in pose.segments:
            projected.append(
                (_project(segment.p0, view), _project(segment.p1, view), color)
            )
            raw_z.extend((segment.p0[2], segment.p1[2]))
    points = [item[0] for item in projected] + [item[1] for item in projected]
    (x0, x1), (y0, y1) = _frame(points)
    figure, axes = plt.subplots(figsize=(_PIXELS / 128, _PIXELS / 128), dpi=128)
    axes.set_facecolor("#f6f1ea")
    _draw_grip(axes, view, grip_radius, raw_z)
    for start, end, color in projected:
        axes.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=7,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )
    axes.set_xlim(x0, x1)
    axes.set_ylim(y0, y1)
    axes.set_aspect("equal")
    axes.set_title(title)
    axes.set_xticks([])
    axes.set_yticks([])
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _draw_grip(axes, view: str, radius: float, raw_z: list[float]) -> None:
    z0, z1 = min(raw_z), max(raw_z)
    pad = 0.2 * (z1 - z0)
    zs = [z0 - pad, z1 + pad]
    axis = [_project((0.0, 0.0, z), view) for z in zs]
    axes.plot(
        [point[0] for point in axis],
        [point[1] for point in axis],
        color=_GRIP,
        linewidth=5,
        solid_capstyle="butt",
        zorder=1,
    )
    for sign in (-1.0, 1.0):
        edge = [_project((sign * radius, 0.0, z), view) for z in zs]
        axes.plot(
            [point[0] for point in edge],
            [point[1] for point in edge],
            color="#c5cad3",
            linewidth=1,
            zorder=1,
        )


def _frame(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-4)
    frame = span / _FRAME
    cx = 0.5 * (max(xs) + min(xs))
    cy = 0.5 * (max(ys) + min(ys))
    half = 0.5 * frame
    return (cx - half, cx + half), (cy - half, cy + half)


def _project(point: tuple[float, float, float], view: str) -> tuple[float, float]:
    if view == "face":
        return (point[0], point[2])
    if view == "side":
        return (-point[1], point[2])
    right, up = _oblique_basis()
    return (_dot(point, right), _dot(point, up))


def _oblique_basis() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    camera = _unit((1.0, -1.0, 0.35))
    forward = _scale(camera, -1.0)
    right = _unit(_cross(forward, (0.0, 0.0, 1.0)))
    up = _unit(_cross(right, forward))
    return right, up


def _sheet(path: Path, images: list[Path]) -> None:
    chosen = [
        image
        for image in images
        if image.name
        in {
            "both_face_roll_0.png",
            "both_side_roll_0.png",
            "both_oblique_roll_0.png",
            "both_oblique_roll_p20.png",
            "both_oblique_roll_m20.png",
            "both_oblique_left_roll_0.png",
            "lead_oblique_roll_0.png",
            "trail_oblique_roll_0.png",
        }
    ]
    if not chosen:
        return
    columns = 4
    rows = math.ceil(len(chosen) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 3.2, rows * 3.2), dpi=120
    )
    flat = list(axes.ravel()) if rows > 1 or columns > 1 else [axes]
    for axis, image in zip(flat, chosen, strict=False):
        axis.imshow(plt.imread(image))
        axis.set_title(image.stem.replace("_", " "), fontsize=8)
        axis.set_axis_off()
    for axis in flat[len(chosen) :]:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _title(which: str, view: str, roll: float) -> str:
    hand = {"lead": "Lead Hand", "trail": "Trail Hand", "both": "Both Hands"}[which]
    name = {"face": "Face View", "side": "Side View", "oblique": "Oblique View"}[view]
    return f"{hand}, {name}, Shaft Roll {roll:.0f}"


def _tag(roll: float) -> str:
    if roll > 0.0:
        return f"p{roll:.0f}"
    if roll < 0.0:
        return f"m{abs(roll):.0f}"
    return "0"


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.dist(vector, (0.0, 0.0, 0.0))
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _scale(
    vector: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(
    point: tuple[float, float, float], vector: tuple[float, float, float]
) -> float:
    return point[0] * vector[0] + point[1] * vector[1] + point[2] * vector[2]


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


if __name__ == "__main__":
    main()
