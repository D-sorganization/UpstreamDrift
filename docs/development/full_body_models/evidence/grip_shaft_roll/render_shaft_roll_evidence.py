"""Render HRG-GB1b shaft-roll evidence strips (#15138).

Produces solid-mesh 3-panel strips (−15°, 0°, +15°) for lead and trail hands,
each in a butt-looking view (down the shaft) and a side view, with shaft axis
and a dorsum-direction marker.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.geometry import Mesh, concat_meshes
from src.shared.python.model_appearance.grip_hands import (
    assembly_meshes_with_hands,
    grip_hand_frame,
)
from src.shared.python.model_appearance.grip_pose import GripPose, Hand

SIGN_CONVENTION = "RHR about shaft +Y (butt → head); LH mirrors sign"
ROLLS_DEG = (-15.0, 0.0, 15.0)
SHAFT_AXIS = np.array([0.0, 1.0, 0.0])
_EVIDENCE_DIR = Path(__file__).resolve().parent


def _iron7_club() -> ClubAssembly:
    return ClubAssembly("iron7", 0.95, 0.25, 0.0065)


def _pose_for_roll(hand: Hand, roll_deg: float) -> GripPose:
    spec = GripSpec.legacy()
    if hand is Hand.LEAD:
        spec = replace(spec, lead=HandGripSpec(shaft_roll_deg=roll_deg))
    else:
        spec = replace(spec, trail=HandGripSpec(shaft_roll_deg=roll_deg))
    return spec.to_grip_pose()


def _dorsum_marker_points(
    club: ClubAssembly, hand: Hand, pose: GripPose
) -> tuple[np.ndarray, np.ndarray]:
    """Base point on shaft through hand centre and dorsum arrow tip (club frame)."""
    frame = grip_hand_frame(club, hand, pose)
    centre = frame[:3, 3]
    dorsal = frame[:3, 0] - SHAFT_AXIS * float(np.dot(frame[:3, 0], SHAFT_AXIS))
    norm = float(np.linalg.norm(dorsal))
    if norm < 1e-9:
        dorsal = frame[:3, 1]
        norm = float(np.linalg.norm(dorsal))
    dorsal = dorsal / norm
    span = 0.045
    return centre, centre + span * dorsal


def _shaft_axis_segment(club: ClubAssembly) -> tuple[np.ndarray, np.ndarray]:
    z = club.axis_offset_m
    y0 = -(club.length_m - club.grip_length_m)
    y1 = -club.length_m + 0.05
    p0 = np.array([0.0, y0, z])
    p1 = np.array([0.0, y1, z])
    return p0, p1


def _club_scene(club: ClubAssembly, pose: GripPose) -> dict[str, Mesh]:
    return assembly_meshes_with_hands(club, pose)


def _scene_mesh(parts: dict[str, Mesh]) -> Mesh:
    keys = ("shaft", "grip", "head", "lead_hand", "trail_hand")
    present = [parts[k] for k in keys if k in parts]
    return concat_meshes(*present)


def _display_coords(points: np.ndarray, view: Literal["butt", "side"]) -> np.ndarray:
    """Map club frame XYZ to matplotlib XYZ (butt view puts shaft on display Z)."""
    if view == "side":
        return np.asarray(points, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        return np.array([pts[0], pts[2], pts[1]])
    return np.column_stack([pts[:, 0], pts[:, 2], pts[:, 1]])


def _plot_scene(
    ax,
    mesh: Mesh,
    club: ClubAssembly,
    hand: Hand,
    pose: GripPose,
    *,
    view: Literal["butt", "side"],
    roll_deg: float,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plot_v = _display_coords(mesh.vertices, view)
    elev, azim = (90.0, 0.0) if view == "butt" else (12.0, -58.0)
    x, y, z = plot_v[:, 0], plot_v[:, 1], plot_v[:, 2]

    tri = plot_v[mesh.faces]
    z_span = float(z.max() - z.min())
    colors = np.clip(
        0.55 + 0.35 * (tri.mean(axis=1)[:, 2] - z.min()) / max(z_span, 1e-6), 0, 1
    )
    face_rgba = np.zeros((len(tri), 4))
    face_rgba[:, :3] = np.stack(
        [0.25 + 0.5 * colors, 0.35 + 0.3 * colors, 0.5 + 0.4 * colors], axis=1
    )
    face_rgba[:, 3] = 0.92
    ax.add_collection3d(
        Poly3DCollection(
            tri,
            facecolors=face_rgba,
            edgecolors=(0.1, 0.1, 0.15, 0.25),
            linewidths=0.15,
        )
    )

    p0, p1 = _shaft_axis_segment(club)
    shaft_pts = _display_coords(np.vstack([p0, p1]), view)
    ax.plot(
        shaft_pts[:, 0],
        shaft_pts[:, 1],
        shaft_pts[:, 2],
        color="crimson",
        linewidth=2.5,
        label="shaft axis",
    )

    base, tip = _dorsum_marker_points(club, hand, pose)
    b = _display_coords(base, view)
    t = _display_coords(tip, view)
    ax.plot(
        [b[0], t[0]],
        [b[1], t[1]],
        [b[2], t[2]],
        color="gold",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax.quiver(
        b[0],
        b[1],
        b[2],
        t[0] - b[0],
        t[1] - b[1],
        t[2] - b[2],
        color="orange",
        linewidth=1.5,
        arrow_length_ratio=0.25,
    )

    pad = 0.06
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_zlim(z.min() - pad, z.max() + pad)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(
        f"{hand.value}  shaft_roll={roll_deg:+.0f}°\n{SIGN_CONVENTION}",
        fontsize=9,
    )


def _render_strip(
    club: ClubAssembly,
    hand: Hand,
    view: Literal["butt", "side"],
    out_path: Path,
) -> list[dict[str, object]]:
    from matplotlib import pyplot as plt  # noqa: PLC0415

    fig = plt.figure(figsize=(13.5, 4.8), dpi=140)
    entries: list[dict[str, object]] = []
    for col, roll in enumerate(ROLLS_DEG, start=1):
        pose = _pose_for_roll(hand, roll)
        mesh = _scene_mesh(_club_scene(club, pose))
        ax = fig.add_subplot(1, 3, col, projection="3d")
        _plot_scene(ax, mesh, club, hand, pose, view=view, roll_deg=roll)
        if col == 1:
            ax.set_ylabel(
                "Butt → head (down shaft axis)" if view == "butt" else "Side view",
                fontsize=10,
            )
        entries.append(
            {
                "hand": hand.value,
                "view": view,
                "shaft_roll_deg": roll,
                "sign_convention": SIGN_CONVENTION,
            }
        )
    fig.suptitle(
        f"HRG-GB1b shaft roll evidence — {hand.value} hand — {view} view",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return entries


def render_all(evidence_dir: Path | None = None) -> dict[str, object]:
    """Write four strip PNGs and return receipt payload."""
    out_dir = _EVIDENCE_DIR if evidence_dir is None else evidence_dir
    club = _iron7_club()
    receipts: dict[str, object] = {
        "sign_convention": SIGN_CONVENTION,
        "rolls_deg": list(ROLLS_DEG),
        "club": {"alias": club.head_alias, "length_m": club.length_m},
        "artifacts": {},
    }
    for hand in (Hand.LEAD, Hand.TRAIL):
        for view in ("butt", "side"):
            name = f"{hand.value}_{view}_shaft_roll_strip.png"
            path = out_dir / name
            meta = _render_strip(club, hand, view, path)
            receipts["artifacts"][name] = {
                "path": str(path.relative_to(out_dir.parents[2])),
                "panels": meta,
            }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipts["receipt_path"] = str(receipt_path)
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_EVIDENCE_DIR,
        help="Directory for PNG strips and receipt.json",
    )
    args = parser.parse_args()
    render_all(args.out_dir)


if __name__ == "__main__":
    main()
