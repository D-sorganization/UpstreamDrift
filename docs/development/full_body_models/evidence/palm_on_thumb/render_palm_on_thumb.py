"""Stills and sections of the trail palm on the lead thumb (#15265).

Usage, from the repository root, headless (the shared renderer of #15154 is
read from ``--renderer-root`` until it is on this branch)::

    MUJOCO_GL=egl PYTHONPATH=. python3 \\
        docs/development/full_body_models/evidence/palm_on_thumb/render_palm_on_thumb.py \\
        --renderer-root <checkout with hand_backends/render.py>

Writes ``palm_on_thumb_stills.png`` and ``palm_on_thumb_sections.png`` beside
this file. In both, the top row is the legacy trail cross angle, the middle
row the owner's (``owner_trail_cross_angle_deg``), which is the default, and
the bottom row a trial: the owner's with the palm pads folded round the thumb
(``pair.palm_wrap`` set to ``thumb_wrap_trial_pad``; not the default, the skin
moves too far from its bones). Stills: the golfer's view,
then three views round the trail side. Sections: across the grip at stations
toward the head, then along the grip through the lead thumb (lead red, trail
blue, grip black). Gaps and wrap shares along the thumb go to standard output.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.shared.python.grip_spec.types import GripSpec  # noqa: E402
from src.shared.python.model_appearance import hand_backends  # noqa: E402
from src.shared.python.model_appearance.hand_backends.golf_grip import (  # noqa: E402
    load_golf_grip,
)
from src.shared.python.model_appearance.hand_backends.qa import (  # noqa: E402
    load_hand_visual_quality,
    skinning_residual_m,
    thumb_crown_gaps_m,
    thumb_wrap_shares,
)
from src.shared.python.model_appearance.hand_backends.skinned_pair import (  # noqa: E402
    build_grip_hands,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (  # noqa: E402
    default_settings,
)
from src.shared.python.model_appearance.skinned_club_hands import (  # noqa: E402
    owner_grip_spec,
)

OUT = Path(__file__).resolve().parent
_PACKAGE = Path("src/shared/python/model_appearance/hand_backends")
#: Side views: degrees round the grip from its top toward the trail side.
SIDE_VIEWS_DEG = (45.0, 90.0, 135.0)
#: Cross-sections: millimetres toward the head from the lead hand's station.
STATIONS_MM = (-5.0, 10.0, 25.0, 40.0, 50.0)
FRAME_PX = 700


def _cut(mesh, origin: np.ndarray, normal: np.ndarray) -> list[np.ndarray]:
    """Line segments where the plane through ``origin`` cuts ``mesh``."""
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    side = (v - origin) @ normal
    segments = []
    for tri in f[np.ptp(np.sign(side[f]), axis=1) > 0]:
        ends = [
            v[a] + side[a] / (side[a] - side[b]) * (v[b] - v[a])
            for a, b in zip(tri, np.roll(tri, -1), strict=True)
            if (side[a] > 0) != (side[b] > 0)
        ]
        if len(ends) == 2:
            segments.append(np.array(ends))
    return segments


def _sections(axes, pair) -> None:
    lead, trail = pair["lead"], pair["trail"]
    ids = lead.rest_asset.dominant_vertex_ids(("thumb_prox", "thumb_dist"))
    centre = lead.mesh.vertices[ids].mean(axis=0)
    out = np.array([centre[0], centre[1], 0.0]) / np.hypot(centre[0], centre[1])
    across = np.cross(np.array([0.0, 0.0, 1.0]), out)
    radius = 1e3 * lead.grip_radius_m
    turn = np.linspace(0.0, 2.0 * np.pi, 80)
    hands = ((lead, "tab:red"), (trail, "tab:blue"))
    for ax, z in zip(axes, STATIONS_MM, strict=False):
        for hand, colour in hands:
            for s in _cut(hand.mesh, np.array([0, 0, 1e-3 * z]), np.array([0, 0, 1.0])):
                ax.plot(1e3 * s[:, 0], 1e3 * s[:, 1], c=colour, lw=1)
        ax.plot(radius * np.cos(turn), radius * np.sin(turn), "k-")
        ax.set(xlim=(-50, 70), ylim=(-50, 70), aspect="equal")
        ax.set_title(f"Across the Grip, {z:.0f} mm Toward the Head", fontsize=9)
    ax = axes[-1]
    for hand, colour in hands:
        for s in _cut(hand.mesh, np.zeros(3), across):
            ax.plot(1e3 * s[:, 2], 1e3 * (s @ out), c=colour, lw=1)
    ax.axhspan(-radius, radius, color="k", alpha=0.3)
    ax.set(xlim=(-60, 120), ylim=(-50, 80), aspect="equal")
    ax.set_title("Along the Grip Through the Lead Thumb", fontsize=9)


def _stills(render, pair) -> np.ndarray:
    quality, grip = load_hand_visual_quality(), load_golf_grip()
    settings = render.load_hand_render_settings().with_overrides(frame_px=FRAME_PX)
    drawing = render.GripDrawing(
        knots=grip.drawing_knots(float(quality["owner_lead_below_cap_edge_m"])),
        cap_radius_m=grip.cap_radius_m,
        cap_height_m=grip.cap_height_m,
        cap_round_m=grip.cap_round_m,
        vent_radius_m=grip.vent_radius_m,
    )
    parts = [
        render.ScenePart(r.mesh, (*r.skin_rgb, 1.0), "skin", r.frame_focus)
        for r in pair.values()
    ]
    z = np.concatenate([p.mesh.vertices[:, 2] for p in parts])
    club = render.build_club_parts(
        float(z.min()), float(z.max()), settings=settings, grip=drawing
    )
    bases = [(np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]))]
    for deg in SIDE_VIEWS_DEG:
        toward = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])
        bases.append((np.cross(np.array([0.0, 0.0, 1.0]), toward), toward))
    frames = []
    for basis in bases:
        frame = render.render_scene(
            [*parts, *club], "back", basis=basis, settings=settings, side="lead"
        )
        path = OUT / "_frame.png"
        render.save_png(frame, path)
        frames.append(np.asarray(Image.open(path).convert("RGB")))
        path.unlink()
    return np.hstack(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer-root", default=os.environ.get("HAND_RENDERER_ROOT"))
    args = parser.parse_args()
    if not (Path(hand_backends.__file__).parent / "render.py").is_file():
        if not args.renderer_root:
            raise SystemExit("shared renderer not found: pass --renderer-root")
        hand_backends.__path__.append(str(Path(args.renderer_root) / _PACKAGE))
    from src.shared.python.model_appearance.hand_backends import render

    quality = load_hand_visual_quality()
    radius = float(quality["grip_radius_m"])
    owner = owner_grip_spec()
    legacy = dataclasses.replace(
        owner,
        trail=dataclasses.replace(
            owner.trail,
            club_cross_angle_deg=GripSpec.legacy().trail.club_cross_angle_deg,
        ),
    )
    flat = default_settings()
    wide = dataclasses.replace(flat.palm_pad, **quality["thumb_wrap_trial_pad"])
    cupped = dataclasses.replace(flat, pair_palm_wrap=wide)
    cases = (
        ("legacy_flat", legacy, flat),
        ("owner_flat", owner, flat),
        ("owner_trial_cupped", owner, cupped),
    )
    figure, axes = plt.subplots(len(cases), len(STATIONS_MM) + 1, figsize=(26, 13.5))
    rows, report = [], {}
    for row, (name, spec, settings) in enumerate(cases):
        pair = build_grip_hands(spec, grip_radius_m=radius, settings=settings)
        _sections(axes[row], pair)
        rows.append(_stills(render, pair))
        gaps = thumb_crown_gaps_m(
            pair["lead"],
            pair["trail"],
            station_m=float(quality["thumb_channel_station_m"]),
            min_facing=float(quality["thumb_channel_crown_facing"]),
        )
        report[name] = {
            "trail_cross_angle_deg": spec.trail.club_cross_angle_deg,
            "thumb_gaps_mm": [round(1e3 * float(g), 1) for g in gaps],
            "thumb_wrap_share": [
                round(float(share), 2)
                for share in thumb_wrap_shares(
                    pair["lead"],
                    pair["trail"],
                    station_m=float(quality["thumb_channel_station_m"]),
                    contact_m=float(quality["thumb_wrap_contact_m"]),
                )
            ],
            "trail_skin_off_bones_mm": round(
                1e3 * skinning_residual_m(pair["trail"]), 1
            ),
            "trail_station_mm": round(
                1e3 * float(pair["trail"].frame_from_rest[2, 3]), 1
            ),
        }
    figure.tight_layout()
    figure.savefig(OUT / "palm_on_thumb_sections.png", dpi=60)
    Image.fromarray(np.vstack(rows)).save(OUT / "palm_on_thumb_stills.png")
    sys.stdout.write(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
