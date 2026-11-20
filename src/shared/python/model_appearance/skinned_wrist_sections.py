"""The forearm's wrist end shaped from the skinned hand's wrist (#15264).

A skinned hand drawn on the club ends in a short cuff of forearm skin. The
model's forearm must meet it: same width, same thickness, turned the same
way. :func:`forearm_wrist_sections` reads each hand's section at its wrist
joint and gives it to the forearm body that carries that wrist, in that
body's frame with the wrist straight. Appearance only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    club_body_name,
)
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
)
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

Array = NDArray[np.float64]


@dataclass(frozen=True)
class WristSection:
    """Half width and half depth (m) of a wrist, and its width direction."""

    half_width_m: float
    half_depth_m: float
    width_axis: Array
    #: Length of the forearm's rounded end over the half width, at most 1.
    cap: float = 1.0
    #: Centre of the skin section off the wrist joint (m).
    centre_m: Array = field(default_factory=lambda: np.zeros(3))
    #: Fraction of the forearm by which its taper has reached this section.
    taper_end: float = 1.0
    #: False on the near body of a two-body forearm (draws no wrist end itself).
    shapes_end: bool = True
    #: Muscle belly ``(gain, position, width)`` of the forearm that ends here.
    belly: tuple[float, float, float] = library.NO_BELLY


def wrist_margin() -> float:
    """Forearm wrist size over the hand's wrist size (a setting)."""
    quality: Mapping[str, Any] = load_hand_visual_quality()
    margin = float(quality["forearm_wrist_margin"])
    if not 0.9 <= margin <= 1.5:
        raise ValueError(f"forearm_wrist_margin must be in [0.9, 1.5], got {margin}")
    return margin


def wrist_cap() -> float:
    """Length of the forearm's rounded wrist end over its half depth (a setting).

    At 1 the end is round about the wrist's width axis, the axis the hand
    flexes about, so the hand's cut edge stays under the forearm's skin at
    every flexion angle. A ball of the half width overhangs the hand's back.
    """
    quality: Mapping[str, Any] = load_hand_visual_quality()
    cap = float(quality["forearm_wrist_cap"])
    if not 0.0 < cap <= 1.4:
        raise ValueError(f"forearm_wrist_cap must be in (0, 1.4], got {cap}")
    return cap


def wrist_taper_end() -> float:
    """Fraction of the forearm by which it has narrowed to the wrist (a setting,
    just short of 1 so it is narrowest at the wrist joint; #15299)."""
    quality: Mapping[str, Any] = load_hand_visual_quality()
    end = float(quality["forearm_wrist_shape"]["taper_end"])
    if not library.PART_FLATTEN_FROM["forearm"] < end <= 1.0:
        raise ValueError(f"taper_end must be past the flattening and <= 1, got {end}")
    return end


def forearm_belly() -> tuple[float, float, float]:
    """Muscle belly ``(gain, position, width)`` of a forearm that ends in a
    skinned hand (a setting): fullest a third of the way down (#15299)."""
    quality: Mapping[str, Any] = load_hand_visual_quality()
    gain, position, width = map(float, quality["forearm_wrist_shape"]["belly"])
    if not (0.0 <= gain <= 0.5 and 0.0 < position < 1.0 and width > 0.0):
        raise ValueError(f"belly gain, position or width out of range: {gain=}")
    return gain, position, width


def hand_wrist_sections(club: ClubAssembly) -> dict[str, WristSection]:
    """Per hand (``lead``/``trail``): the scanned hand's own skin section at the
    wrist, club frame, before it is widened for drawing."""
    meshes, out = sch.skinned_hand_meshes(club, widen_wrist=False), {}
    for side, (wrist, along) in sch.forearm_reach_m(club).items():
        mesh = meshes[f"{side}_hand"]
        rel = mesh.vertices[np.unique(mesh.faces)] - wrist
        half_width, half_depth, wide, centre = sch.wrist_section(rel, along)
        out[side] = WristSection(half_width, half_depth, wide, centre_m=centre)
    return out


def _wrist_bodies(
    spec: Mapping[str, Any], q: Mapping[str, float] | None = None
) -> dict[str, tuple[str, Array]]:
    """Per hand: its forearm body and the rotation from the club frame to it
    at coordinates ``q`` (the spec's zero pose without).

    The lead forearm is the parent of the club body. The trail forearm is the
    parent of the body welded to the club by ``closure``; that weld carries
    the club frame to it.
    """
    club_body = club_body_name(spec)
    closure = spec.get("closure", {})
    if club_body is None or "placement_a" not in closure:
        return {}
    poses = spec_body_poses(spec, q)
    weld = np.asarray(closure["placement_a"], float) @ np.linalg.inv(
        np.asarray(closure["placement_b"], float)
    )
    parents = {j["child"]: j["parent"] for j in spec["joints"]}
    out = {}
    for side, hand_body, turn in (
        ("lead", club_body, np.eye(3)),
        ("trail", closure.get("body_a"), weld[:3, :3]),
    ):
        forearm = parents.get(hand_body)
        if forearm is None or library.classify_body(forearm) != "forearm":
            continue
        to_forearm = poses[forearm][:3, :3].T @ poses[hand_body][:3, :3] @ turn
        out[side] = (forearm, to_forearm)
    return out


def straight_wrists(
    spec: Mapping[str, Any], club: ClubAssembly
) -> dict[str, tuple[str, Array]]:
    """Per hand: its forearm body and the rotation from the club frame to it
    with the wrist straight (#15299).

    The spec's zero pose bends both wrists about 50 degrees, so a forearm end
    shaped there sits turned and off centre under the hand. Straight is where
    the wrist angles bring the hand's forearm direction onto the model's.
    """
    bent = _wrist_bodies(spec)
    forearms = {body for body, _ in bent.values()}
    names = [
        primitive["coordinate"]
        for joint in spec["joints"]
        if joint["parent"] in forearms
        and library.classify_body(joint["child"]) != "forearm"
        for primitive in joint["primitives"]
    ]
    if not names:
        return bent
    reach = sch.forearm_reach_m(club)
    axes = {
        c.body: np.subtract(c.end_m, c.start_m) / c.length_m()
        for c in derive_visual_skeleton(spec).capsules
        if c.body in forearms
    }

    def miss(angles: Array) -> Array:
        wrists = _wrist_bodies(spec, dict(zip(names, map(float, angles), strict=True)))
        parts = [turn @ reach[s][1] + axes[b] for s, (b, turn) in wrists.items()]
        return np.concatenate(parts)

    angles = least_squares(miss, np.zeros(len(names))).x
    return _wrist_bodies(spec, dict(zip(names, map(float, angles), strict=True)))


def forearm_wrist_sections(
    spec: Mapping[str, Any], club: ClubAssembly | None
) -> dict[str, WristSection]:
    """Wrist section per forearm body, in its frame; empty without skinned hands.

    Sizes carry :func:`wrist_margin`, which brings the scanned hand's thin
    wrist to a human girth; the drawn hand's rim is widened to lie just under
    it (``skinned_club_hands.wrist_blend``). Postcondition: every width axis
    is a unit vector.
    """
    backend = spec.get("subject", {}).get("hand_visual_backend")
    if club is None or backend != "skinned_human":
        return {}
    margin, hands, out = wrist_margin(), hand_wrist_sections(club), {}
    blend = sch.wrist_blend()
    for side, (body, to_forearm) in straight_wrists(spec, club).items():
        section = hands[side]
        axis = to_forearm @ section.width_axis
        centre = to_forearm @ blend.centred(section.centre_m)
        out[body] = WristSection(
            margin * section.half_width_m,
            margin * section.half_depth_m,
            axis / np.linalg.norm(axis),
            min(1.0, wrist_cap() * section.half_depth_m / section.half_width_m),
            centre,
            wrist_taper_end(),
            belly=forearm_belly(),
        )
    return out
