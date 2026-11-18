"""The forearm's wrist end shaped from the skinned hand's wrist (#15264).

A skinned hand drawn on the club ends in a short cuff of forearm skin. The
model's forearm must meet it: same width, same thickness, turned the same
way. :func:`forearm_wrist_sections` reads each hand's section at its wrist
joint and gives it to the forearm body that carries that wrist, in that
body's frame at the spec's zero pose. Appearance only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

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

Array = NDArray[np.float64]
#: Half length of the band of hand skin about the wrist joint that is read.
_BAND_M = 0.004


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


def hand_wrist_sections(club: ClubAssembly) -> dict[str, WristSection]:
    """Per hand (``lead``/``trail``): its skin section at the wrist, club frame."""
    meshes, out = sch.skinned_hand_meshes(club), {}
    for side, (wrist, along) in sch.forearm_reach_m(club).items():
        mesh = meshes[f"{side}_hand"]
        rel = mesh.vertices[np.unique(mesh.faces)] - wrist
        band = rel[np.abs(rel @ along) < _BAND_M]
        flat = band - np.outer(band @ along, along)
        _, axes = np.linalg.eigh(np.cov(flat.T))
        wide, deep = axes[:, 2], axes[:, 1]
        (w0, w1), (d0, d1) = (
            (np.min(flat @ k), np.max(flat @ k)) for k in (wide, deep)
        )
        centre = 0.5 * (w0 + w1) * wide + 0.5 * (d0 + d1) * deep
        out[side] = WristSection(
            float(0.5 * (w1 - w0)), float(0.5 * (d1 - d0)), wide, centre_m=centre
        )
    return out


def _wrist_bodies(spec: Mapping[str, Any]) -> dict[str, tuple[str, Array]]:
    """Per hand: its forearm body and the rotation from the club frame to it.

    The lead forearm is the parent of the club body. The trail forearm is the
    parent of the body welded to the club by ``closure``; that weld carries
    the club frame to it.
    """
    club_body = club_body_name(spec)
    closure = spec.get("closure", {})
    if club_body is None or "placement_a" not in closure:
        return {}
    poses = spec_body_poses(spec)
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


def forearm_wrist_sections(
    spec: Mapping[str, Any], club: ClubAssembly | None
) -> dict[str, WristSection]:
    """Wrist section per forearm body, in its frame; empty without skinned hands.

    Sizes carry :func:`wrist_margin`, so the forearm's skin just covers the
    hand's cuff. Postcondition: every width axis is a unit vector.
    """
    backend = spec.get("subject", {}).get("hand_visual_backend")
    if club is None or backend != "skinned_human":
        return {}
    margin, hands, out = wrist_margin(), hand_wrist_sections(club), {}
    for side, (body, to_forearm) in _wrist_bodies(spec).items():
        section = hands[side]
        axis = to_forearm @ section.width_axis
        out[body] = WristSection(
            margin * section.half_width_m,
            margin * section.half_depth_m,
            axis / np.linalg.norm(axis),
            min(1.0, wrist_cap() * section.half_depth_m / section.half_width_m),
            to_forearm @ section.centre_m,
        )
    return out
