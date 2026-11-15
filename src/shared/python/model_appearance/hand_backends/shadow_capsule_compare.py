"""Per-finger pad retarget checks: Shadow visuals vs capsule hand (#15122)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import GripPose, Hand
from src.shared.python.model_appearance.hand_backends.shadow import (
    ShadowHandPose,
    shadow_distal_pad_centroid,
    solve_shadow_hand_qpos,
)

Array = NDArray[np.float64]
FingerKey = Literal["index", "middle", "ring", "little", "thumb"]
_CONFIG_PATH = (
    Path(__file__).resolve().parents[5] / "scripts/config/grip_shadow_hand_visual.json"
)


@dataclass(frozen=True)
class PadDivergenceSettings:
    per_finger_tolerance_m: dict[str, float]
    thumb_documented_max_m: float


@lru_cache(maxsize=1)
def pad_divergence_settings() -> PadDivergenceSettings:
    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    block = raw.get("capsule_pad_divergence_tolerance_m", {})
    tol = {
        k: float(block[k])
        for k in ("index", "middle", "ring", "little")
        if k in block
    }
    for finger in ("index", "middle", "ring", "little"):
        tol.setdefault(finger, 0.014)
    thumb_max = float(raw.get("capsule_pad_divergence_thumb_max_m", 0.028))
    return PadDivergenceSettings(per_finger_tolerance_m=tol, thumb_documented_max_m=thumb_max)


def _capsule_pad_centroid(
    hand: Hand,
    *,
    grip_radius_m: float,
    pose: GripPose,
    finger: FingerKey,
) -> Array:
    baked = build_grip_hand_mesh(hand, grip_radius_m=grip_radius_m, pose=pose)
    return baked.fingertip_position(finger)


def per_finger_capsule_divergence_m(
    hand: Hand,
    pose: ShadowHandPose,
    *,
    grip_radius_m: float,
    grip_pose: GripPose,
    grip_spec: GripSpec,
) -> dict[str, float]:
    """Euclidean distance (m) between Shadow and capsule distal pad centroids."""
    out: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "little", "thumb"):
        shadow = shadow_distal_pad_centroid(hand, pose, finger)
        capsule = _capsule_pad_centroid(
            hand,
            grip_radius_m=grip_radius_m,
            pose=grip_pose,
            finger=finger,
        )
        out[finger] = float(np.linalg.norm(shadow - capsule))
    return out


def solve_and_compare(
    hand: Hand,
    *,
    grip_spec: GripSpec,
    grip_radius_m: float,
    grip_pose: GripPose,
) -> tuple[ShadowHandPose, dict[str, float]]:
    spec_hand = grip_spec.lead if hand == Hand.LEAD else grip_spec.trail
    solved = solve_shadow_hand_qpos(
        hand, spec_hand, grip_spec=grip_spec, grip_radius_m=grip_radius_m
    )
    gaps = per_finger_capsule_divergence_m(
        hand,
        solved,
        grip_radius_m=grip_radius_m,
        grip_pose=grip_pose,
        grip_spec=grip_spec,
    )
    return solved, gaps


__all__ = [
    "PadDivergenceSettings",
    "pad_divergence_settings",
    "per_finger_capsule_divergence_m",
    "solve_and_compare",
]