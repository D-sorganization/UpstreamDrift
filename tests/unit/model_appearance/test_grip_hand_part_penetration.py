"""Part-part penetration guards for the procedural grip hand mesh (#15141)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand
from src.shared.python.model_appearance.mesh_penetration import signed_mesh_distances

pytestmark = pytest.mark.unit

_CONFIG = (
    Path(__file__).resolve().parents[3] / "scripts/config/grip_hand_mesh_quality.json"
)


def _load_limits() -> tuple[dict[str, float], float]:
    raw = json.loads(_CONFIG.read_text(encoding="utf-8"))
    limits = {str(k): float(v) for k, v in raw["max_part_penetration_fraction"].items()}
    clearance = float(raw["part_penetration_depth_m"])
    return limits, clearance


def _penetration_fraction(source: np.ndarray, targets, clearance_m: float) -> float:
    """Share deeper than the allowed clearance inside any target surface."""
    signed = np.vstack([signed_mesh_distances(source, target) for target in targets])
    return float((signed.min(axis=0) < -clearance_m).mean())


def _max_finger_block_overlap(result, block: str, clearance_m: float) -> float:
    parts = {p.name: p for p in result.components}
    return max(
        _penetration_fraction(
            parts[f"{finger}_proximal"].mesh.vertices, [parts[block].mesh], clearance_m
        )
        for finger in FINGER_NAMES
    )


@pytest.mark.parametrize("hand", [Hand.LEAD, Hand.TRAIL])
@pytest.mark.parametrize("radius", [0.0127, 0.014, 0.0145])
def test_hand_parts_do_not_over_penetrate(hand, radius):
    limits, clearance_m = _load_limits()
    result = build_grip_hand_mesh(hand, grip_radius_m=radius, pose=DEFAULT_GRIP_POSE)
    parts = {p.name: p for p in result.components}
    thumb_meshes = [parts[name].mesh for name in ("thumb_proximal", "thumb_distal")]

    distal_in_thumb = _penetration_fraction(
        parts["index_distal"].mesh.vertices, thumb_meshes, clearance_m
    )
    middle_in_thumb = _penetration_fraction(
        parts["index_middle"].mesh.vertices, thumb_meshes, clearance_m
    )
    assert distal_in_thumb <= limits["lead_index_distal_in_thumb"], distal_in_thumb
    assert middle_in_thumb <= limits["lead_index_middle_in_thumb"], middle_in_thumb

    block_pen = max(
        _max_finger_block_overlap(result, "palm", clearance_m),
        _max_finger_block_overlap(result, "dorsum", clearance_m),
    )
    assert block_pen <= limits["proximal_in_palm_or_dorsum"], block_pen
