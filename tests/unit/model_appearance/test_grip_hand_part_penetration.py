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

pytestmark = pytest.mark.unit

_CONFIG = (
    Path(__file__).resolve().parents[3] / "scripts/config/grip_hand_mesh_quality.json"
)


def _load_limits() -> tuple[dict[str, float], float]:
    raw = json.loads(_CONFIG.read_text(encoding="utf-8"))
    limits = {str(k): float(v) for k, v in raw["max_part_penetration_fraction"].items()}
    clearance = float(raw.get("part_vertex_clearance_m", 0.002))
    return limits, clearance


def _vertex_overlap_fraction(
    source: np.ndarray, target: np.ndarray, clearance_m: float
) -> float:
    """Share of source vertices within ``clearance_m`` of any target vertex."""
    if source.size == 0 or target.size == 0:
        return 0.0
    deltas = source[:, None, :] - target[None, :, :]
    nearest = np.linalg.norm(deltas, axis=2).min(axis=1)
    return float((nearest < clearance_m).mean())


def _max_finger_block_overlap(result, block: str, clearance_m: float) -> float:
    parts = {p.name: p for p in result.components}
    block_vertices = parts[block].mesh.vertices
    return max(
        _vertex_overlap_fraction(
            parts[f"{finger}_proximal"].mesh.vertices, block_vertices, clearance_m
        )
        for finger in FINGER_NAMES
    )


def test_lead_hand_parts_do_not_over_penetrate():
    limits, clearance_m = _load_limits()
    result = build_grip_hand_mesh(
        Hand.LEAD, grip_radius_m=0.014, pose=DEFAULT_GRIP_POSE
    )
    parts = {p.name: p for p in result.components}
    thumb_vertices = np.vstack(
        [parts[name].mesh.vertices for name in ("thumb_proximal", "thumb_distal")]
    )

    distal_in_thumb = _vertex_overlap_fraction(
        parts["index_distal"].mesh.vertices, thumb_vertices, clearance_m
    )
    middle_in_thumb = _vertex_overlap_fraction(
        parts["index_middle"].mesh.vertices, thumb_vertices, clearance_m
    )
    assert distal_in_thumb <= limits["lead_index_distal_in_thumb"], distal_in_thumb
    assert middle_in_thumb <= limits["lead_index_middle_in_thumb"], middle_in_thumb

    block_pen = max(
        _max_finger_block_overlap(result, "palm", clearance_m),
        _max_finger_block_overlap(result, "dorsum", clearance_m),
    )
    assert block_pen <= limits["proximal_in_palm_or_dorsum"], block_pen
