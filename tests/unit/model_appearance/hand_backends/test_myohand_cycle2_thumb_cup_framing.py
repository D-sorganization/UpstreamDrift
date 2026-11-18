"""TDD Cycle 2 tests for MyoHand: thumb chord clearance, 74% framing, and skeleton export."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.hand_backends.myohand import (
    build_myohand_bone_meshes,
    myohand_assets_ready,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    default_myohand_settings,
    load_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.skeleton_export import (
    export_bilateral_myohand_skeleton,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not myohand_assets_ready(), reason="myo_sim submodule not initialized"
    ),
]


def _xy_segment_min_radius(p0: np.ndarray, p1: np.ndarray) -> float:
    a = p0[:2].astype(np.float64)
    b = p1[:2].astype(np.float64)
    d = b - a
    denom = float(np.dot(d, d))
    t = 0.0 if denom < 1e-12 else float(np.clip(-np.dot(a, d) / denom, 0.0, 1.0))
    closest = a + t * d
    return float(np.hypot(closest[0], closest[1]))


def test_myohand_settings_keep_research_gate_and_calibrated_fill() -> None:
    """Research-only license gate stays enforced while render_frame_fill targets 0.74."""
    default_myohand_settings.cache_clear()
    cfg = load_myohand_settings()
    assert cfg.research_only is True
    assert cfg.license_gate_issue == 13763
    assert 0.72 <= cfg.render_frame_fill <= 0.76


@pytest.mark.parametrize("roll_deg", (-20.0, 0.0, 20.0))
@pytest.mark.parametrize("side", ("lead", "trail"))
def test_thumb_bone_chords_clear_grip_cylinder_and_distal_wraps(
    side: str, roll_deg: float
) -> None:
    """Thumb bone centroids must wrap around the grip without chord pass-through."""
    default_myohand_settings.cache_clear()
    cfg = load_myohand_settings()
    base = GripSpec.legacy()
    spec = (
        replace(base, lead=replace(base.lead, shaft_roll_deg=roll_deg))
        if side == "lead"
        else replace(base, trail=replace(base.trail, shaft_roll_deg=roll_deg))
    )
    meshes = build_myohand_bone_meshes(spec, side)  # type: ignore[arg-type]

    c_mc = meshes["firstmc"].vertices.mean(axis=0)
    c_px = meshes["proximal_thumb"].vertices.mean(axis=0)
    c_ds = meshes["distal_thumb"].vertices.mean(axis=0)
    min_allowed = cfg.club_cylinder_radius_m - cfg.penetration_tolerance_m
    assert _xy_segment_min_radius(c_mc, c_px) >= min_allowed
    assert _xy_segment_min_radius(c_px, c_ds) >= min_allowed

    v_ds = meshes["distal_thumb"].vertices
    distal_gap = float(np.hypot(v_ds[:, 0], v_ds[:, 1]).min() - cfg.grip_radius_m)
    assert -cfg.penetration_tolerance_m <= distal_gap <= cfg.thumb_contact_tolerance_m


def test_bilateral_skeleton_export_healthy_on_neutral_spec() -> None:
    """Shared skeleton JSON consumer receives >=23 finite joints on both sides."""
    payload = export_bilateral_myohand_skeleton(GripSpec.neutral())
    for side in ("lead", "trail"):
        side_data = payload[side]
        assert side_data["research_only"] is True
        joints = side_data["joints"]
        assert len(joints) >= 23
        for joint_info in joints.values():
            assert np.isfinite(joint_info["origin_m"]).all()
            assert np.isfinite(joint_info["axis_unit"]).all()
