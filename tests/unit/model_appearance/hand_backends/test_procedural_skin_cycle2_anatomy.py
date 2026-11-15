"""TDD Cycle 2 tests for procedural_skin: distal contact, opposed thumb, knuckle relief."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.grip_pose import GripStyle
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
)
from src.shared.python.model_appearance.hand_backends.procedural_skin import (
    _build_canonical_hand,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    HandVisualSettings,
)

pytestmark = pytest.mark.unit


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / max(norm, 1e-12)


def test_cycle2_anatomy_settings_present() -> None:
    """All Cycle 2 anatomical tuning constants live on HandVisualSettings."""
    cfg = DEFAULT_HAND_VISUAL_SETTINGS
    assert isinstance(cfg, HandVisualSettings)
    assert cfg.interdigital_valley_m > 0.0
    assert cfg.thumb_base_opposition_rad > 0.0
    assert cfg.thenar_flare_scale > 1.0
    assert cfg.distal_pad_t_start >= 0.55
    assert cfg.distal_cap_shift_m > 0.0


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_distal_finger_pads_wrap_onto_grip_both_hands(side: HandRole) -> None:
    """Distal finger tips must seat onto the grip rather than hovering open."""
    spec = GripSpec.legacy(style=GripStyle.TEN_FINGER)
    angles = HandJointAngles.from_grip_spec(spec, side)
    built = _build_canonical_hand(
        spec,
        side,
        angles,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=DEFAULT_HAND_VISUAL_SETTINGS,
    )
    for finger in ("index", "middle", "ring", "little"):
        verts = built.finger_meshes[finger].vertices
        tip = built.finger_joints[finger][-1]
        dist = np.linalg.norm(verts - tip, axis=1)
        distal = verts[dist <= 0.014]
        assert len(distal) >= 16, (side, finger, len(distal))
        clearances = np.hypot(distal[:, 0], distal[:, 1]) - DEFAULT_VISUAL_GRIP_RADIUS_M
        min_gap = float(clearances.min())
        assert -0.0005 <= min_gap <= 0.0022, (side, finger, min_gap)


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_thumb_opposes_across_grip_at_default_straightness(side: HandRole) -> None:
    """Thumb must wrap diagonally and seat on the grip even at thumb_straightness=1.0."""
    spec = GripSpec.legacy()
    hand_spec = spec.lead if side == "lead" else spec.trail
    assert hand_spec.thumb_straightness == pytest.approx(1.0)
    angles = HandJointAngles.from_grip_spec(spec, side)
    built = _build_canonical_hand(
        spec,
        side,
        angles,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=DEFAULT_HAND_VISUAL_SETTINGS,
    )
    th_v = built.finger_meshes["thumb"].vertices
    clearances = np.hypot(th_v[:, 0], th_v[:, 1]) - DEFAULT_VISUAL_GRIP_RADIUS_M
    min_gap = float(clearances.min())
    assert -0.0005 <= min_gap <= 0.0025, (side, min_gap)

    z_lo = float(np.quantile(th_v[:, 2], 0.15))
    z_hi = float(np.quantile(th_v[:, 2], 0.85))
    base_center = th_v[th_v[:, 2] <= z_lo].mean(axis=0)
    tip_center = th_v[th_v[:, 2] >= z_hi].mean(axis=0)
    r_dir = _unit(np.array([base_center[0], base_center[1], 0.0], dtype=float))
    tan_dir = _unit(np.cross(np.array([0.0, 0.0, 1.0], dtype=float), r_dir))
    tangential_sweep_m = abs(float(np.dot(tip_center - base_center, tan_dir)))
    assert tangential_sweep_m >= 0.010, (side, tangential_sweep_m)


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_palm_dorsum_shows_knuckle_domes_and_interdigital_valleys(
    side: HandRole,
) -> None:
    """Dorsal knuckle domes must stand proud of adjacent interdigital valleys."""
    spec = GripSpec.legacy()
    angles = HandJointAngles.from_grip_spec(spec, side)
    built = _build_canonical_hand(
        spec,
        side,
        angles,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=DEFAULT_HAND_VISUAL_SETTINGS,
    )
    palm_v = built.palm_mesh.vertices
    ordered = ("little", "ring", "middle", "index")
    anchors = np.array([built.mcp_anchors[f] for f in ordered], dtype=float)
    knuckle_span = anchors[-1] - anchors[0]
    w_dir = knuckle_span / float(np.linalg.norm(knuckle_span))

    def _local_dorsal_rho(center: np.ndarray) -> float:
        delta = palm_v - center
        d_w = np.abs(delta @ w_dir)
        d_3d = np.linalg.norm(delta, axis=1)
        near = palm_v[(d_w <= 0.0045) & (d_3d <= 0.018)]
        assert len(near) >= 4
        return float(np.quantile(np.hypot(near[:, 0], near[:, 1]), 0.95))

    knuckle_peaks = [_local_dorsal_rho(anchors[i]) for i in range(4)]
    valley_peaks = [
        _local_dorsal_rho(0.5 * (anchors[i] + anchors[i + 1])) for i in range(3)
    ]
    mean_relief = min(
        0.5 * (knuckle_peaks[i] + knuckle_peaks[i + 1]) - valley_peaks[i]
        for i in range(3)
    )
    assert mean_relief >= 0.0018, (side, knuckle_peaks, valley_peaks, mean_relief)
    min_side_relief = min(
        min(knuckle_peaks[i] - valley_peaks[i], knuckle_peaks[i + 1] - valley_peaks[i])
        for i in range(3)
    )
    assert min_side_relief >= 0.0010, (
        side,
        knuckle_peaks,
        valley_peaks,
        min_side_relief,
    )
