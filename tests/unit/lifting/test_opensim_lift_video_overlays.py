"""Pure-data tests for lift video overlays and HUD (OSL-8, #13045)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.msk_lift_video_overlays import (
    draw_lift_legend,
    extract_lift_wrenches,
    muscle_color_rgb,
)
from src.shared.python.force_overlay.contracts import WrenchKind

pytestmark = pytest.mark.unit


def test_muscle_color_rgb_endpoints_and_midpoint() -> None:
    """Muscle coloring scales strictly from 0 (blue) to 1 (red)."""
    b_r, b_g, b_b = muscle_color_rgb(0.0)
    assert (b_r, b_g, b_b) == (0.0, 0.0, 1.0)

    r_r, r_g, r_b = muscle_color_rgb(1.0)
    assert (r_r, r_g, r_b) == (1.0, 0.0, 0.0)

    m_r, m_g, m_b = muscle_color_rgb(0.5)
    assert (m_r, m_g, m_b) == pytest.approx((0.5, 0.0, 0.5))


def test_muscle_color_rgb_clamping() -> None:
    """Out-of-bound activations are clamped to [0, 1]."""
    assert muscle_color_rgb(-0.5) == (0.0, 0.0, 1.0)
    assert muscle_color_rgb(1.5) == (1.0, 0.0, 0.0)


def test_extract_lift_wrenches_structure_and_units() -> None:
    """Overlay wrenches extract 4 vectors (2 foot GRF, 2 hand bar wrenches)."""
    from src.shared.python.lifting.bundle_targets import lift_targets
    from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES

    targets = lift_targets("squat", SAME_INPUT_LIFT_BUNDLES["squat"], stride=5)
    wrenches = extract_lift_wrenches(targets, frame_idx=0)
    assert len(wrenches) == 4

    labels = {w.label for w in wrenches}
    assert labels == {
        "contact:grf_left",
        "contact:grf_right",
        "grip:hand_l",
        "grip:hand_r",
    }

    w_dict = {w.label: w for w in wrenches}
    grf_l = w_dict["contact:grf_left"]
    assert grf_l.kind == WrenchKind.CONTACT
    assert grf_l.body == "calcn_l"
    assert grf_l.force_n is not None
    assert grf_l.force_n[1] > 500.0
    assert abs(grf_l.point_m[1]) < 0.05

    grip_l = w_dict["grip:hand_l"]
    assert grip_l.kind == WrenchKind.GRIP
    assert grip_l.body == "hand_l"
    assert grip_l.force_n is not None
    assert grip_l.point_m[1] > 0.8


def test_draw_lift_legend_layout_and_elements() -> None:
    """Legend renders 2D HUD with force scale, activation gradient, and status."""
    frame = np.full((600, 800, 3), 40, dtype=np.uint8)

    frame_active = draw_lift_legend(
        frame,
        lift="squat",
        time_s=1.50,
        in_activation_window=True,
        force_scale_n=500.0,
        force_scale_px=55,
    )
    assert frame_active.shape == (600, 800, 3)
    assert not np.array_equal(frame_active[:160, :270], frame[:160, :270])

    frame_inactive = draw_lift_legend(
        frame,
        lift="squat",
        time_s=0.20,
        in_activation_window=False,
        force_scale_n=500.0,
        force_scale_px=55,
    )
    assert frame_inactive.shape == (600, 800, 3)
    assert not np.array_equal(frame_active, frame_inactive)
