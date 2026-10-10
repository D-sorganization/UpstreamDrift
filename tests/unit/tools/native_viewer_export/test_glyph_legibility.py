"""Tests for default glyph style legibility at 720p (NV-9, #11697)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
    build_glyphs,
)
from src.shared.python.golf_view_presets import (
    VIEW_ORDER,
    VIEWER_FOV_Y_RAD,
    get_view_preset,
)
from src.tools.native_viewer_export.core import (
    HQ_SIZE,
    default_glyph_style,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

LOOKAT = (1.0, 0.0, 0.9)


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_default_glyph_style_is_legible_at_720p(view: str) -> None:
    preset = get_view_preset(view)
    style = default_glyph_style()
    dist = preset.default_distance_m

    # 720p full-frame and 2x2 multiview tile
    for _width, height, min_shaft_px, min_arrow_px in [
        (HQ_SIZE[0], HQ_SIZE[1], 7.0, 80.0),  # 1280x720 full frame
        (HQ_SIZE[0] // 2, HQ_SIZE[1] // 2, 3.5, 40.0),  # 640x360 2x2 tile
    ]:
        fy = (height / 2.0) / math.tan(VIEWER_FOV_Y_RAD / 2.0)
        # Shaft diameter in pixels at target distance
        shaft_px = (2.0 * style.shaft_radius_m) * (fy / dist)
        assert shaft_px >= min_shaft_px

        # Reference 500 N arrow length in pixels
        arrow_len_m = 500.0 * style.force_scale_for(WrenchKind.CONTACT)
        arrow_px = arrow_len_m * (fy / dist)
        assert arrow_px >= min_arrow_px


def test_default_glyph_style_caps_torque_arc_radius() -> None:
    style = default_glyph_style(body_mass_kg=77.6)
    assert style.max_torque_length_m is not None
    assert style.max_torque_length_m <= 0.7
    # Max torque radius must be <= 0.35 m (so arcs don't balloon to 1m+)
    assert (style.max_torque_length_m / 2.0) <= 0.35

    # Build a frame with a huge torque (e.g. 500 N*m near impact)
    frame = ForceTorqueFrame(
        time_s=0.0,
        engine="test",
        wrenches=(
            OverlayWrench(
                WrenchKind.JOINT_ACTUATOR,
                "joint:lumbar_torque",
                "lumbar",
                (1.0, 0.0, 0.9),
                torque_nm=(0.0, 0.0, 500.0),
                source="test",
            ),
        ),
    )
    glyphs = build_glyphs(frame, style)
    assert len(glyphs.torque_arcs) == 1
    arc = glyphs.torque_arcs[0]
    assert arc.radius_m <= 0.35
    assert arc.clamped is True
