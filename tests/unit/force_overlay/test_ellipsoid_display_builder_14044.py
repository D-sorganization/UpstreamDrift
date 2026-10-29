"""Contract tests for CAP-5a ellipsoid display builder (#14038, #14044)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.force_overlay.palette import (
    FORCE_KIND_PALETTE,
    hand_ellipsoid_rgba,
)
from src.shared.python.force_overlay.ellipsoid_display import (
    EllipsoidDisplayBuilder,
    EllipsoidDisplayScale,
    LegendScaleBar,
    format_hud_capability,
)

pytestmark = pytest.mark.unit


def test_trail_hand_palette_does_not_clash_with_muscle() -> None:
    """Trail hand hue must use Okabe-Ito yellow (#F0E442) and not clash with muscle (#D55E00)."""
    trail = hand_ellipsoid_rgba("trail")
    lead = hand_ellipsoid_rgba("lead")
    assert FORCE_KIND_PALETTE["muscle"] == "#D55E00"
    assert not (
        math.isclose(trail[0], 0.835, abs_tol=0.01)
        and math.isclose(trail[1], 0.369, abs_tol=0.01)
    )
    # Okabe-Ito yellow (240/255, 228/255, 66/255) and blue (0, 114/255, 178/255)
    assert (
        math.isclose(trail[0], 240 / 255, abs_tol=0.02)
        and math.isclose(trail[1], 228 / 255, abs_tol=0.02)
        and math.isclose(trail[2], 66 / 255, abs_tol=0.02)
    )
    assert (
        math.isclose(lead[0], 0.0, abs_tol=0.01)
        and math.isclose(lead[1], 114 / 255, abs_tol=0.02)
        and math.isclose(lead[2], 178 / 255, abs_tol=0.02)
    )


def test_linear_percentile_scale_and_singular_filtering() -> None:
    """Linear radius scale maps 95th-percentile non-singular semi-axis to target_radius_m."""
    series = [(float(i), float(i) * 0.8, float(i) * 0.5) for i in range(1, 101)]
    # Add singular/non-finite samples to ensure exclusion
    series.extend([(0.0, float("nan"), 1e-12), (-5.0, 0.0, float("inf"))])
    scale = EllipsoidDisplayScale.compute(
        semi_axes_series=series,
        target_radius_m=0.09,
        kind="force-capacity",
        mode="auto",
    )
    valid = [v for f in series for v in f if math.isfinite(v) and v > 1e-9]
    expected = 0.09 / float(np.percentile(valid, 95))
    assert scale.scale_m_per_unit == pytest.approx(expected, rel=1e-4)
    assert scale.kind == "force-capacity" and scale.unit == "N"


def test_fixed_mode_scale() -> None:
    """Fixed mode sets exact scale factor for cross-clip comparison."""
    scale = EllipsoidDisplayScale.compute(
        [], target_radius_m=0.09, kind="acceleration-capacity", mode="fixed:0.005"
    )
    assert scale.scale_m_per_unit == pytest.approx(0.005)
    assert scale.unit == "m/s^2"
    with pytest.raises(ValueError, match="fixed:"):
        EllipsoidDisplayScale.compute([], mode="fixed:-0.01")


def test_legend_scale_bar_formatting() -> None:
    """Legend scale bar must format 1 cm = X <unit>."""
    scale = EllipsoidDisplayScale(0.01, "force-capacity", "auto", "N")
    bar = LegendScaleBar.from_scale(scale)
    assert bar.bar_length_m == pytest.approx(0.01)
    assert bar.physical_value == pytest.approx(1.0)
    assert bar.text == "1 cm = 1.0 N"

    scale_acc = EllipsoidDisplayScale(
        0.001, "acceleration-capacity", "fixed:0.001", "m/s^2"
    )
    assert LegendScaleBar.from_scale(scale_acc).text == "1 cm = 10.0 m/s^2"


def test_clamped_glyphs_are_dashed_and_flagged() -> None:
    """Scaled radii outside min/max bounds are clamped, dashed, and flagged."""
    b = EllipsoidDisplayBuilder(
        EllipsoidDisplayScale(0.01, "force-capacity", "auto", "N"),
        min_radius_m=0.01,
        max_radius_m=0.10,
    )
    normal = b.build_glyph((0, 0, 0), np.eye(3), (5, 5, 5), "lead")
    assert (
        normal.radii == (0.05, 0.05, 0.05)
        and not normal.clamped
        and normal.style == "wireframe"
    )

    clamped_lo = b.build_glyph((0, 0, 0), np.eye(3), (0.5, 5, 5), "trail")
    assert clamped_lo.radii[0] == pytest.approx(0.01) and clamped_lo.clamped
    assert clamped_lo.style == "dashed"

    clamped_hi = b.build_glyph((0, 0, 0), np.eye(3), (20, 5, 5), "lead")
    assert clamped_hi.radii[0] == pytest.approx(0.10) and clamped_hi.clamped
    assert clamped_hi.style == "dashed"


def test_kind_style_and_identity_hue() -> None:
    """Kind determines style (shell vs wireframe), identity determines Okabe-Ito hue."""
    b_shell = EllipsoidDisplayBuilder(
        EllipsoidDisplayScale(0.1, "mobility", "auto", "kg^-1/2")
    )
    lead_item = b_shell.build_glyph((0, 0, 0), np.eye(3), (1, 1, 1), "lead")
    assert lead_item.style == "shell" and lead_item.rgba == hand_ellipsoid_rgba("lead")

    b_wire = EllipsoidDisplayBuilder(
        EllipsoidDisplayScale(0.1, "force-capacity", "auto", "N")
    )
    trail_item = b_wire.build_glyph((0, 0, 0), np.eye(3), (1, 1, 1), "trail")
    assert trail_item.style == "wireframe" and trail_item.rgba == hand_ellipsoid_rgba(
        "trail"
    )


def test_no_colour_channel_for_condition_number() -> None:
    """Condition number appears in HUD output and never alters RGBA."""
    b = EllipsoidDisplayBuilder(
        EllipsoidDisplayScale(0.01, "force-capacity", "auto", "N")
    )
    i1 = b.build_glyph((0, 0, 0), np.eye(3), (10, 10, 10), "lead")
    i2 = b.build_glyph((0, 0, 0), np.eye(3), (100, 10, 1), "lead")
    assert i1.rgba == i2.rgba == hand_ellipsoid_rgba("lead")

    hud = format_hud_capability("lead", "force-capacity", (100, 10, 1), "N")
    assert "cond=100.0" in hud and "N" in hud
