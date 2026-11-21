"""Stored kinetic-energy surface shading settings (#15396)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale

from .contracts import SegmentKineticEnergyFrame

STORED_KINETIC_ENERGY_VIEW_LABEL: Final[str] = "Stored kinetic energy"
KE_PROXY_GLOBE_LABEL: Final[str] = "KE proxy"
KE_COMPONENTS: Final[tuple[str, ...]] = ("total", "translational", "rotational")

# Peak total segment KE (J) on the tour-average anthro driver swing
# (tests/fixtures/club_face/swing_q_driver.npz on full_body_spec_anthro_driver.json).
# Recomputed via reference_swing_limits.compute_reference_swing_peak_segment_ke_j();
# regression-tested in tests/unit/segment_kinetic_energy/test_reference_limits.py.
REFERENCE_SWING_PEAK_SEGMENT_KE_J: Final[float] = 14931.485758486757


def default_stored_ke_color_scale(
    *,
    high_limit_j: float | None = None,
) -> ScalarColorScale:
    """Fixed swing-wide cividis scale; defaults from the reference swing peak."""
    hi = float(
        high_limit_j if high_limit_j is not None else REFERENCE_SWING_PEAK_SEGMENT_KE_J
    )
    if not math.isfinite(hi) or hi <= 0.0:
        raise ValueError("high_limit_j must be a finite positive number")
    return ScalarColorScale(
        enabled=False,
        scale_kind="sequential",
        units="J",
        sign_convention="unsigned",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=hi,
        display_transform="linear",
        opacity=1.0,
        blend_mode="replace",
    )


def component_values_j(
    frame: SegmentKineticEnergyFrame,
    mode: str = "total",
) -> dict[str, float | None]:
    """Map a KE frame to per-body scalar samples in joules."""
    if mode not in KE_COMPONENTS:
        raise ValueError(f"mode must be one of {KE_COMPONENTS}")
    values: dict[str, float | None] = {}
    for seg in frame.segments:
        if mode == "total":
            sample = float(seg.total_j)
        elif mode == "translational":
            sample = float(seg.translational_j)
        else:
            sample = float(seg.rotational_j)
        if not math.isfinite(sample) or sample < 0.0:
            values[seg.body] = None
        else:
            values[seg.body] = sample
    return values


def format_stored_ke_legend(
    scale: ScalarColorScale,
    *,
    mode: str,
    clipped_segments: frozenset[str] | None = None,
) -> str:
    """Build a status-bar legend line (no transfer wording)."""
    if not scale.enabled:
        return ""
    legend = scale.legend_descriptor()
    transform = legend.display_transform
    clipped = clipped_segments or frozenset()
    clip_note = ""
    if clipped:
        if len(clipped) <= 4:
            names = ", ".join(sorted(clipped))
            clip_note = f"; clipped: {names}"
        else:
            clip_note = f"; clipped: {len(clipped)} segments"
    return (
        f"{STORED_KINETIC_ENERGY_VIEW_LABEL} ({mode}): "
        f"{legend.low_limit:.1f}–{legend.high_limit:.1f} {legend.units}, "
        f"map {legend.colormap_id}, scale {transform}; "
        f"{legend.saturation_note}{clip_note}"
    )


def ui_strings_for_ke_view() -> Mapping[str, str]:
    """Aggregate user-visible strings for regression tests (no transfer wording)."""
    return {
        "view_label": STORED_KINETIC_ENERGY_VIEW_LABEL,
        "proxy_globe_label": KE_PROXY_GLOBE_LABEL,
    }


__all__ = [
    "KE_COMPONENTS",
    "KE_PROXY_GLOBE_LABEL",
    "REFERENCE_SWING_PEAK_SEGMENT_KE_J",
    "STORED_KINETIC_ENERGY_VIEW_LABEL",
    "component_values_j",
    "default_stored_ke_color_scale",
    "format_stored_ke_legend",
    "ui_strings_for_ke_view",
]
