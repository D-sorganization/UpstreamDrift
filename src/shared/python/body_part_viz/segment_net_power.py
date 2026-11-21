"""Segment net-power (dE/dt) shading and interface-power presentation (#15399)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.shared.python.biomechanics.robertson_winter_interface_power import (
    InterfacePowerFrame,
    InterfacePowerJointSample,
)

from .scalar_shading import ScalarColorLegend, ScalarColorScale
from .segment_scalar_frame import SegmentScalarFrame

SEGMENT_NET_POWER_LABEL: Final[str] = "dE/dt (net power)"
SEGMENT_NET_POWER_UNITS: Final[str] = "W"
INTERFACE_TRANSFER_LABEL: Final[str] = "Interface transfer"
JOINT_COORDINATE_POWER_LABEL: Final[str] = "Joint coordinate power (tau·qdot)"
NET_POWER_COLORMAP_ID: Final[str] = "brown-teal"

# Color-blind-safe diverging endpoints (distinct from tension/compression red/blue).
NET_POWER_LOW_COLOR: Final[str] = "#a6611a"
NET_POWER_HIGH_COLOR: Final[str] = "#018571"
NET_POWER_NEUTRAL_COLOR: Final[str] = "#f7f7f7"

_LIMITS_PATH = (
    Path(__file__).resolve().parents[4] / "scripts/config/segment_net_power_limits.json"
)

__all__ = [
    "INTERFACE_TRANSFER_LABEL",
    "JOINT_COORDINATE_POWER_LABEL",
    "NET_POWER_COLORMAP_ID",
    "NET_POWER_HIGH_COLOR",
    "NET_POWER_LOW_COLOR",
    "NET_POWER_NEUTRAL_COLOR",
    "SEGMENT_NET_POWER_LABEL",
    "SEGMENT_NET_POWER_UNITS",
    "NetPowerLegendExtras",
    "SegmentNetPowerPresentation",
    "default_net_power_color_scale",
    "format_net_power_legend",
    "interface_edge_label",
    "load_default_symmetric_limit_w",
    "net_power_frame_from_interface",
]


@dataclass(frozen=True)
class NetPowerLegendExtras:
    """Gravity, applied-force power and balance residual beside the colour bar."""

    gravity_power_w: float
    applied_force_power_w: float
    max_balance_residual_w: float
    clipped_segments: tuple[str, ...]

    def summary_lines(self) -> tuple[str, ...]:
        lines = [
            f"Gravity power (m·g·v_com): {self.gravity_power_w:.2f} W",
            f"Applied-force power (xfrc): {self.applied_force_power_w:.2f} W",
            f"Max |d(KE+PE)/dt balance residual|: {self.max_balance_residual_w:.2f} W",
        ]
        if self.clipped_segments:
            names = ", ".join(self.clipped_segments[:5])
            extra = " …" if len(self.clipped_segments) > 5 else ""
            lines.append(f"Saturated segments: {names}{extra}")
        return tuple(lines)


@dataclass(frozen=True)
class SegmentNetPowerPresentation:
    """Wire bundle for one rendered frame."""

    segment_frame: SegmentScalarFrame
    interface_frame: InterfacePowerFrame
    legend: ScalarColorLegend
    extras: NetPowerLegendExtras


def load_default_symmetric_limit_w() -> float:
    payload = json.loads(_LIMITS_PATH.read_text(encoding="utf-8"))
    limit = float(payload["symmetric_limit_w"])
    if limit <= 0.0 or not limit < float("inf"):
        raise ValueError("symmetric_limit_w must be a positive finite number")
    return limit


def default_net_power_color_scale(
    symmetric_limit_w: float | None = None,
) -> ScalarColorScale:
    limit = (
        float(symmetric_limit_w)
        if symmetric_limit_w is not None
        else load_default_symmetric_limit_w()
    )
    if limit <= 0.0:
        raise ValueError("symmetric_limit_w must be positive")
    return ScalarColorScale(
        enabled=False,
        scale_kind="diverging",
        units=SEGMENT_NET_POWER_UNITS,
        sign_convention="signed-zero-center",
        low_limit=-limit,
        high_limit=limit,
        deadband=0.0,
        low_color=NET_POWER_LOW_COLOR,
        high_color=NET_POWER_HIGH_COLOR,
        neutral_color=NET_POWER_NEUTRAL_COLOR,
        colormap_id=NET_POWER_COLORMAP_ID,
        opacity=0.82,
        blend_mode="replace",
    )


def _clipped_segments(
    frame: InterfacePowerFrame, scale: ScalarColorScale
) -> tuple[str, ...]:
    if not scale.enabled:
        return ()
    limit = abs(scale.high_limit)
    clipped: list[str] = []
    for segment in frame.segments:
        rate = segment.mechanical_energy_rate_w
        if abs(rate) > limit:
            clipped.append(segment.body)
    return tuple(clipped)


def net_power_frame_from_interface(
    interface: InterfacePowerFrame,
    scale: ScalarColorScale,
) -> SegmentNetPowerPresentation:
    values = {seg.body: seg.mechanical_energy_rate_w for seg in interface.segments}
    segment_frame = SegmentScalarFrame(
        time_s=interface.time_s,
        values=values,
        source=("Robertson-Winter segment d(KE+PE)/dt (#15390)"),
        units=SEGMENT_NET_POWER_UNITS,
        quantity_label=SEGMENT_NET_POWER_LABEL,
    )
    gravity = sum(seg.gravity_power_w for seg in interface.segments)
    applied = sum(seg.external_power_w for seg in interface.segments)
    residual = max(
        (abs(seg.balance_residual_w) for seg in interface.segments),
        default=0.0,
    )
    clipped = _clipped_segments(interface, scale)
    legend = scale.legend_descriptor()
    extras = NetPowerLegendExtras(gravity, applied, residual, clipped)
    return SegmentNetPowerPresentation(
        segment_frame=segment_frame,
        interface_frame=interface,
        legend=legend,
        extras=extras,
    )


def format_net_power_legend(
    presentation: SegmentNetPowerPresentation,
    *,
    interface_labels: tuple[str, ...] = (),
) -> str:
    legend = presentation.legend
    parts = [
        SEGMENT_NET_POWER_LABEL,
        f"{legend.low_limit:.0f} … {legend.high_limit:.0f} {legend.units}",
        f"colormap {NET_POWER_COLORMAP_ID}",
        legend.saturation_note,
        *presentation.extras.summary_lines(),
    ]
    if interface_labels:
        preview = "; ".join(interface_labels[:3])
        extra = " …" if len(interface_labels) > 3 else ""
        parts.append(f"{INTERFACE_TRANSFER_LABEL}: {preview}{extra}")
    return " | ".join(parts)


def interface_edge_label(joint: InterfacePowerJointSample) -> str:
    """Interface-edge text; the word transfer appears only here."""
    magnitude = joint.power_on_child_w
    direction = "to child" if magnitude >= 0.0 else "from child"
    return (
        f"{INTERFACE_TRANSFER_LABEL}: {magnitude:.1f} W "
        f"({direction}, {joint.parent_body}–{joint.child_body})"
    )
