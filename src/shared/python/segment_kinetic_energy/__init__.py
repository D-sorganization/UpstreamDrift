"""Engine-agnostic segment kinetic-energy time series and globe overlay (#15389)."""

from __future__ import annotations

from .contracts import (
    MUJOCO_SYSTEM_KE_ATOL,
    MUJOCO_SYSTEM_KE_RTOL,
    PEAK_HIGHLIGHT_FRACTION,
    SegmentKineticEnergyFrame,
    SegmentKineticEnergyPeak,
    SegmentKineticEnergyProvider,
    SegmentKineticEnergySample,
    read_segment_kinetic_energy_frame,
)
from .disclaimer import PROXY_DOC_REFERENCE, SEGMENT_KE_PROXY_DISCLAIMER
from .render_globes import globe_rgba_for_ke, radius_for_ke
from .series import SegmentKineticEnergySeries

__all__ = [
    "MUJOCO_SYSTEM_KE_ATOL",
    "MUJOCO_SYSTEM_KE_RTOL",
    "PEAK_HIGHLIGHT_FRACTION",
    "PROXY_DOC_REFERENCE",
    "SEGMENT_KE_PROXY_DISCLAIMER",
    "SegmentKineticEnergyFrame",
    "SegmentKineticEnergyPeak",
    "SegmentKineticEnergyProvider",
    "SegmentKineticEnergySample",
    "SegmentKineticEnergySeries",
    "globe_rgba_for_ke",
    "radius_for_ke",
    "read_segment_kinetic_energy_frame",
]
