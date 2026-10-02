"""Engine-agnostic force and torque overlay pipeline (ADR-0052, #11285).

This package provides:
- Core force/torque contracts and schemas (contracts.py)
- Time-indexed immutable series with linear interpolation (series.py)

Headless import safe: this module and its core submodules never import GUI
or physics engine SDKs (PyQt6, matplotlib.pyplot, mujoco, pydrake,
pinocchio, opensim).
"""

from __future__ import annotations

from .contracts import (
    DEFAULT_OVERLAY_UNITS,
    ForceTorqueFrame,
    ForceTorqueProvider,
    OverlayWrench,
    WrenchKind,
    read_force_torque_frame,
)
from .glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
    build_glyphs,
    scale_for_view,
)
from .palette import FORCE_KIND_PALETTE
from .series import ForceTorqueSeries

__all__ = [
    "ArrowGlyph",
    "DEFAULT_OVERLAY_UNITS",
    "FORCE_KIND_PALETTE",
    "ForceGlyphStyle",
    "ForceTorqueFrame",
    "ForceTorqueProvider",
    "ForceTorqueSeries",
    "GlyphSet",
    "LegendSpec",
    "OverlayWrench",
    "TorqueArcGlyph",
    "WrenchKind",
    "build_glyphs",
    "read_force_torque_frame",
    "scale_for_view",
]
