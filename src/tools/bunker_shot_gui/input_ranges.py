"""Accepted input ranges for the BunkerShot3D workbench (issue #9545).

One definition, two consumers: the PyQt panels (through
:mod:`~src.tools.bunker_shot_gui.design`, which re-exports it) and the v1 API
route. It lives in its own module because it must import nothing: the API
route registry imports every route module at boot, and ``design`` pulls in
the geometry and sand packages (seconds of import time, see #8943).
"""

from __future__ import annotations

from types import MappingProxyType

__all__ = ["INPUT_RANGES"]

INPUT_RANGES = MappingProxyType(
    {
        "loft_deg": (40.0, 66.0),
        "marketed_bounce_deg": (0.0, 20.0),
        "sole_width_mm": (8.0, 26.0),
        "entry_height_mm": (1.5, 7.5),
        "leading_edge_radius_mm": (2.0, 12.0),
        "camber_area_mm2": (20.0, 70.0),
        "heel_relief_fraction": (0.0, 0.6),
        "toe_relief_fraction": (0.0, 0.6),
        "firmness_kg_per_cm2": (1.2, 3.2),
        "clubhead_speed_mps": (10.0, 40.0),
        "attack_angle_deg": (-20.0, -0.5),
        "face_open_deg": (0.0, 40.0),
        "shaft_lean_deg": (-10.0, 25.0),
        "entry_distance_behind_ball_m": (0.010, 0.200),
        "ball_depth_m": (-0.010, 0.020),
        "target_carry_m": (2.0, 40.0),
        "carry_tolerance_fraction": (0.02, 0.5),
    }
)
"""Accepted ``(low, high)`` bounds per input field, in the field's own unit.

The single source of truth for the PyQt panels and the v1 API route (#9545):
keys are the :class:`~src.tools.bunker_shot_gui.design.WedgeDesign`,
``SandCondition``, ``SwingSetup`` and ``SolverSetup`` field names, and every
``low < high``. Read-only, so no consumer can widen a bound for the other.
"""
