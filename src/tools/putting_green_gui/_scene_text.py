"""Plain-text readouts of a putt for the putting tile (#14465).

The impact and optimum-read panels, in the chosen display units, and the
unit factors the tile's readouts share. Qt-free, so the widget stays thin.
"""

from __future__ import annotations

import math

from src.shared.python.ui.units import UnitSystem, format_speed, from_display_distance
from src.tools.putting_green_gui._scene_builder import PuttScene, PuttSolution

_M_TO_FT = 3.280839895
_M_TO_CM = 100.0
_MS_TO_MPH = 2.236936
_M_TO_IN = 12.0 / from_display_distance(1.0, UnitSystem.IMPERIAL, unit="ft")
_RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


def _format_read(solution: PuttSolution, unit_system: UnitSystem) -> str:
    """One line naming the solved read, in the chosen units."""
    if unit_system == UnitSystem.METRIC:
        speed = f"{solution.putter_speed_ms:.2f} m/s"
        lean = f"{abs(solution.break_m) * _M_TO_CM:.1f} cm"
    else:
        speed = f"{solution.putter_speed_ms * _MS_TO_MPH:.1f} mph"
        lean = f"{abs(solution.break_m) * _M_TO_IN:.1f} in"
    side = "left" if solution.break_m >= 0.0 else "right"
    found = "" if solution.holed else " (no holing line found)"
    return (
        f"Optimum read:  {speed} at {solution.aim_deg:+.1f}°, "
        f"aim {lean} {side} of the cup{found}\n\n"
    )


def _format_impact(scene: PuttScene, unit_system: UnitSystem) -> str:
    """The putter-ball impact block (#14461); empty for scenes without one."""
    impact = scene.impact
    if impact is None:
        return ""
    metric = unit_system == UnitSystem.METRIC
    toe = scene.impact_toe_m
    side = "toe" if toe > 0 else "heel"
    if toe == 0.0:
        strike = "centred"
    elif metric:
        strike = f"{abs(toe) * 1000:.1f} mm {side}"
    else:
        strike = f"{abs(toe) * _M_TO_IN:.2f} in {side}"
    skid_m = scene.skid_distance_m or 0.0
    skid = f"{skid_m * _M_TO_CM:.1f} cm" if metric else f"{skid_m * _M_TO_IN:.1f} in"
    twist = math.degrees(impact.face_twist_rad_s)
    rpm = impact.spin_rad_s * _RAD_S_TO_RPM
    face = "opens" if twist > 0 else "closes" if twist < 0 else "square"
    return (
        f"\nImpact\n{'-' * 44}\n"
        f"Strike:        {strike}\n"
        f"Launch angle:  {impact.launch_angle_deg:.1f}°\n"
        f"Spin:          {abs(rpm):.0f} rpm {'topspin' if rpm >= 0 else 'backspin'}\n"
        f"Skid:          {skid}\n"
        f"Face twist:    {twist:+.1f} °/s (face {face})\n"
        f"Energy loss:   {impact.kinetic_energy_loss_j * 1000:.1f} mJ\n"
        f"Putter slows:  {format_speed(impact.putter_dv_mps, unit_system, decimals=2)}\n"
    )
