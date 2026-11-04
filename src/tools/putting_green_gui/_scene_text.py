"""Plain-text readouts of a putt for the putting tile (#14465, #14473).

The metrics, impact and optimum-read panels, in the chosen display units,
and the unit factors the tile's readouts share. Qt-free, so the widget
stays a thin renderer over them.
"""

from __future__ import annotations

import math

from src.shared.python.ui.units import UnitSystem, format_speed, from_display_distance
from src.tools.putting_green_gui._scene_builder import (
    PuttScene,
    PuttSolution,
    green_label,
)

M_TO_FT = 3.280839895
M_TO_CM = 100.0
MS_TO_MPH = 2.236936
M_TO_IN = 12.0 / from_display_distance(1.0, UnitSystem.IMPERIAL, unit="ft")
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


def format_read(solution: PuttSolution, unit_system: UnitSystem) -> str:
    """One line naming the solved read, in the chosen units."""
    if unit_system == UnitSystem.METRIC:
        speed = f"{solution.putter_speed_ms:.2f} m/s"
        lean = f"{abs(solution.break_m) * M_TO_CM:.1f} cm"
    else:
        speed = f"{solution.putter_speed_ms * MS_TO_MPH:.1f} mph"
        lean = f"{abs(solution.break_m) * M_TO_IN:.1f} in"
    side = "left" if solution.break_m >= 0.0 else "right"
    found = "" if solution.holed else " (no holing line found)"
    return (
        f"Optimum read:  {speed} at {solution.aim_deg:+.1f}°, "
        f"aim {lean} {side} of the cup{found}\n\n"
    )


def format_impact(scene: PuttScene, unit_system: UnitSystem) -> str:
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
        strike = f"{abs(toe) * M_TO_IN:.2f} in {side}"
    skid_m = scene.skid_distance_m or 0.0
    skid = f"{skid_m * M_TO_CM:.1f} cm" if metric else f"{skid_m * M_TO_IN:.1f} in"
    twist = math.degrees(impact.face_twist_rad_s)
    rpm = impact.spin_rad_s * RAD_S_TO_RPM
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


def format_metrics(
    scene: PuttScene, unit_system: UnitSystem = UnitSystem.METRIC
) -> str:
    """Render a human-readable metrics panel for a finished putt."""
    outcome = "HOLED" if scene.holed else "Missed"
    dist_cm = scene.final_distance_to_cup_m * 100.0

    if unit_system == UnitSystem.METRIC:
        miss_line = (
            "Result:        HOLED"
            if scene.holed
            else f"Result:        Missed by {dist_cm:.1f} cm"
        )
        roll_line = (
            f"Total roll:    {scene.total_roll_m:.2f} m "
            f"({scene.total_roll_m * M_TO_FT:.1f} ft)\n"
        )
        break_line = (
            f"Peak break:    {scene.peak_break_m * M_TO_CM:.1f} cm "
            f"({scene.peak_break_m * M_TO_IN:.2f} in)\n"
        )
        speed_line = (
            f"Launch speed:  {scene.launch_speed_ms:.2f} m/s "
            f"({scene.launch_speed_ms * MS_TO_MPH:.1f} mph)\n"
        )
    else:
        dist_in = dist_cm / 2.54
        miss_line = (
            "Result:        HOLED"
            if scene.holed
            else f"Result:        Missed by {dist_in:.1f} in ({dist_cm:.1f} cm)"
        )
        roll_line = (
            f"Total roll:    {scene.total_roll_m * M_TO_FT:.1f} ft "
            f"({scene.total_roll_m:.2f} m)\n"
        )
        break_line = (
            f"Peak break:    {scene.peak_break_m * M_TO_IN:.2f} in "
            f"({scene.peak_break_m * M_TO_CM:.1f} cm)\n"
        )
        speed_line = (
            f"Launch speed:  {scene.launch_speed_ms * MS_TO_MPH:.1f} mph "
            f"({scene.launch_speed_ms:.2f} m/s)\n"
        )

    return (
        f"Putting Simulation - {outcome}\n"
        f"{'=' * 44}\n"
        f"{miss_line}\n"
        f"{roll_line}"
        f"Roll time:     {scene.duration_s:.2f} s\n"
        f"{break_line}"
        f"{speed_line}"
        f"Green:         {green_label(scene.green_type)}\n"
        f"Roll model:    {scene.roll_model}\n"
        f"Track colour:  amber = skidding, green = pure roll, grey = stopped\n"
        f"{format_impact(scene, unit_system)}"
    )
