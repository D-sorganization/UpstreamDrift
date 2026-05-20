from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engines.physics_engines.putting_green.python._sim_config import (
    SimulationResult,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    StrokeParameters,
)

if TYPE_CHECKING:
    from src.engines.physics_engines.putting_green.python._sim_core import (
        PuttingGreenSimulator,
    )


def simulate_with_feedback(
    sim: PuttingGreenSimulator,
    stroke_params: StrokeParameters,
) -> dict[str, Any]:
    """Simulate putt with practice feedback.

    Args:
        sim: The simulator instance
        stroke_params: Stroke parameters

    Returns:
        Dictionary with result and feedback
    """
    if stroke_params is None:
        raise ValueError("stroke_params must be provided")
    result = sim.simulate_putt(stroke_params)

    arr = np.asarray(
        result.final_position - sim.green.hole_position, dtype=float
    ).reshape(-1)
    distance_from_hole = 0.0 if arr.size == 0 else math.hypot(*arr)

    feedback: dict[str, Any] = {
        "distance_from_hole": distance_from_hole,
        "holed": result.holed,
        "total_distance": result.total_distance,
    }

    if result.holed:
        feedback["suggested_adjustment"] = "Great putt"
    else:
        if distance_from_hole < 0.5:
            if result.final_position[0] < sim.green.hole_position[0]:
                feedback["suggested_adjustment"] = "Hit slightly firmer"
            else:
                feedback["suggested_adjustment"] = "Hit slightly softer"
        else:
            feedback["suggested_adjustment"] = "Check your aim line"

    return feedback


def simulate_scatter(
    sim: PuttingGreenSimulator,
    start_position: np.ndarray,
    stroke_params: StrokeParameters,
    n_simulations: int = 10,
    speed_variance: float = 0.1,
    direction_variance_deg: float = 2.0,
    rng: np.random.Generator | None = None,
) -> list[SimulationResult]:
    """Simulate multiple putts with variance for scatter analysis.

    Args:
        sim: The simulator instance
        start_position: Starting ball position
        stroke_params: Base stroke parameters
        n_simulations: Number of simulations
        speed_variance: Standard deviation of speed [m/s]
        direction_variance_deg: Standard deviation of direction [degrees]
        rng: Optional random generator (defaults to simulator RNG)

    Returns:
        List of simulation results
    """
    if start_position is None:
        raise ValueError("start_position must be provided")
    results = []
    rng = rng or sim._rng

    for _ in range(n_simulations):
        speed = stroke_params.speed + rng.normal(0, speed_variance)
        speed = max(0.1, speed)

        angle_var = rng.normal(0, direction_variance_deg * np.pi / 180)
        cos_a, sin_a = np.cos(angle_var), np.sin(angle_var)
        direction = np.array(
            [
                cos_a * stroke_params.direction[0] - sin_a * stroke_params.direction[1],
                sin_a * stroke_params.direction[0] + cos_a * stroke_params.direction[1],
            ]
        )

        varied_params = StrokeParameters(
            speed=speed,
            direction=direction,
            face_angle=stroke_params.face_angle + rng.normal(0, 1.0),
            attack_angle=stroke_params.attack_angle,
        )

        result = sim.simulate_putt(varied_params, ball_position=start_position)
        results.append(result)

    return results


def compute_aim_line(
    sim: PuttingGreenSimulator,
    ball_position: np.ndarray,
) -> dict[str, Any]:
    """Compute aim line accounting for break.

    Args:
        sim: The simulator instance
        ball_position: Current ball position

    Returns:
        Dictionary with aim information
    """
    if ball_position is None:
        raise ValueError("ball_position must be provided")
    target = sim.green.hole_position

    break_info = sim.green.calculate_break(ball_position, target)

    aim_point = target - break_info["break_direction"] * break_info["total_break"]

    arr = np.asarray(target - ball_position, dtype=float).reshape(-1)
    distance = float(0.0 if arr.size == 0 else math.hypot(*arr))
    avg_slope = np.dot(
        break_info["average_slope"], (target - ball_position) / (distance + 1e-10)
    )
    recommended_speed = sim.putter.estimate_required_speed(
        distance, sim.green.turf.stimp_rating, slope_percent=avg_slope * 100
    )

    return {
        "aim_point": aim_point,
        "break": break_info["total_break"],
        "break_direction": break_info["break_direction"],
        "recommended_speed": recommended_speed,
        "distance": distance,
    }


def read_green(
    sim: PuttingGreenSimulator,
    ball_position: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Read green between ball and target.

    Args:
        sim: The simulator instance
        ball_position: Ball position
        target: Target position

    Returns:
        Green reading with slopes and recommendations
    """
    if ball_position is None:
        raise ValueError("ball_position must be provided")
    reading = sim.green.read_putt_line(ball_position, target)
    break_info = sim.green.calculate_break(ball_position, target)
    aim_info = compute_aim_line(sim, ball_position)

    return {
        "positions": reading["positions"],
        "elevations": reading["elevations"],
        "slopes": reading["slopes"],
        "distance": reading["distance"],
        "total_break": break_info["total_break"],
        "recommended_speed": aim_info["recommended_speed"],
        "aim_point": aim_info["aim_point"],
    }


def export_result(result: SimulationResult, path: str) -> None:
    """Export simulation result to file.

    Args:
        result: Simulation result
        path: Output file path
    """
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
