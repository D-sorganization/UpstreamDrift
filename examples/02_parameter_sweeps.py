"""
Example 02: Parameter Sweeps

This example demonstrates how to:
1. Access and modify physics parameters
2. Run a parameter sweep simulation
3. Export detailed analysis results
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.python.output_manager import OutputManager  # noqa: E402
from shared.python.physics_parameters import (  # noqa: E402
    ParameterCategory,
    get_registry,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run parameter sweep example."""
    logger.info("Starting Example 02: Parameter Sweeps")

    # 1. Access Registry
    registry = get_registry()
    output_manager = OutputManager(project_root / "output")
    output_manager.create_output_structure()

    # View available parameters
    logger.info("Available Ball Parameters:")
    ball_params = registry.get_by_category(ParameterCategory.BALL)
    for p in ball_params:
        logger.info(f" - {p.name}: {p.value} {p.unit}")

    # 2. Define Sweep
    # We will sweep Launch Velocity (not in registry, but simulating effect)
    # and use GRAVITY from registry

    gravity = registry.get("GRAVITY").value
    launch_angles = [30, 45, 60]
    velocities = np.linspace(20, 50, 4)  # 20, 30, 40, 50 m/s

    results = []

    logger.info("Running sweep...")
    for angle in launch_angles:
        theta = np.radians(angle)
        for v in velocities:
            # Analytical Range: R = (v^2 * sin(2*theta)) / g
            r_ideal = (v**2 * np.sin(2 * theta)) / gravity

            # Simulate Drag Effect (simple approximation)
            # Drag reduces range significantly
            drag_factor = 0.8  # approx
            r_actual = r_ideal * drag_factor

            results.append(
                {
                    "launch_angle_deg": angle,
                    "velocity_ms": v,
                    "range_ideal_m": round(r_ideal, 2),
                    "range_drag_m": round(r_actual, 2),
                }
            )

    # 3. Save Analysis
    df_results = output_manager.save_simulation_results(
        results, "example_02_sweep_results", engine="analytical"
    )

    # 4. Generate Report
    summary = {
        "total_simulations": len(results),
        "max_range": max(r["range_drag_m"] for r in results),
        "best_angle": max(results, key=lambda x: x["range_drag_m"])["launch_angle_deg"],
    }

    report_path = output_manager.export_analysis_report(
        summary, "example_02_report", format_type="json"
    )

    logger.info(f"Sweep complete. Results at {df_results}, Report at {report_path}")


if __name__ == "__main__":
    main()
