"""
Example 01: Basic Simulation

This example demonstrates how to:
1. Initialize the Engine Manager
2. Load the MuJoCo engine
3. Run a basic simulation loop
4. Save results
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.python.engine_manager import EngineManager, EngineType  # noqa: E402
from shared.python.output_manager import OutputManager  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run basic simulation example."""
    logger.info("Starting Example 01: Basic Simulation")

    # 1. Initialize Managers
    engine_manager = EngineManager(project_root)
    output_manager = OutputManager(project_root / "output")
    output_manager.create_output_structure()

    # 2. Load Engine
    # Note: This checks for actual installation.
    # If not installed, we'll handle gracefully for this example.
    logger.info("Initializing MuJoCo engine...")
    if not engine_manager.switch_engine(EngineType.MUJOCO):
        logger.warning("MuJoCo not found. Please install it to run simulation.")
        return

    # 3. Simulation Loop (Conceptual)
    logger.info("Running simulation...")

    # In a real scenario, this would interface with engine_manager._mujoco_module
    # For this example, we generate synthetic data representing a ball trajectory
    duration = 3.0
    dt = 0.01
    steps = int(duration / dt)

    times = []
    heights = []
    velocities = []

    # Simple ballistic trajectory: z = v0*t - 0.5*g*t^2
    v0 = 20.0
    GRAVITY_M_S2 = 9.81

    for i in range(steps):
        t = i * dt
        z = v0 * t - 0.5 * GRAVITY_M_S2 * t**2
        v = v0 - GRAVITY_M_S2 * t

        if z < 0:
            z = 0
            v = 0

        times.append(t)
        heights.append(z)
        velocities.append(v)

    logger.info(f"Simulation complete. {steps} steps computed.")

    # 4. Save Results
    results = {"time": times, "height": heights, "velocity": velocities}

    metadata = {
        "example": "01_basic_simulation",
        "engine": "MuJoCo (Simulated)",
        "date": time.strftime("%Y-%m-%d"),
    }

    save_path = output_manager.save_simulation_results(
        results, "example_01_trajectory", engine="mujoco", metadata=metadata
    )

    logger.info(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
