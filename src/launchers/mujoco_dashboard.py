"""MuJoCo Dashboard Launcher (Unified).

Launches the Unified Dashboard with the MuJoCo Physics Engine.
This serves as an alternative to the specialized AdvancedGolfAnalysisWindow.
"""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)
from src.shared.python.dashboard.launcher import launch_dashboard


def main() -> None:
    """Main entry point."""
    launch_dashboard(
        engine_class=MuJoCoPhysicsEngine,
        title="MuJoCo Golf Analysis Dashboard (Unified)",
    )


if __name__ == "__main__":
    main()
