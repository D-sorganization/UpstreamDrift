"""Pinocchio Dashboard Launcher.

Launches the Unified Dashboard with the Pinocchio Physics Engine.
"""

from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
from shared.python.dashboard.launcher import launch_dashboard


def main() -> None:
    """Main entry point."""
    launch_dashboard(
        engine_class=PinocchioPhysicsEngine, title="Pinocchio Golf Analysis Dashboard"
    )


if __name__ == "__main__":
    main()
