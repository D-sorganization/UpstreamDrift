"""Entry point for Drake Physics Engine dashboard."""

from src.engines.physics_engines.drake.python.drake_physics_engine import (
    DrakePhysicsEngine,
)
from src.shared.python.dashboard.launcher import launch_dashboard

if __name__ == "__main__":
    launch_dashboard(DrakePhysicsEngine, title="Drake Physics Engine")  # type: ignore[type-abstract]
