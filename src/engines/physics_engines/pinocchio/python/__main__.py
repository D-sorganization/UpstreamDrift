"""Entry point for Pinocchio Physics Engine dashboard."""

from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
from src.shared.python.dashboard.launcher import launch_dashboard

if __name__ == "__main__":
    launch_dashboard(PinocchioPhysicsEngine, title="Pinocchio Physics Engine")  # type: ignore[type-abstract]
