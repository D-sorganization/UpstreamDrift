"""Entry point for Pinocchio Physics Engine dashboard."""

import sys
from pathlib import Path

# Add suite root to sys.path
try:
    current_path = Path(__file__).resolve()
    suite_root: Path | None = None
    for parent in current_path.parents:
        if (parent / ".git").exists() or (parent / ".antigravityignore").exists():
            suite_root = parent
            break

    if suite_root and str(suite_root) not in sys.path:
except (FileNotFoundError, OSError):
    pass

from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
from src.shared.python.dashboard.launcher import launch_dashboard

if __name__ == "__main__":
    launch_dashboard(PinocchioPhysicsEngine, title="Pinocchio Physics Engine")  # type: ignore[type-abstract]
