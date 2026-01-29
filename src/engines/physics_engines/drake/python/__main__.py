"""Entry point for Drake Physics Engine dashboard."""

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
        sys.path.insert(0, str(suite_root))
except Exception:
    pass

from src.engines.physics_engines.drake.python.drake_physics_engine import (
    DrakePhysicsEngine,
)
from src.shared.python.dashboard.launcher import launch_dashboard

if __name__ == "__main__":
    launch_dashboard(DrakePhysicsEngine, title="Drake Physics Engine")
