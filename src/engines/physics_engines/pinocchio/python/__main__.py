"""Entry point for Pinocchio Physics Engine dashboard."""

import sys
from pathlib import Path

# Bootstrap: add repo root to sys.path for src.* imports
_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(__file__)

from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: E402
    PinocchioPhysicsEngine,
)
from src.shared.python.dashboard.launcher import launch_dashboard  # noqa: E402

if __name__ == "__main__":
    launch_dashboard(PinocchioPhysicsEngine, title="Pinocchio Physics Engine")  # type: ignore[type-abstract]
