"""Pinocchio Dashboard Launcher.

Launches the Unified Dashboard with the Pinocchio Physics Engine.
"""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
from shared.python.dashboard.window import UnifiedDashboardWindow


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    # Initialize Pinocchio Engine
    engine = PinocchioPhysicsEngine()

    # Try to load a model
    # engine.load_from_path("path/to/model.urdf")

    window = UnifiedDashboardWindow(engine, title="Pinocchio Golf Analysis Dashboard")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
