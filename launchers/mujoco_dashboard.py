"""MuJoCo Dashboard Launcher (Unified).

Launches the Unified Dashboard with the MuJoCo Physics Engine.
This serves as an alternative to the specialized AdvancedGolfAnalysisWindow.
"""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)
from shared.python.dashboard.window import UnifiedDashboardWindow


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    # Initialize MuJoCo Engine
    engine = MuJoCoPhysicsEngine()

    # Load model if possible
    # engine.load_from_path(...)

    window = UnifiedDashboardWindow(
        engine, title="MuJoCo Golf Analysis Dashboard (Unified)"
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
