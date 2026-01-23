"""Drake Dashboard Launcher.

Launches the Unified Dashboard with the Drake Physics Engine.
"""

import argparse
import sys

from PyQt6.QtWidgets import QApplication, QFileDialog
from shared.python.dashboard.launcher import launch_dashboard

from engines.physics_engines.drake.python.drake_physics_engine import DrakePhysicsEngine


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Drake Golf Analysis Dashboard")
    parser.add_argument(
        "--model", type=str, help="Path to model file (URDF/SDF)", default=None
    )
    args = parser.parse_args()

    model_path = args.model

    if not model_path:
        # Ensure QApplication exists for QFileDialog
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        dialog = QFileDialog()
        dialog.setNameFilter("Model Files (*.urdf *.sdf *.xml)")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                model_path = selected[0]

    launch_dashboard(
        engine_class=DrakePhysicsEngine,
        title="Drake Golf Analysis Dashboard",
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
