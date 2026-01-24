"""Drake Dashboard Launcher.

Launches the Unified Dashboard with the Drake Physics Engine.
"""

import argparse
import sys

from PyQt6.QtWidgets import QApplication, QFileDialog
from src.shared.python.gui_utils import get_qapp


from src.engines.physics_engines.drake.python.drake_physics_engine import (
    DrakePhysicsEngine,
)
from src.shared.python.dashboard.launcher import launch_dashboard


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
        app = get_qapp()

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
