"""Drake Dashboard Launcher.

Launches the Unified Dashboard with the Drake Physics Engine.
"""

import argparse
import logging
import sys

from PyQt6.QtWidgets import QApplication, QFileDialog

from engines.physics_engines.drake.python.drake_physics_engine import DrakePhysicsEngine
from shared.python.dashboard.window import UnifiedDashboardWindow


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Drake Golf Analysis Dashboard")
    parser.add_argument(
        "--model", type=str, help="Path to model file (URDF/SDF)", default=None
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Initialize Drake Engine
    engine = DrakePhysicsEngine()

    model_path = args.model
    if not model_path:
        # Prompt user for model
        # Using a QFileDialog before main window is shown is fine for setup
        dialog = QFileDialog()
        dialog.setNameFilter("Model Files (*.urdf *.sdf *.xml)")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                model_path = selected[0]

    if model_path:
        try:
            logging.info(f"Loading model: {model_path}")
            engine.load_from_path(model_path)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            # Continue with empty engine, but warn
    else:
        logging.warning("No model loaded. Dashboard started with empty engine.")

    window = UnifiedDashboardWindow(engine, title="Drake Golf Analysis Dashboard")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
