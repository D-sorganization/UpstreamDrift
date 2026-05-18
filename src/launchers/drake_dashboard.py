"""Drake Dashboard Launcher.

Launches the Unified Dashboard with the Drake Physics Engine.

The Drake physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the Drake/pydrake dependency chain.
"""

import argparse

from PyQt6.QtWidgets import QFileDialog

from src.shared.python.dashboard.launcher import launch_dashboard
from src.shared.python.dashboard.window import UnifiedDashboardWindow
from src.shared.python.ui.qt.utils import get_qapp
import contextlib


class DrakeDashboard(UnifiedDashboardWindow):
    """Unified Dashboard with Drake Physics Engine."""

    def __init__(
        self, exercise_filter: str | None = None, model_path: str | None = None
    ) -> None:
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        title = "Drake Golf Analysis Dashboard"
        if exercise_filter:
            title += f" - {exercise_filter.title()}"
            if not model_path:
                try:
                    from src.shared.python.config.model_source_providers import (
                        drake_models_source,
                    )

                    root = drake_models_source()
                    exercise_dir = root / "exercises" / exercise_filter
                    if exercise_dir.exists():
                        import glob

                        models = glob.glob(str(exercise_dir / "*.urdf")) + glob.glob(
                            str(exercise_dir / "*.sdf")
                        )
                        if models:
                            model_path = models[0]
                except Exception:
                    pass

        if model_path:
            with contextlib.suppress(Exception):
                engine.load_from_path(model_path)

        super().__init__(engine, title=title)


def main() -> None:
    """Main entry point."""
    import sys
    from src.shared.python.logging_pkg.logging_config import configure_gui_logging

    parser = argparse.ArgumentParser(description="Drake Golf Analysis Dashboard")
    parser.add_argument(
        "--model", type=str, help="Path to model file (URDF/SDF)", default=None
    )
    args = parser.parse_args()

    model_path = args.model

    if not model_path:
        # Ensure QApplication exists for QFileDialog
        app = get_qapp()
        from PyQt6.QtWidgets import QFileDialog

        dialog = QFileDialog()
        dialog.setNameFilter("Model Files (*.urdf *.sdf *.xml)")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                model_path = selected[0]
    else:
        app = get_qapp()

    configure_gui_logging()
    window = DrakeDashboard(model_path=model_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
