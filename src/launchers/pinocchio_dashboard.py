"""Pinocchio Dashboard Launcher.

Launches the Unified Dashboard with the Pinocchio Physics Engine.

The Pinocchio physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the Pinocchio dependency chain.
"""

from src.shared.python.dashboard.launcher import launch_dashboard
from src.shared.python.dashboard.window import UnifiedDashboardWindow


class PinocchioDashboard(UnifiedDashboardWindow):
    """Unified Dashboard with Pinocchio Physics Engine."""

    def __init__(self, exercise_filter: str | None = None) -> None:
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        title = "Pinocchio Golf Analysis Dashboard"
        if exercise_filter:
            title += f" - {exercise_filter.title()}"
            try:
                from src.shared.python.config.model_source_providers import (
                    pinocchio_models_source,
                )

                root = pinocchio_models_source()
                exercise_dir = root / "exercises" / exercise_filter
                if exercise_dir.exists():
                    import glob

                    models = glob.glob(str(exercise_dir / "*.urdf"))
                    if models:
                        engine.load_from_path(models[0])
            except Exception:
                pass

        super().__init__(engine, title=title)


def main() -> None:
    """Main entry point."""
    from src.shared.python.logging_pkg.logging_config import configure_gui_logging
    from src.shared.python.ui.qt.utils import get_qapp
    import sys

    configure_gui_logging()
    app = get_qapp()
    window = PinocchioDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
