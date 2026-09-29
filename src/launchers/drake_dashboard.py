"""Drake Dashboard Launcher.

Launches the Unified Dashboard with the Drake Physics Engine.

The Drake physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the Drake/pydrake dependency chain.
"""

import argparse

from src.shared.python.dashboard.window import ModelLoadStatus, UnifiedDashboardWindow
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.ui.qt.utils import get_qapp

logger = get_logger(__name__)

# See mujoco_dashboard._EXPECTED_LOAD_ERRORS for rationale: narrower than
# bare ``except Exception`` per ADR-0016 / narrow_catch while still covering
# the realistic failure surface of this load boundary (issue #8829).
_EXPECTED_LOAD_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    RuntimeError,
    KeyError,
    AttributeError,
    TypeError,
)


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
        model_status = ModelLoadStatus(engine_name="Drake")
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
                except _EXPECTED_LOAD_ERRORS as exc:
                    logger.exception(
                        "Drake model discovery failed for exercise %r",
                        exercise_filter,
                    )
                    model_status = ModelLoadStatus(
                        engine_name="Drake", loaded=False, error=str(exc)
                    )

        if model_path:
            try:
                engine.load_from_path(model_path)
                model_status = ModelLoadStatus(
                    engine_name="Drake", model_name=model_path
                )
            except _EXPECTED_LOAD_ERRORS as exc:
                logger.exception("Drake failed to load model %s", model_path)
                model_status = ModelLoadStatus(
                    engine_name="Drake",
                    model_name=model_path,
                    loaded=False,
                    error=str(exc),
                )

        super().__init__(engine, title=title, model_status=model_status)


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
