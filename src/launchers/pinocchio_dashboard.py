"""Pinocchio Dashboard Launcher.

Launches the Unified Dashboard with the Pinocchio Physics Engine.

The Pinocchio physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the Pinocchio dependency chain.
"""

from typing import Any, cast

from src.shared.python.dashboard.launcher import launch_dashboard
from src.shared.python.dashboard.window import ModelLoadStatus, UnifiedDashboardWindow
from src.shared.python.logging_pkg.logging_config import get_logger

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


class PinocchioDashboard(UnifiedDashboardWindow):
    """Unified Dashboard with Pinocchio Physics Engine."""

    def __init__(self, exercise_filter: str | None = None) -> None:
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = cast(Any, PinocchioPhysicsEngine)()
        title = "Pinocchio Golf Analysis Dashboard"
        model_status = ModelLoadStatus(engine_name="Pinocchio")
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
                        try:
                            engine.load_from_path(models[0])
                            model_status = ModelLoadStatus(
                                engine_name="Pinocchio", model_name=models[0]
                            )
                        except _EXPECTED_LOAD_ERRORS as exc:
                            logger.exception(
                                "Pinocchio failed to load model %s", models[0]
                            )
                            model_status = ModelLoadStatus(
                                engine_name="Pinocchio",
                                model_name=models[0],
                                loaded=False,
                                error=str(exc),
                            )
            except _EXPECTED_LOAD_ERRORS as exc:
                logger.exception(
                    "Pinocchio model discovery failed for exercise %r",
                    exercise_filter,
                )
                model_status = ModelLoadStatus(
                    engine_name="Pinocchio", loaded=False, error=str(exc)
                )

        super().__init__(engine, title=title, model_status=model_status)


def main() -> None:
    """Main entry point."""
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    launch_dashboard(
        engine_class=cast(Any, PinocchioPhysicsEngine),
        title="Pinocchio Golf Analysis Dashboard",
    )


if __name__ == "__main__":
    main()
