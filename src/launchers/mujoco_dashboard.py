"""MuJoCo Dashboard Launcher (Unified).

Launches the Unified Dashboard with the MuJoCo Physics Engine.
This serves as an alternative to the specialized AdvancedGolfAnalysisWindow.

The MuJoCo physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the MuJoCo dependency chain.
"""

from typing import Any, cast

from src.shared.python.dashboard.launcher import launch_dashboard
from src.shared.python.dashboard.window import ModelLoadStatus, UnifiedDashboardWindow
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Expected failure modes for model discovery/loading at this boundary:
# missing sibling repos or bad env config (OSError incl. FileNotFoundError),
# invalid/malformed model files (ValueError, RuntimeError from the native
# MuJoCo bindings), and misconfigured provider lookups (KeyError,
# AttributeError). Narrower than bare ``except Exception`` per ADR-0016 /
# narrow_catch while still covering the realistic failure surface of this
# load boundary (issue #8829).
_EXPECTED_LOAD_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    RuntimeError,
    KeyError,
    AttributeError,
    TypeError,
)


class MuJoCoDashboard(UnifiedDashboardWindow):
    """Unified Dashboard with MuJoCo Physics Engine."""

    def __init__(self, exercise_filter: str | None = None) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = cast(Any, MuJoCoPhysicsEngine)()
        title = "MuJoCo Golf Analysis Dashboard (Unified)"
        model_status = ModelLoadStatus(engine_name="MuJoCo")
        if exercise_filter:
            title += f" - {exercise_filter.title()}"

            try:
                from src.shared.python.config.model_source_providers import (
                    mujoco_models_source,
                )

                root = mujoco_models_source()
                exercise_dir = root / "exercises" / exercise_filter
                if exercise_dir.exists():
                    import glob

                    models = glob.glob(str(exercise_dir / "*.xml"))
                    if models:
                        try:
                            engine.load_from_path(models[0])
                            model_status = ModelLoadStatus(
                                engine_name="MuJoCo", model_name=models[0]
                            )
                        except _EXPECTED_LOAD_ERRORS as exc:
                            logger.exception(
                                "MuJoCo failed to load model %s", models[0]
                            )
                            model_status = ModelLoadStatus(
                                engine_name="MuJoCo",
                                model_name=models[0],
                                loaded=False,
                                error=str(exc),
                            )
            except _EXPECTED_LOAD_ERRORS as exc:
                logger.exception(
                    "MuJoCo model discovery failed for exercise %r", exercise_filter
                )
                model_status = ModelLoadStatus(
                    engine_name="MuJoCo", loaded=False, error=str(exc)
                )

        super().__init__(engine, title=title, model_status=model_status)


def main() -> None:
    """Main entry point."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    launch_dashboard(
        engine_class=cast(Any, MuJoCoPhysicsEngine),
        title="MuJoCo Golf Analysis Dashboard (Unified)",
    )


if __name__ == "__main__":
    main()
