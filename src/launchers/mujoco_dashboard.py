"""MuJoCo Dashboard Launcher (Unified).

Launches the Unified Dashboard with the MuJoCo Physics Engine.
This serves as an alternative to the specialized AdvancedGolfAnalysisWindow.

The MuJoCo physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the MuJoCo dependency chain.
"""

from src.shared.python.dashboard.launcher import launch_dashboard
from src.shared.python.dashboard.window import UnifiedDashboardWindow


class MuJoCoDashboard(UnifiedDashboardWindow):
    """Unified Dashboard with MuJoCo Physics Engine."""

    def __init__(self, exercise_filter: str | None = None) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()
        title = "MuJoCo Golf Analysis Dashboard (Unified)"
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
                        engine.load_from_path(models[0])
            except Exception:
                pass

        super().__init__(engine, title=title)


def main() -> None:
    """Main entry point."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    launch_dashboard(
        engine_class=MuJoCoPhysicsEngine,
        title="MuJoCo Golf Analysis Dashboard (Unified)",
    )


if __name__ == "__main__":
    main()
