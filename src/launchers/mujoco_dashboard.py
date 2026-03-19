"""MuJoCo Dashboard Launcher (Unified).

Launches the Unified Dashboard with the MuJoCo Physics Engine.
This serves as an alternative to the specialized AdvancedGolfAnalysisWindow.

The MuJoCo physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the MuJoCo dependency chain.
"""

from src.shared.python.dashboard.launcher import launch_dashboard


def main() -> None:
    """Main entry point."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: PLC0415
        MuJoCoPhysicsEngine,
    )

    launch_dashboard(
        engine_class=MuJoCoPhysicsEngine,  # type: ignore[type-abstract]
        title="MuJoCo Golf Analysis Dashboard (Unified)",
    )


if __name__ == "__main__":
    main()
