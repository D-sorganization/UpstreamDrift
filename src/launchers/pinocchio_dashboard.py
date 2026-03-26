"""Pinocchio Dashboard Launcher.

Launches the Unified Dashboard with the Pinocchio Physics Engine.

The Pinocchio physics engine import is deferred to ``main()`` to ensure strict
lazy-loading: importing this module does not trigger the Pinocchio dependency chain.
"""

from src.shared.python.dashboard.launcher import launch_dashboard


def main() -> None:
    """Main entry point."""
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: PLC0415
        PinocchioPhysicsEngine,
    )

    launch_dashboard(
        engine_class=PinocchioPhysicsEngine,  # type: ignore[type-abstract]
        title="Pinocchio Golf Analysis Dashboard",
    )


if __name__ == "__main__":
    main()
