"""Pinocchio Dashboard Launcher.

Launches the Unified Dashboard with the Pinocchio Physics Engine.

The Pinocchio engine package is deferred to inside ``main()`` so that
importing this module does NOT trigger pinocchio package loading.  This
satisfies the lazy-loading requirement from Guideline Issue #1956.
"""

from src.shared.python.dashboard.launcher import launch_dashboard


def main() -> None:
    """Main entry point."""
    # Deferred import: only loads pinocchio when this launcher is actually invoked
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: PLC0415
        PinocchioPhysicsEngine,
    )

    launch_dashboard(
        engine_class=PinocchioPhysicsEngine,  # type: ignore[type-abstract]
        title="Pinocchio Golf Analysis Dashboard",
    )


if __name__ == "__main__":
    main()
