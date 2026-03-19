"""MuJoCo Dashboard Launcher (Unified).

Launches the Unified Dashboard with the MuJoCo Physics Engine.
This serves as an alternative to the specialized AdvancedGolfAnalysisWindow.

The MuJoCo engine package (``import mujoco``) is intentionally deferred to
inside ``main()`` so that importing this module at the Python level does NOT
trigger a heavy MuJoCo DLL load.  This satisfies the lazy-loading requirement
from Guideline Issue #1956.
"""

from src.shared.python.dashboard.launcher import launch_dashboard


def main() -> None:
    """Main entry point."""
    # Deferred import: only loads mujoco when this launcher is actually invoked
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: PLC0415
        MuJoCoPhysicsEngine,
    )

    launch_dashboard(
        engine_class=MuJoCoPhysicsEngine,  # type: ignore[type-abstract]
        title="MuJoCo Golf Analysis Dashboard (Unified)",
    )


if __name__ == "__main__":
    main()
