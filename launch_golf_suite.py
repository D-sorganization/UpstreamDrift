#!/usr/bin/env python3
"""
Golf Modeling Suite - Unified Launcher

Usage:
    golf-suite              # Launch web UI (default, recommended)
    golf-suite --classic    # Launch classic PyQt6 launcher
    golf-suite --api-only   # Launch API server only (for development)
    golf-suite --engine X   # Launch specific engine directly
"""

import argparse
import importlib
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("golf_launcher")


def main():
    parser = argparse.ArgumentParser(
        description="Golf Modeling Suite - Biomechanical Golf Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    golf-suite                    Launch web UI (opens in browser)
    golf-suite --classic          Launch classic desktop UI
    golf-suite --api-only         Start API server without UI
    golf-suite --engine mujoco    Launch MuJoCo engine directly
        """,
    )

    parser.add_argument(
        "--classic",
        action="store_true",
        help="Use classic PyQt6 desktop launcher instead of web UI",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start API server only (no UI)",
    )
    parser.add_argument(
        "--engine",
        choices=[
            "mujoco",
            "drake",
            "pinocchio",
            "opensim",
            "myosim",
            "matlab_2d",
            "matlab_3d",
            "pendulum",
        ],
        help="Launch a specific engine directly",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for local server (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open browser",
    )

    args = parser.parse_args()

    if args.engine:
        # Direct engine launch (legacy support)
        launch_engine_directly(args.engine)
    elif args.classic:
        # Classic PyQt6 launcher
        try:
            # Try new location first
            from src.launchers.golf_launcher import main as classic_main

            classic_main()
        except ImportError:
            logger.error("Could not load classic launcher. Check installation.")
            sys.exit(1)
    elif args.api_only:
        # API server only
        os.environ["GOLF_NO_BROWSER"] = "true"
        os.environ["GOLF_PORT"] = str(args.port)
        from src.api.local_server import main as api_main

        api_main()
    else:
        # Default: Web UI (recommended)
        os.environ["GOLF_PORT"] = str(args.port)
        if args.no_browser:
            os.environ["GOLF_NO_BROWSER"] = "true"
        from src.api.local_server import main as server_main

        server_main()


def launch_engine_directly(engine: str):
    """Launch a specific engine GUI directly (legacy mode)."""
    engine_launchers = {
        "mujoco": "src.engines.physics_engines.mujoco.python.humanoid_launcher",
        "drake": "src.engines.physics_engines.drake.python.drake_gui_app",
        "pinocchio": "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui",
        "opensim": "src.engines.physics_engines.opensim.python.opensim_launcher",
        "myosim": "src.engines.physics_engines.myosim.python.myosim_launcher",
        "pendulum": "src.engines.pendulum_models.python.pendulum_launcher",
    }

    # Engines that don't support direct launch
    web_only_engines = {"matlab_2d", "matlab_3d"}

    if engine in web_only_engines:
        logger.info(
            "Engine '%s' requires the web UI. Launching web UI instead...", engine
        )
        os.environ["GOLF_DEFAULT_ENGINE"] = engine
        from src.api.local_server import main as server_main

        server_main()
        return

    if engine not in engine_launchers:
        logger.error("Direct launch not available for %s. Use web UI instead.", engine)
        sys.exit(1)

    try:
        module = importlib.import_module(engine_launchers[engine])
        if hasattr(module, "main"):
            module.main()
        else:
            logger.error("Module %s has no main() function.", engine_launchers[engine])
            sys.exit(1)
    except ImportError as e:
        logger.error("Failed to launch %s: %s", engine, e)
        logger.info("Try using 'golf-suite' without --engine to use the web UI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
