#!/usr/bin/env python3
"""
Golf Modeling Suite - Unified Launcher

Usage:
    upstream-drift              # Launch web UI (default, recommended)
    upstream-drift --classic    # Launch classic PyQt6 launcher
    upstream-drift --api-only   # Launch API server only (for development)
    upstream-drift --engine X   # Launch specific engine directly
"""

import argparse
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
    upstream-drift                    Launch web UI (opens in browser)
    upstream-drift --classic          Launch classic desktop UI
    upstream-drift --api-only         Start API server without UI
    upstream-drift --engine mujoco    Launch MuJoCo engine directly
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

    try:
        from src.shared.python.engine_core.engine_manager import EngineType

        engine_choices = [e.value for e in EngineType]
    except ImportError:
        engine_choices = ["mujoco", "drake", "pinocchio", "opensim", "myosim"]
        engine_choices.extend(["matlab_2d", "matlab_3d", "pendulum"])

    parser.add_argument(
        "--engine",
        choices=engine_choices,
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
        try:
            from src.shared.python.launcher_factory import launch_engine_directly
        except ImportError:
            # Fallback if PYTHONPATH is not set correctly
            sys.path.append(os.getcwd())
            from src.shared.python.launcher_factory import launch_engine_directly

        # Check if engine is web-only
        web_only_engines = {"matlab_2d", "matlab_3d"}
        if args.engine in web_only_engines:
            logger.info(
                "Engine '%s' requires the web UI. Launching web UI instead...",
                args.engine,
            )
            os.environ["GOLF_DEFAULT_ENGINE"] = args.engine
            from src.api.local_server import main as server_main

            server_main()
            return

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


if __name__ == "__main__":
    main()
