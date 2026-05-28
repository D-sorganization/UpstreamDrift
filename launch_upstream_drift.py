#!/usr/bin/env python3
"""
Upstream Drift - Unified Launcher

Usage:
    upstream-drift              # Launch web UI (default, recommended)
    upstream-drift --classic    # Launch classic PyQt6 launcher
    upstream-drift --api-only   # Launch API server only (for development)
    upstream-drift --engine X   # Launch specific engine directly
"""

import sys
import os

os.environ.setdefault("GOLF_SUITE_MODE", "local")


if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "UpstreamDrift.Launcher.1"
        )
    except (AttributeError, OSError, NameError, ImportError):
        pass

import argparse
import logging
import os
from os import environ, getcwd
from pathlib import Path
from sys import exit, path

# Bootstrap import paths before any project imports. The repo root must be on
# sys.path so `src.*` resolves; the vendored Tools tree must be on sys.path so
# the shared Sidekick package (`upstream_drift_tools.ui.tools_sidebar`) can be
# imported at runtime — pytest reads this from pyproject.toml, but a direct
# `python launch_upstream_drift.py` invocation does not.
_REPO_ROOT = Path(__file__).resolve().parent

# Discover sibling Tools repository to prioritize real subtabs over stubs
_TOOLS_ROOT = None
_env_tools = os.environ.get("TOOLS_REPO_PATH")
if _env_tools and Path(_env_tools).is_dir():
    _TOOLS_ROOT = Path(_env_tools)
else:
    _vendor_resolved: Path | None = None
    try:
        _vendor_resolved = (_REPO_ROOT / "vendor" / "ud-tools").resolve()
    except Exception:  # noqa: BLE001
        _vendor_resolved = None
    _p_walk = _REPO_ROOT
    for _ in range(10):
        _p_walk = _p_walk.parent
        for _candidate in (
            _p_walk / "Tools",
            _p_walk / "Repositories" / "Tools",
            Path.home() / "Repositories" / "Tools",
        ):
            if _candidate.is_dir() and (_candidate / "src").is_dir():
                try:
                    # Skip candidate if it is nested inside our repo (e.g. the vendored copy)
                    # to prioritize a true sibling checkout.
                    if _candidate.is_relative_to(_REPO_ROOT):
                        continue
                except (ValueError, AttributeError):
                    if str(_REPO_ROOT) in str(_candidate.resolve()):
                        continue
                except Exception:  # noqa: BLE001
                    pass
                _TOOLS_ROOT = _candidate
                break
        if _TOOLS_ROOT:
            break

_paths_to_add = []
if _TOOLS_ROOT:
    _paths_to_add.extend(
        [str(_TOOLS_ROOT / "src"), str(_TOOLS_ROOT / "src" / "shared" / "python")]
    )

_paths_to_add.extend([str(_REPO_ROOT / "src"), str(_REPO_ROOT)])

_VENDOR_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
_paths_to_add.append(str(_VENDOR_SHARED))

# Insert paths to the front of sys.path in reverse order so the first item in _paths_to_add
# ends up at the very beginning of sys.path.
for _path_entry in reversed(_paths_to_add):
    if _path_entry not in path:
        path.insert(0, _path_entry)

from src.api._version import warn_if_unsupported_platform  # noqa: E402

path.append(os.path.join(getcwd(), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("upstream_drift_launcher")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upstream Drift - Biomechanical Golf Simulation",
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

    return parser.parse_args()


def route_launch(args: argparse.Namespace) -> None:
    """Route the launch based on parsed arguments."""
    import warnings

    # Protection: Error out if deprecated sidekick/tools imports are used
    warnings.filterwarnings(
        "error",
        category=DeprecationWarning,
        module=r".*(sidekick|upstream_drift_tools).*",
    )
    if args is None:
        raise ValueError("Parsed arguments must be provided")
    if not isinstance(args, argparse.Namespace):
        raise ValueError("args must be a Namespace object")

    engine_arg = getattr(args, "engine", None)
    classic_arg = getattr(args, "classic", False)
    api_only_arg = getattr(args, "api_only", False)
    port_arg = getattr(args, "port", 8000)
    no_browser_arg = getattr(args, "no_browser", False)

    if engine_arg:
        # Direct engine launch (legacy support)
        try:
            from src.shared.python.launcher_factory import launch_engine_directly
        except ImportError:
            # Fallback if PYTHONPATH is not set correctly
            path.append(getcwd())
            from src.shared.python.launcher_factory import launch_engine_directly

        # Check if engine is web-only
        web_only_engines = {"matlab_2d", "matlab_3d"}
        if engine_arg in web_only_engines:
            logger.info(
                "Engine '%s' requires the web UI. Launching web UI instead...",
                engine_arg,
            )
            environ["GOLF_DEFAULT_ENGINE"] = str(engine_arg)
            from src.api.local_server import main as server_main

            server_main()
            return

        launch_engine_directly(engine_arg)

    elif classic_arg:
        # Classic PyQt6 launcher
        try:
            # Try new location first
            from src.launchers.upstream_drift_launcher import main as classic_main

            classic_main()
        except ImportError as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Could not load classic launcher: {e}")
            exit(1)

    elif api_only_arg:
        # API server only
        environ["GOLF_NO_BROWSER"] = "true"
        environ["GOLF_PORT"] = str(port_arg)
        from src.api.local_server import main as api_main

        api_main()

    else:
        # Default: Web UI (recommended)
        environ["GOLF_PORT"] = str(port_arg)
        if no_browser_arg:
            environ["GOLF_NO_BROWSER"] = "true"
        from src.api.local_server import main as server_main

        server_main()


def main() -> None:
    """Main entry point for unified launcher."""
    warn_if_unsupported_platform()
    args = parse_arguments()
    route_launch(args)


if __name__ == "__main__":
    main()
