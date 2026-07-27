#!/usr/bin/env python3
"""
Upstream Drift - Unified Launcher

Usage:
    upstream-drift              # Launch web UI (default, recommended)
    upstream-drift --classic    # Launch classic PyQt6 launcher
    upstream-drift --api-only   # Launch API server only (for development)
    upstream-drift --engine X   # Launch specific engine directly
"""

import os
import sys

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
import importlib
import logging
import os
from os import environ, getcwd
from pathlib import Path
from sys import exit, path
from types import ModuleType

# Bootstrap canonical parent paths before importing the Upstream-owned
# resolver below. The resolver then validates the optional explicit checkout,
# and the same ordered contract is reinstalled before alias retry.
_REPO_ROOT = Path(__file__).resolve().parent


def _launcher_bootstrap_paths(repo_root: Path, tools_root: Path | None) -> list[str]:
    """Build deterministic, parent-source-first import precedence."""
    paths_to_add: list[str] = []
    if tools_root is not None:
        paths_to_add.extend(
            [
                str(tools_root / "src" / "shared" / "python"),
                str(tools_root / "src"),
                str(tools_root / "src" / "python" / "src"),
            ]
        )

    vendor_src = repo_root / "vendor" / "ud-tools" / "src"
    paths_to_add.extend(
        [
            str(vendor_src / "shared" / "python"),
            str(vendor_src),
            str(vendor_src / "python" / "src"),
        ]
    )

    installed_tools_packages = all(
        (repo_root / package_name).is_dir() for package_name in ("chat", "sidekick")
    )
    if installed_tools_packages:
        paths_to_add.append(str(repo_root))
    paths_to_add.extend(
        [
            str(repo_root / "src" / "shared" / "python"),
            str(repo_root / "src"),
        ]
    )
    if not installed_tools_packages:
        paths_to_add.append(str(repo_root))
    return paths_to_add


def _bootstrap_import_paths(paths_to_add: list[str]) -> None:
    """Prepend bootstrap paths while preserving the supplied precedence order."""
    for path_entry in reversed(paths_to_add):
        while path_entry in path:
            path.remove(path_entry)
        path.insert(0, path_entry)


_requested_tools_path = os.environ.get("TOOLS_REPO_PATH")
_requested_tools_root = (
    Path(_requested_tools_path).expanduser().resolve()
    if _requested_tools_path
    else None
)
_bootstrap_import_paths(_launcher_bootstrap_paths(_REPO_ROOT, _requested_tools_root))
from src.launchers.tools_repo_path import (  # noqa: E402
    resolve_explicit_tools_root as _resolve_explicit_tools_root,
)

_TOOLS_ROOT = _resolve_explicit_tools_root(_requested_tools_path)
_bootstrap_import_paths(_launcher_bootstrap_paths(_REPO_ROOT, _TOOLS_ROOT))


def _retry_parent_shared_alias_installer() -> bool:
    """Retry the source-package alias installer after canonical path bootstrap."""
    import src

    installed = src._install_parent_shared_aliases()
    src._PARENT_SHARED_ALIASES_INSTALLED = installed
    return installed


_PARENT_SHARED_ALIASES_INSTALLED = _retry_parent_shared_alias_installer()
_PARENT_CONTRACTS: ModuleType | None = None


def _load_parent_contracts() -> ModuleType | None:
    """Load the selected Tools contract before downstream aliases exist."""
    parent_root = _TOOLS_ROOT or (_REPO_ROOT / "vendor" / "ud-tools")
    expected_paths = {
        (parent_root / "src" / "shared" / "python" / "contracts.py").resolve(),
        (parent_root / "src" / "contracts.py").resolve(),
    }
    if not any(candidate.is_file() for candidate in expected_paths):
        return None
    module = importlib.import_module("contracts")
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if module_path not in expected_paths:
        expected_text = ", ".join(str(path) for path in sorted(expected_paths))
        raise RuntimeError(
            "Parent Tools contract resolution failed: "
            f"expected one of [{expected_text}], resolved {module_path}"
        )
    return module


def _restore_parent_contract_aliases(parent_contracts: ModuleType | None) -> None:
    """Restore Tools-owned legacy aliases after Upstream imports run."""
    if parent_contracts is None:
        return
    sys.modules["contracts"] = parent_contracts
    sys.modules["shared.python.contracts"] = parent_contracts


from src.api._version import warn_if_unsupported_platform  # noqa: E402

path.append(os.path.join(getcwd(), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("upstream_drift_launcher")


def _ensure_classic_qt_environment() -> None:
    """Prepare and validate the Qt platform before classic launcher startup."""
    if (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
        and not os.environ.get("QT_QPA_PLATFORM")
    ):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        logger.info(
            "No display server detected; using QT_QPA_PLATFORM=offscreen for "
            "classic launcher startup."
        )

    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Classic PyQt6 launcher requires PyQt6 to be installed and loadable. "
            "Use --api-only or the default web UI in headless environments."
        ) from exc

    try:
        QApplication.instance() or QApplication([])
    except (RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Classic PyQt6 launcher could not initialize QApplication. "
            "Set QT_QPA_PLATFORM=offscreen for headless Linux or install the "
            "required Qt platform plugins."
        ) from exc


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
            _ensure_classic_qt_environment()
            # Try new location first
            from src.launchers.upstream_drift_launcher import main as classic_main

            _restore_parent_contract_aliases(_PARENT_CONTRACTS)
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
    global _PARENT_CONTRACTS
    _PARENT_CONTRACTS = _load_parent_contracts()
    warn_if_unsupported_platform()
    args = parse_arguments()
    route_launch(args)


if __name__ == "__main__":
    main()
