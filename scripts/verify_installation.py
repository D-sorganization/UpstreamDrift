#!/usr/bin/env python
"""Verify Golf Modeling Suite installation.

This script validates that all required dependencies are installed,
can be imported, and work correctly. It also checks physics engines,
required files, environment variables, and the API server.

Usage:
    python scripts/verify_installation.py

Exit codes:
    0 - All critical checks passed
    1 - Some critical checks failed
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


def check_import(
    display_name: str, import_path: str | None = None, version_attr: str = "__version__"
) -> tuple[bool, str]:
    """Try to import a module and report status.

    Args:
        display_name: Name to display in output
        import_path: Module path to import (defaults to display_name)
        version_attr: Attribute to check for version (default: __version__)

    Returns:
        Tuple of (success, message)
    """
    if not isinstance(display_name, str):
        raise ValueError("display_name must be a string")
    if not (import_path is None or isinstance(import_path, str)):
        raise ValueError("import_path must be None or string")
    if not isinstance(version_attr, str):
        raise ValueError("version_attr must be a string")

    module_path = import_path or display_name
    try:
        module = __import__(module_path, fromlist=[""])
        version = getattr(module, version_attr, "unknown")
        return True, f"✓ {display_name} (v{version})"
    except ImportError as e:
        return False, f"✗ {display_name}: {e}"
    except (RuntimeError, OSError) as e:
        return False, f"✗ {display_name}: Unexpected error - {e}"


def check_deep_import(
    display_name: str, import_path: str, test_func: Any
) -> tuple[bool, str]:
    """Import a module and run a test function on it.

    Args:
        display_name: Name to display in output
        import_path: Module path to import
        test_func: Callable that takes the module and returns bool

    Returns:
        Tuple of (success, message)
    """
    try:
        module = __import__(import_path, fromlist=[""])
        if test_func(module):
            return True, f"✓ {display_name} (functional)"
        return (
            False,
            f"✗ {display_name}: Import succeeded but functionality check failed",
        )
    except ImportError as e:
        return False, f"✗ {display_name}: Import failed - {e}"
    except Exception as e:
        return False, f"✗ {display_name}: Unexpected error - {e}"


def test_numpy(module: Any) -> bool:
    """Test numpy basic functionality."""
    try:
        arr = module.array([1, 2, 3])
        return arr.sum() == 6
    except Exception:
        return False


def test_scipy(module: Any) -> bool:
    """Test scipy basic functionality."""
    try:
        from scipy import special

        return special.erf(1.0) > 0
    except Exception:
        return False


def test_yaml(module: Any) -> bool:
    """Test yaml basic functionality."""
    try:
        data = module.safe_load("test: 123")
        return isinstance(data, dict)
    except Exception:
        return False


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is 3.10+."""
    version_info = sys.version_info
    if version_info.major >= 3 and version_info.minor >= 10:
        return (
            True,
            f"✓ Python {version_info.major}.{version_info.minor}.{version_info.micro}",
        )
    return False, f"✗ Python {version_info.major}.{version_info.minor} (need 3.10+)"


def check_virtual_env() -> tuple[bool, str]:
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        return True, f"✓ Running in virtual environment ({sys.prefix})"
    return False, "✗ Not running in a virtual environment (recommended)"


def check_pythonpath() -> tuple[bool, str]:
    """Check if repo root is in PYTHONPATH."""
    try:
        repo_root = pathlib.Path(__file__).parent.parent.resolve()
        repo_root_str = str(repo_root)

        # Check if repo root is accessible (can import src)
        sys.path.insert(0, str(repo_root))
        try:
            __import__("src")
            return True, f"✓ Repo root ({repo_root_str}) is accessible"
        except ImportError:
            return False, f"✗ Cannot import 'src' from repo root ({repo_root_str})"
    except Exception as e:
        return False, f"✗ PYTHONPATH check failed: {e}"


def check_physics_engine(engine_name: str, import_path: str) -> tuple[str, str]:
    """Check if a physics engine is available and working.

    Args:
        engine_name: Display name for the engine
        import_path: Module path to import

    Returns:
        Tuple of (status, message) where status is "AVAILABLE", "UNAVAILABLE", or "BROKEN"
    """
    try:
        __import__(import_path, fromlist=[""])
        # For MuJoCo, try to create a minimal model
        if engine_name == "MuJoCo":
            try:
                import mujoco

                mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
                return "AVAILABLE", f"✓ {engine_name}: Functional"
            except Exception as e:
                return (
                    "BROKEN",
                    f"✗ {engine_name}: Import OK but initialization failed - {e}",
                )
        else:
            # For other engines, just test import
            return "AVAILABLE", f"✓ {engine_name}: Importable"
    except ImportError as e:
        return "UNAVAILABLE", f"- {engine_name}: Not installed ({e})"
    except Exception as e:
        return "BROKEN", f"✗ {engine_name}: Unexpected error - {e}"


def check_required_files() -> tuple[bool, list[str]]:
    """Check if required data files and model files exist.

    Returns:
        Tuple of (all_exist, messages)
    """
    repo_root = pathlib.Path(__file__).parent.parent
    required_files = [
        repo_root / "assets" / "config",
        repo_root / "data",
        repo_root
        / "src"
        / "engines"
        / "physics_engines"
        / "pinocchio"
        / "models"
        / "generated"
        / "golfer.urdf",
        repo_root
        / "src"
        / "shared"
        / "python"
        / "model_generation"
        / "library"
        / "bundled"
        / "simple_arm"
        / "arm.urdf",
    ]

    messages = []
    all_exist = True

    for file_path in required_files:
        if file_path.exists():
            rel_path = file_path.relative_to(repo_root)
            messages.append(f"✓ {rel_path}")
        else:
            rel_path = file_path.relative_to(repo_root)
            messages.append(f"✗ {rel_path} (missing)")
            all_exist = False

    return all_exist, messages


def check_api_server() -> tuple[bool, str]:
    """Try to start the API server and ping the /health endpoint.

    Returns:
        Tuple of (success, message)
    """
    try:
        import socket

        # Check if port 8001 is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", 8001))
        sock.close()

        if result == 0:
            return False, "✗ API Server: Port 8001 already in use"

        # Try to start the server in background
        repo_root = pathlib.Path(__file__).parent.parent
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "src.api.server:create_app",
                    "--port",
                    "8001",
                    "--host",
                    "127.0.0.1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait up to 5 seconds for server to start
            start_time = time.time()
            while time.time() - start_time < 5:
                try:
                    import requests

                    response = requests.get("http://127.0.0.1:8001/health", timeout=1)
                    if response.ok:
                        process.terminate()
                        process.wait(timeout=2)
                        return True, "✓ API Server: Started and healthy (port 8001)"
                except Exception:
                    time.sleep(0.5)

            # Timeout or failed to connect
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

            return (
                False,
                "✗ API Server: Failed to start or /health unreachable within 5s",
            )

        except FileNotFoundError:
            return False, "✗ API Server: uvicorn not found"
        except Exception as e:
            return False, f"✗ API Server: Unexpected error - {e}"

    except Exception as e:
        return False, f"✗ API Server: Check failed - {e}"


def main() -> int:
    """Run all verification checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("Golf Modeling Suite - Installation Verification")
    logger.info("=" * 70)
    logger.info("")

    # Track all results
    results: dict[str, list[bool]] = {
        "environment": [],
        "core": [],
        "deep": [],
        "suite": [],
        "physics_engines": [],
        "files": [],
        "api": [],
    }

    # ENVIRONMENT CHECKS
    logger.info("Environment Validation:")
    logger.info("-" * 70)

    success, message = check_python_version()
    logger.info(message)
    results["environment"].append(success)

    _venv_ok, message = check_virtual_env()
    logger.info(message)
    # check_virtual_env is recommended, not critical — don't fold into env results

    success, message = check_pythonpath()
    logger.info(message)
    results["environment"].append(success)

    logger.info("")

    # CORE DEPENDENCIES
    logger.info("Core Dependencies:")
    logger.info("-" * 70)

    checks: list[tuple[str, str | None, str]] = [
        # Core scientific computing
        ("numpy", None, "__version__"),
        ("scipy", None, "__version__"),
        ("pandas", None, "__version__"),
        ("matplotlib", None, "__version__"),
        ("sympy", None, "__version__"),
        # GUI
        ("PyQt6", "PyQt6.QtCore", "PYQT_VERSION_STR"),
        # Physics engines
        ("mujoco", None, "__version__"),
        # Web framework
        ("fastapi", None, "__version__"),
        ("uvicorn", None, "__version__"),
        # Data formats
        ("yaml", "yaml", "__version__"),
        ("defusedxml", None, "__version__"),
        # Security
        ("passlib", None, "__version__"),
        ("jose", None, "__version__"),
    ]

    for display_name, import_path, version_attr in checks:
        success, message = check_import(display_name, import_path, version_attr)
        logger.info(message)
        results["core"].append(success)

    logger.info("")

    # DEEP DEPENDENCY CHECKS (actually use packages)
    logger.info("Dependency Functionality (Deep Check):")
    logger.info("-" * 70)

    deep_checks: list[tuple[str, str, Any]] = [
        ("numpy", "numpy", test_numpy),
        ("scipy", "scipy", test_scipy),
        ("PyYAML", "yaml", test_yaml),
    ]

    for display_name, import_path, test_func in deep_checks:
        success, message = check_deep_import(display_name, import_path, test_func)
        logger.info(message)
        results["deep"].append(success)

    logger.info("")

    # SUITE MODULES
    logger.info("Golf Suite Modules:")
    logger.info("-" * 70)

    suite_checks: list[tuple[str, str | None]] = [
        ("src.shared.python.interfaces", None),
        ("src.shared.python.ball_flight_physics", None),
        ("src.shared.python.flight_models", None),
        ("src.shared.python.engine_core.engine_manager", None),
        ("src.shared.python.engine_registry", None),
        ("src.shared.python.statistical_analysis", None),
        ("src.shared.python.plotting", None),
        ("src.api.server", None),
    ]

    for display_name, import_path in suite_checks:
        success, message = check_import(display_name, import_path, "__version__")
        # Project modules may not have __version__, adjust message
        if success:
            logger.info("✓ %s", display_name)
        else:
            logger.info("✗ %s: Import failed", display_name)
        results["suite"].append(success)

    logger.info("")

    # PHYSICS ENGINES
    logger.info("Physics Engines:")
    logger.info("-" * 70)

    engines = [
        ("MuJoCo", "mujoco"),
        ("Drake", "pydrake"),
        ("Pinocchio", "pinocchio"),
        ("OpenSim", "opensim"),
    ]

    for engine_name, import_path in engines:
        status, message = check_physics_engine(engine_name, import_path)
        logger.info(message)
        results["physics_engines"].append(status == "AVAILABLE")

    logger.info("")

    # REQUIRED FILES
    logger.info("Required Files and Assets:")
    logger.info("-" * 70)

    all_files_exist, file_messages = check_required_files()
    for msg in file_messages:
        logger.info(msg)
    results["files"].append(all_files_exist)

    logger.info("")

    # API SERVER TEST
    logger.info("API Server Test:")
    logger.info("-" * 70)

    success, message = check_api_server()
    logger.info(message)
    results["api"].append(success)

    logger.info("")
    logger.info("=" * 70)

    # SUMMARY
    logger.info("Summary:")
    logger.info("-" * 70)

    def count_results(results_list: list[bool]) -> tuple[int, int]:
        """Count passed tests."""
        return sum(results_list), len(results_list)

    env_passed, env_total = count_results(results["environment"])
    core_passed, core_total = count_results(results["core"])
    deep_passed, deep_total = count_results(results["deep"])
    suite_passed, suite_total = count_results(results["suite"])
    physics_passed, physics_total = count_results(results["physics_engines"])
    files_passed, files_total = count_results(results["files"])
    api_passed, api_total = count_results(results["api"])

    logger.info("Environment:       %d/%d", env_passed, env_total)
    logger.info("Core dependencies: %d/%d", core_passed, core_total)
    logger.info("Deep checks:       %d/%d", deep_passed, deep_total)
    logger.info("Suite modules:     %d/%d", suite_passed, suite_total)
    logger.info("Physics engines:   %d/%d available", physics_passed, physics_total)
    logger.info("Required files:    %d/%d", files_passed, files_total)
    logger.info("API server:        %d/%d", api_passed, api_total)
    logger.info("")

    # Determine overall pass/fail
    critical_results = (
        env_passed == env_total
        and core_passed == core_total
        and deep_passed == deep_total
        and suite_passed == suite_total
        and files_passed == files_total
    )

    if critical_results:
        logger.info("✓ Installation verified successfully!")
        logger.info("")
        logger.info("Physics engines available:")
        if physics_passed == 0:
            logger.info("  (None - install optional physics engine packages)")
        else:
            logger.info("  %d/%d engines installed", physics_passed, physics_total)
        logger.info("")
        logger.info("You can now run:")
        logger.info("  python examples/01_basic_simulation.py")
        logger.info("  python -m uvicorn src.api.server:create_app --reload")
        logger.info("")
        return 0

    logger.warning("✗ Some critical checks failed.")
    logger.info("")
    logger.info("Troubleshooting:")
    logger.info("  1. Check Python version: python --version (need 3.10+)")
    logger.info("  2. Install dependencies: pip install -e '.[dev]'")
    logger.info("  3. For physics engines: pip install -e '.[engines]'")
    logger.info("  4. See docs/troubleshooting/installation.md")
    logger.info("")
    return 1


if __name__ == "__main__":
    sys.exit(main())
