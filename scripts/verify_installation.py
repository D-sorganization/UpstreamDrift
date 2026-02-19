#!/usr/bin/env python
"""Verify Golf Modeling Suite installation.

This script checks that all required dependencies are installed and
the core modules can be imported successfully.

Usage:
    python scripts/verify_installation.py

Exit codes:
    0 - All checks passed
    1 - Some checks failed
"""

from __future__ import annotations

import logging
import sys

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
    module_path = import_path or display_name
    try:
        module = __import__(module_path, fromlist=[""])
        version = getattr(module, version_attr, "unknown")
        return True, f"✓ {display_name} (v{version})"
    except ImportError as e:
        return False, f"✗ {display_name}: {e}"
    except (RuntimeError, OSError) as e:
        return False, f"✗ {display_name}: Unexpected error - {e}"


def main() -> int:
    """Run all verification checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 60)
    logger.info("Golf Modeling Suite - Installation Verification")
    logger.info("=" * 60)
    logger.info("")

    # Define checks: (display_name, import_path, version_attr)
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

    logger.info("Checking core dependencies:")
    logger.info("-" * 40)

    core_results = []
    for display_name, import_path, version_attr in checks:
        success, message = check_import(display_name, import_path, version_attr)
        logger.info(message)
        core_results.append(success)

    logger.info("")
    logger.info("Checking Golf Suite modules:")
    logger.info("-" * 40)

    # Project-specific modules
    suite_checks: list[tuple[str, str | None]] = [
        ("shared.python.interfaces", None),
        ("shared.python.ball_flight_physics", None),
        ("shared.python.flight_models", None),
        ("shared.python.engine_manager", None),
        ("shared.python.engine_registry", None),
        ("shared.python.statistical_analysis", None),
        ("shared.python.plotting", None),
        ("api.server", None),
    ]

    suite_results = []
    for display_name, import_path in suite_checks:
        success, message = check_import(display_name, import_path, "__version__")
        # Project modules may not have __version__, adjust message
        if success:
            logger.info("✓ %s", display_name)
        else:
            logger.warning("✗ %s: Import failed", display_name)
        suite_results.append(success)

    logger.info("")
    logger.info("=" * 60)

    # Summary
    core_passed = sum(core_results)
    core_total = len(core_results)
    suite_passed = sum(suite_results)
    suite_total = len(suite_results)
    total_passed = core_passed + suite_passed
    total_checks = core_total + suite_total

    logger.info("Core dependencies: %d/%d passed", core_passed, core_total)
    logger.info("Suite modules:     %d/%d passed", suite_passed, suite_total)
    logger.info("Overall:           %d/%d passed", total_passed, total_checks)
    logger.info("")

    if total_passed == total_checks:
        logger.info("✓ Installation verified successfully!")
        logger.info("")
        logger.info("You can now run:")
        logger.info("  python launchers/golf_suite_launcher.py")
        logger.info("  python -m api.server")
        return 0
    logger.warning("✗ Some checks failed.")
    logger.info("")
    logger.info("Troubleshooting:")
    logger.info("  1. See docs/troubleshooting/installation.md")
    logger.info("  2. Try: conda env create -f environment.yml")
    logger.info("  3. Or:  pip install -e '.[dev,engines]'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
