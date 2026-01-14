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

import sys


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
    except Exception as e:
        return False, f"✗ {display_name}: Unexpected error - {e}"


def main() -> int:
    """Run all verification checks."""
    print("=" * 60)
    print("Golf Modeling Suite - Installation Verification")
    print("=" * 60)
    print()

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

    print("Checking core dependencies:")
    print("-" * 40)

    core_results = []
    for display_name, import_path, version_attr in checks:
        success, message = check_import(display_name, import_path, version_attr)
        print(message)
        core_results.append(success)

    print()
    print("Checking Golf Suite modules:")
    print("-" * 40)

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
            print(f"✓ {display_name}")
        else:
            print(f"✗ {display_name}: Import failed")
        suite_results.append(success)

    print()
    print("=" * 60)

    # Summary
    core_passed = sum(core_results)
    core_total = len(core_results)
    suite_passed = sum(suite_results)
    suite_total = len(suite_results)
    total_passed = core_passed + suite_passed
    total_checks = core_total + suite_total

    print(f"Core dependencies: {core_passed}/{core_total} passed")
    print(f"Suite modules:     {suite_passed}/{suite_total} passed")
    print(f"Overall:           {total_passed}/{total_checks} passed")
    print()

    if total_passed == total_checks:
        print("✓ Installation verified successfully!")
        print()
        print("You can now run:")
        print("  python launchers/golf_suite_launcher.py")
        print("  python -m api.server")
        return 0
    else:
        print("✗ Some checks failed.")
        print()
        print("Troubleshooting:")
        print("  1. See docs/troubleshooting/installation.md")
        print("  2. Try: conda env create -f environment.yml")
        print("  3. Or:  pip install -e '.[dev,engines]'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
