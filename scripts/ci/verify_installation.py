#!/usr/bin/env python
"""Verify Golf Modeling Suite installation.

This script checks that all required dependencies are installed and
the core modules can be imported successfully.

Usage:
    python scripts/ci/verify_installation.py [--json]

Exit codes:
    0 - All critical checks passed
    1 - Some critical checks failed

Options:
    --json    Output structured JSON result
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# This script lives in scripts/ci/, so sys.path[0] is scripts/ci -- not the
# repository root. Without this the `src.*` suite checks either fail outright
# or silently resolve against some *other* editable install on the machine.
# Mirrors the `pythonpath` entries in pyproject.toml [tool.pytest.ini_options].
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (
    _REPO_ROOT,
    _REPO_ROOT / "src",
    _REPO_ROOT / "src" / "shared" / "python",
):
    _path = str(_entry)
    if _entry.is_dir() and _path not in sys.path:
        sys.path.insert(0, _path)


def check_python_version() -> tuple[bool, str]:
    """Check Python version satisfies pyproject's requires-python (>=3.11)."""
    required_major, required_minor = 3, 11
    if sys.version_info >= (required_major, required_minor):
        version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return True, f"✓ Python version {version_str}"
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        False,
        f"✗ Python {version_str} (requires {required_major}.{required_minor}+)",
    )


def check_virtualenv() -> tuple[bool, str]:
    """Check if running in a virtual environment (advisory, not blocking)."""
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        return True, f"✓ Virtual environment detected: {sys.prefix}"
    return False, "⚠ System Python (virtualenv recommended but not required)"


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


def main() -> int:
    """Run all verification checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    json_output = "--json" in sys.argv

    logger.info("=" * 60)
    logger.info("Golf Modeling Suite - Installation Verification")
    logger.info("=" * 60)
    logger.info("")

    logger.info("Environment Checks:")
    logger.info("-" * 40)

    # Check Python version
    py_success, py_msg = check_python_version()
    logger.info(py_msg)

    # Check virtualenv (advisory)
    venv_success, venv_msg = check_virtualenv()
    logger.info(venv_msg)

    logger.info("")

    # Define checks: (display_name, import_path, version_attr).
    # These must all come from [project].dependencies in pyproject.toml --
    # anything that only ships in an optional extra belongs in OPTIONAL_CHECKS.
    checks: list[tuple[str, str | None, str]] = [
        # Core scientific computing
        ("numpy", None, "__version__"),
        ("scipy", None, "__version__"),
        # Physics engines
        ("mujoco", None, "__version__"),
        # Web framework
        ("fastapi", None, "__version__"),
        ("uvicorn", None, "__version__"),
        ("pydantic", None, "__version__"),
        # Data formats
        ("yaml", "yaml", "__version__"),
        ("h5py", None, "__version__"),
        # Persistence
        ("sqlalchemy", None, "__version__"),
        # Security (auth uses bcrypt + PyJWT; passlib/python-jose are not deps)
        ("bcrypt", None, "__version__"),
        ("PyJWT", "jwt", "__version__"),
    ]

    # Optional / extras-only packages. Missing ones are reported but do not
    # fail the run, because a core install is a valid install.
    optional_checks: list[tuple[str, str | None, str]] = [
        ("PyQt6", "PyQt6.QtCore", "PYQT_VERSION_STR"),  # extra: gui-test / tools
        ("pandas", None, "__version__"),  # extra: data / dev
        ("matplotlib", None, "__version__"),  # extra: dev
        ("sympy", None, "__version__"),  # extra: dev
        ("defusedxml", None, "__version__"),  # extra: urdf / dev
    ]

    logger.info("Checking core dependencies:")
    logger.info("-" * 40)

    core_results = []
    for display_name, import_path, version_attr in checks:
        success, message = check_import(display_name, import_path, version_attr)
        logger.info(message)
        core_results.append(success)

    logger.info("")
    logger.info("Checking optional dependencies (advisory):")
    logger.info("-" * 40)

    for display_name, import_path, version_attr in optional_checks:
        success, message = check_import(display_name, import_path, version_attr)
        if success:
            logger.info(message)
        else:
            logger.info("- %s not installed (optional extra)", display_name)

    logger.info("")
    logger.info("Checking Golf Suite modules:")
    logger.info("-" * 40)

    # Project-specific modules
    suite_checks: list[tuple[str, str | None]] = [
        ("src.shared.python.engine_core.interfaces", None),
        ("src.shared.python.physics.ball_flight_physics", None),
        ("src.shared.python.physics.flight_models", None),
        ("src.shared.python.engine_core.engine_manager", None),
        ("src.shared.python.engine_core.engine_registry", None),
        ("src.shared.python.validation_pkg.statistical_analysis", None),
        ("src.api.server", None),
    ]
    # Suite modules that only import with an optional extra installed. They
    # are reported but never fail the run: the always-on CI lane (#9409) runs
    # this script against a bare requirements.lock install, which has no
    # matplotlib (the `dev` extra) and therefore no plotting package.
    optional_suite_checks: list[tuple[str, str | None]] = [
        ("src.shared.python.plotting", None),  # needs matplotlib (extra: dev)
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
    for display_name, import_path in optional_suite_checks:
        success, message = check_import(display_name, import_path, "__version__")
        if success:
            logger.info("✓ %s (optional)", display_name)
        else:
            logger.info("- %s not importable (needs an optional extra)", display_name)

    logger.info("")
    logger.info("=" * 60)

    # Summary
    py_critical = py_success
    core_passed = sum(core_results)
    core_total = len(core_results)
    suite_passed = sum(suite_results)
    suite_total = len(suite_results)
    total_passed = core_passed + suite_passed
    total_checks = core_total + suite_total

    logger.info("Python version:    %s", "OK" if py_critical else "FAILED")
    logger.info("Core dependencies: %d/%d passed", core_passed, core_total)
    logger.info("Suite modules:     %d/%d passed", suite_passed, suite_total)
    logger.info("Overall:           %d/%d passed", total_passed, total_checks)
    logger.info("")

    if json_output:
        result = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_ok": py_critical,
            "in_virtualenv": venv_success,
            "core_checks": {"passed": core_passed, "total": core_total},
            "suite_checks": {"passed": suite_passed, "total": suite_total},
            "overall": {"passed": total_passed, "total": total_checks},
            "status": (
                "passed" if (py_critical and total_passed == total_checks) else "failed"
            ),
        }
        print(json.dumps(result, indent=2))

    if py_critical and total_passed == total_checks:
        logger.info("✓ Installation verified successfully!")
        logger.info("")
        logger.info("You can now run:")
        logger.info("  upstream-drift")
        logger.info("  python launch_upstream_drift.py")
        logger.info("  python -m src.api.local_server")
        return 0
    logger.warning("✗ Some critical checks failed.")
    logger.info("")
    logger.info("Troubleshooting:")
    logger.info("  1. See docs/troubleshooting/installation.md")
    logger.info("  2. Try: conda env create -f environment.yml")
    logger.info("  3. Or:  pip install -e '.[dev]'")
    logger.info("     (optional engines: '.[all-engines]', '.[biomechanics]')")
    if not py_critical:
        logger.info("  4. Your Python version is too old; upgrade to 3.11+")
    return 1


if __name__ == "__main__":
    sys.exit(main())
