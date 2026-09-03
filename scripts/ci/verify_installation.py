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

import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# This script lives in scripts/ci/, so sys.path[0] is scripts/ci -- not the
# repository root. Without this the `src.*` suite checks either fail outright
# or silently resolve against some *other* editable install on the machine.
# Mirrors the `pythonpath` entries in pyproject.toml [tool.pytest.ini_options]:
# the pinned Tools submodule (vendor/ud-tools) must win over the committed
# src/shared/python child copy, exactly as it does under pytest.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_TOOLS_ROOT = _REPO_ROOT / "vendor" / "ud-tools"
VENDOR_SUBMODULE_HINT = "git submodule update --init --recursive vendor/ud-tools"
for _entry in (
    _REPO_ROOT / "src" / "shared" / "python",
    _REPO_ROOT / "src",
    _REPO_ROOT,
    _VENDOR_TOOLS_ROOT / "src",
    _VENDOR_TOOLS_ROOT / "src" / "python" / "src",
    _VENDOR_TOOLS_ROOT / "src" / "shared" / "python",
):
    _path = str(_entry)
    if _entry.is_dir() and _path not in sys.path:
        sys.path.insert(0, _path)


def vendor_shared_python_root(repo_root: Path = _REPO_ROOT) -> Path:
    """Return the canonical Tools ``src/shared/python`` path inside the submodule."""
    return repo_root / "vendor" / "ud-tools" / "src" / "shared" / "python"


def check_vendor_tools(repo_root: Path = _REPO_ROOT) -> tuple[bool, str]:
    """Check that the pinned ``vendor/ud-tools`` submodule is materialised.

    Without it, ``theme``, ``sidekick``, ``chat`` and every other alias-resolved
    Tools package either falls back to the diverged child copy or fails to
    import at all (``utils`` only exists in the submodule).
    """
    vendor_root = vendor_shared_python_root(repo_root)
    if vendor_root.is_dir():
        return True, f"✓ vendor/ud-tools submodule present ({vendor_root})"
    return (
        False,
        f"✗ vendor/ud-tools submodule missing ({vendor_root}); run: "
        f"{VENDOR_SUBMODULE_HINT}",
    )


def shared_alias_roots() -> tuple[str, ...]:
    """Return the roots served by ``SharedImportAliasFinder``, sorted."""
    from src.shared.python import import_aliases

    return tuple(sorted(import_aliases._SHARED_ROOTS))


def check_shared_alias_roots(
    roots: tuple[str, ...] | None = None,
) -> list[tuple[str, bool, str]]:
    """Resolve one import per shared alias root without executing packages.

    ``importlib.util.find_spec`` goes through ``SharedImportAliasFinder`` so
    the result is exactly what ``import <root>`` would bind to. Executing the
    packages is deliberately avoided (``sidekick`` pulls in Qt).

    Returns:
        List of (root, resolved, message) tuples.
    """
    from src.shared.python import import_aliases

    import_aliases.install_shared_import_aliases()
    results: list[tuple[str, bool, str]] = []
    for root in roots or shared_alias_roots():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                spec = importlib.util.find_spec(root)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the report
            results.append((root, False, f"✗ {root}: {exc}"))
            continue
        if spec is None or spec.loader is None:
            results.append((root, False, f"✗ {root}: no module spec"))
            continue
        origin = spec.origin or (
            next(iter(spec.submodule_search_locations or []), None)
        )
        results.append((root, True, f"✓ {root} -> {origin}"))
    return results


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


# Core dependency checks: (display_name, import_path, version_attr). These must
# all come from [project].dependencies in pyproject.toml -- anything that only
# ships in an optional extra belongs in OPTIONAL_CHECKS.
CORE_CHECKS: tuple[tuple[str, str | None, str], ...] = (
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
)

# Optional / extras-only packages. Missing ones are reported but do not fail
# the run, because a core install is a valid install.
OPTIONAL_CHECKS: tuple[tuple[str, str | None, str], ...] = (
    ("PyQt6", "PyQt6.QtCore", "PYQT_VERSION_STR"),  # extra: gui-test / tools
    ("pandas", None, "__version__"),  # extra: data / dev
    ("matplotlib", None, "__version__"),  # extra: dev
    ("sympy", None, "__version__"),  # extra: dev
    ("defusedxml", None, "__version__"),  # extra: urdf / dev
)

# Project-specific modules that must import in a working checkout.
SUITE_CHECKS: tuple[str, ...] = (
    "src.shared.python.engine_core.interfaces",
    "src.shared.python.physics.ball_flight_physics",
    "src.shared.python.physics.flight_models",
    "src.shared.python.engine_core.engine_manager",
    "src.shared.python.engine_core.engine_registry",
    "src.shared.python.validation_pkg.statistical_analysis",
    "src.shared.python.plotting",
    "src.api.server",
)


def _section(title: str) -> None:
    logger.info("")
    logger.info(title)
    logger.info("-" * 40)


def _run_core_checks() -> list[bool]:
    _section("Checking core dependencies:")
    results: list[bool] = []
    for display_name, import_path, version_attr in CORE_CHECKS:
        success, message = check_import(display_name, import_path, version_attr)
        logger.info(message)
        results.append(success)
    return results


def _run_optional_checks() -> None:
    _section("Checking optional dependencies (advisory):")
    for display_name, import_path, version_attr in OPTIONAL_CHECKS:
        success, message = check_import(display_name, import_path, version_attr)
        if success:
            logger.info(message)
        else:
            logger.info("- %s not installed (optional extra)", display_name)


def _run_suite_checks() -> list[bool]:
    _section("Checking Golf Suite modules:")
    results: list[bool] = []
    for module_name in SUITE_CHECKS:
        success, _message = check_import(module_name, None, "__version__")
        # Project modules may not have __version__, so report by name only.
        if success:
            logger.info("✓ %s", module_name)
        else:
            logger.warning("✗ %s: Import failed", module_name)
        results.append(success)
    return results


def _run_alias_root_checks(vendor_present: bool) -> list[bool]:
    _section("Checking shared Tools alias roots (vendor/ud-tools):")
    if not vendor_present:
        logger.warning("- skipped: %s", VENDOR_SUBMODULE_HINT)
        return []
    results: list[bool] = []
    for _root, success, message in check_shared_alias_roots():
        if success:
            logger.info(message)
        else:
            logger.warning(message)
        results.append(success)
    return results


def _report(
    *,
    py_ok: bool,
    venv_ok: bool,
    vendor_ok: bool,
    core: list[bool],
    suite: list[bool],
    alias: list[bool],
    json_output: bool,
) -> int:
    """Log the summary (and JSON when requested); return the exit code."""
    passed = sum(core) + sum(suite) + sum(alias)
    total = len(core) + len(suite) + len(alias)
    all_passed = py_ok and vendor_ok and passed == total

    logger.info("")
    logger.info("=" * 60)
    logger.info("Python version:    %s", "OK" if py_ok else "FAILED")
    logger.info("vendor/ud-tools:   %s", "OK" if vendor_ok else "MISSING")
    logger.info("Core dependencies: %d/%d passed", sum(core), len(core))
    logger.info("Suite modules:     %d/%d passed", sum(suite), len(suite))
    logger.info("Alias roots:       %d/%d passed", sum(alias), len(alias))
    logger.info("Overall:           %d/%d passed", passed, total)
    logger.info("")

    if json_output:
        result = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_ok": py_ok,
            "in_virtualenv": venv_ok,
            "vendor_tools_ok": vendor_ok,
            "core_checks": {"passed": sum(core), "total": len(core)},
            "suite_checks": {"passed": sum(suite), "total": len(suite)},
            "alias_root_checks": {"passed": sum(alias), "total": len(alias)},
            "overall": {"passed": passed, "total": total},
            "status": "passed" if all_passed else "failed",
        }
        print(json.dumps(result, indent=2))

    if all_passed:
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
    if not py_ok:
        logger.info("  4. Your Python version is too old; upgrade to 3.11+")
    if not vendor_ok:
        logger.info("  5. The pinned Tools submodule is missing; run:")
        logger.info("     %s", VENDOR_SUBMODULE_HINT)
    return 1


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

    py_ok, py_msg = check_python_version()
    logger.info(py_msg)
    venv_ok, venv_msg = check_virtualenv()  # advisory
    logger.info(venv_msg)
    vendor_ok, vendor_msg = check_vendor_tools()  # critical for aliased imports
    logger.info(vendor_msg)

    core = _run_core_checks()
    _run_optional_checks()
    suite = _run_suite_checks()
    alias = _run_alias_root_checks(vendor_ok)

    return _report(
        py_ok=py_ok,
        venv_ok=venv_ok,
        vendor_ok=vendor_ok,
        core=core,
        suite=suite,
        alias=alias,
        json_output=json_output,
    )


if __name__ == "__main__":
    sys.exit(main())
