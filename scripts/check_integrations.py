#!/usr/bin/env python3
"""Integration troubleshooting script for UpstreamDrift.

Checks integration status with Tools repo and verifies UI consistency
between PyQt6 and React implementations.

Usage:
    python scripts/check_integrations.py [--verbose] [--fix]

Options:
    --verbose   Show detailed output
    --fix       Attempt to fix common issues
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class CheckResult(NamedTuple):
    """Result of an integration check."""

    name: str
    passed: bool
    message: str
    fix_command: str | None = None


def check_tools_repo_availability() -> CheckResult:
    """Check if Tools repo packages are available."""
    try:
        import model_generation  # noqa: F401

        return CheckResult(
            name="Tools Repo - model_generation",
            passed=True,
            message="model_generation package is installed and importable",
        )
    except ImportError:
        return CheckResult(
            name="Tools Repo - model_generation",
            passed=False,
            message="model_generation not installed. Install with: pip install -e ../Tools[urdf]",
            fix_command="pip install -e ../Tools[urdf]",
        )


def check_signal_toolkit() -> CheckResult:
    """Check if signal_toolkit is available."""
    try:
        import signal_toolkit  # noqa: F401

        return CheckResult(
            name="Tools Repo - signal_toolkit",
            passed=True,
            message="signal_toolkit package is installed",
        )
    except ImportError:
        return CheckResult(
            name="Tools Repo - signal_toolkit",
            passed=False,
            message="signal_toolkit not installed. Install with: pip install -e ../Tools[signal]",
            fix_command="pip install -e ../Tools[signal]",
        )


def check_frontend_dependencies() -> CheckResult:
    """Check if frontend dependencies are installed."""
    ui_path = Path("ui")
    node_modules = ui_path / "node_modules"

    if not ui_path.exists():
        return CheckResult(
            name="Frontend Dependencies",
            passed=False,
            message="UI directory not found",
        )

    if not node_modules.exists():
        return CheckResult(
            name="Frontend Dependencies",
            passed=False,
            message="node_modules not found. Run: cd ui && npm install",
            fix_command="cd ui && npm install",
        )

    return CheckResult(
        name="Frontend Dependencies",
        passed=True,
        message="Frontend dependencies installed",
    )


def check_frontend_build() -> CheckResult:
    """Check if frontend can build successfully."""
    ui_path = Path("ui")

    if not ui_path.exists():
        return CheckResult(
            name="Frontend Build",
            passed=False,
            message="UI directory not found",
        )

    package_json = ui_path / "package.json"
    if not package_json.exists():
        return CheckResult(
            name="Frontend Build",
            passed=False,
            message="package.json not found",
        )

    # Check for build script
    with open(package_json) as f:
        pkg = json.load(f)

    if "build" not in pkg.get("scripts", {}):
        return CheckResult(
            name="Frontend Build",
            passed=False,
            message="No build script in package.json",
        )

    return CheckResult(
        name="Frontend Build",
        passed=True,
        message="Build script available. Run: cd ui && npm run build",
    )


def check_frontend_tests() -> CheckResult:
    """Check if frontend tests are configured."""
    ui_path = Path("ui")
    package_json = ui_path / "package.json"

    if not package_json.exists():
        return CheckResult(
            name="Frontend Tests",
            passed=False,
            message="package.json not found",
        )

    with open(package_json) as f:
        pkg = json.load(f)

    if "test" not in pkg.get("scripts", {}):
        return CheckResult(
            name="Frontend Tests",
            passed=False,
            message="No test script in package.json. Add Vitest configuration.",
            fix_command="npm install -D vitest @testing-library/react jsdom",
        )

    # Check for vitest config
    vitest_config = ui_path / "vitest.config.ts"
    if not vitest_config.exists():
        return CheckResult(
            name="Frontend Tests",
            passed=False,
            message="vitest.config.ts not found",
        )

    return CheckResult(
        name="Frontend Tests",
        passed=True,
        message="Frontend tests configured. Run: cd ui && npm test",
    )


def check_api_server() -> CheckResult:
    """Check if API server module exists."""
    api_path = Path("src/api/local_server.py")

    if not api_path.exists():
        return CheckResult(
            name="API Server",
            passed=False,
            message="API server not found at src/api/local_server.py",
        )

    return CheckResult(
        name="API Server",
        passed=True,
        message="API server module found",
    )


def check_pyqt_launcher() -> CheckResult:
    """Check if PyQt6 launcher exists and imports work."""
    launcher_path = Path("src/launchers/golf_launcher.py")

    if not launcher_path.exists():
        return CheckResult(
            name="PyQt6 Launcher",
            passed=False,
            message="PyQt6 launcher not found",
        )

    try:
        import PyQt6  # noqa: F401

        return CheckResult(
            name="PyQt6 Launcher",
            passed=True,
            message="PyQt6 launcher found and PyQt6 is installed",
        )
    except ImportError:
        return CheckResult(
            name="PyQt6 Launcher",
            passed=False,
            message="PyQt6 not installed. Install with: pip install PyQt6",
            fix_command="pip install PyQt6",
        )


def check_react_components() -> CheckResult:
    """Check for required React components."""
    required_components = [
        "ui/src/components/simulation/EngineSelector.tsx",
        "ui/src/components/simulation/SimulationControls.tsx",
        "ui/src/components/visualization/Scene3D.tsx",
        "ui/src/api/client.ts",
    ]

    missing = [c for c in required_components if not Path(c).exists()]

    if missing:
        return CheckResult(
            name="React Components",
            passed=False,
            message=f"Missing components: {', '.join(missing)}",
        )

    return CheckResult(
        name="React Components",
        passed=True,
        message="All required React components found",
    )


def check_ui_feature_parity() -> CheckResult:
    """Check for basic feature parity between PyQt and React UIs."""
    issues = []

    # Check engine selection
    pyqt_launcher = Path("src/launchers/golf_launcher.py")
    react_engine = Path("ui/src/components/simulation/EngineSelector.tsx")

    if pyqt_launcher.exists() and not react_engine.exists():
        issues.append("Engine selector missing from React UI")
    elif react_engine.exists() and not pyqt_launcher.exists():
        issues.append("PyQt launcher missing")

    # Check 3D visualization
    react_3d = Path("ui/src/components/visualization/Scene3D.tsx")
    if not react_3d.exists():
        issues.append("3D visualization missing from React UI")

    if issues:
        return CheckResult(
            name="UI Feature Parity",
            passed=False,
            message=f"Feature parity issues: {'; '.join(issues)}",
        )

    return CheckResult(
        name="UI Feature Parity",
        passed=True,
        message="Basic feature parity verified",
    )


def run_all_checks(verbose: bool = False) -> list[CheckResult]:
    """Run all integration checks."""
    checks = [
        check_tools_repo_availability,
        check_signal_toolkit,
        check_frontend_dependencies,
        check_frontend_build,
        check_frontend_tests,
        check_api_server,
        check_pyqt_launcher,
        check_react_components,
        check_ui_feature_parity,
    ]

    results = []
    for check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
            if verbose:
                status = "" if result.passed else ""
                print(f"{status} {result.name}: {result.message}")
        except Exception as e:
            results.append(
                CheckResult(
                    name=check_fn.__name__,
                    passed=False,
                    message=f"Check failed with error: {e}",
                )
            )

    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check UpstreamDrift integrations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix common issues"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("UpstreamDrift Integration Check")
    print("=" * 60)
    print()

    results = run_all_checks(verbose=args.verbose)

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if not args.verbose:
        print("\nFailed checks:")
        for result in results:
            if not result.passed:
                print(f"  - {result.name}: {result.message}")
                if result.fix_command:
                    print(f"    Fix: {result.fix_command}")

    if args.fix:
        print("\nAttempting fixes...")
        for result in results:
            if not result.passed and result.fix_command:
                print(f"  Running: {result.fix_command}")
                try:
                    subprocess.run(result.fix_command, shell=True, check=True)
                    print("  Success!")
                except subprocess.CalledProcessError as e:
                    print(f"  Failed: {e}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
