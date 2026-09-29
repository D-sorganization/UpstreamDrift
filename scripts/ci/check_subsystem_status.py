#!/usr/bin/env python3
"""
Subsystem Status Checker

This script checks that all production subsystems have passing tests.
It reads the subsystem registry from docs/status/SUBSYSTEM_STATUS.yaml
and runs tests for each production subsystem.

Usage:
    python scripts/ci/check_subsystem_status.py [--verbose] [--subsystem <name>]

Exit codes:
    0 - All production subsystems have passing tests
    1 - One or more production subsystems have failing tests
    2 - Configuration error (e.g., missing registry file)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_subsystem_registry(registry_path: str) -> list[dict[str, Any]]:
    """Load the subsystem registry from YAML file."""
    path = Path(registry_path)
    if not path.exists():
        print(f"ERROR: Subsystem registry not found at {registry_path}")
        sys.exit(2)

    with open(path) as f:
        data = yaml.safe_load(f)

    return data.get("subsystems", [])


def run_tests_for_path(test_path: str, verbose: bool = False) -> tuple[bool, str]:
    """
    Run pytest for a given test path.

    Returns:
        tuple of (success, output)
    """
    if not Path(test_path).exists():
        return True, f"SKIP: Test path {test_path} does not exist"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-q",
        "--tb=no",
        "-W",
        "ignore::DeprecationWarning",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per subsystem
        )
        success = result.returncode == 0

        if verbose:
            output = result.stdout + result.stderr
        else:
            # Just get the summary line
            lines = result.stdout.strip().split("\n")
            output = lines[-1] if lines else ""

        return success, output

    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: Tests in {test_path} exceeded 5 minutes"
    except Exception as e:  # noqa: BLE001 - report any test failure as status
        return False, f"ERROR: {e}"


def check_subsystem_status(
    subsystems: list[dict[str, Any]],
    verbose: bool = False,
    target_subsystem: str | None = None,
) -> bool:
    """
    Check status of all production subsystems.

    Args:
        subsystems: List of subsystem definitions from registry
        verbose: Whether to print detailed output
        target_subsystem: If set, only check this specific subsystem

    Returns:
        True if all production subsystems pass, False otherwise
    """
    all_passed = True
    production_count = 0
    passed_count = 0

    print("=" * 60)
    print("Subsystem Status Check")
    print("=" * 60)

    for subsystem in subsystems:
        name = subsystem.get("name", "unknown")
        status = subsystem.get("status", "alpha")
        test_paths = subsystem.get("test_paths", [])

        # Skip if targeting a specific subsystem
        if target_subsystem and name != target_subsystem:
            continue

        # Only enforce production subsystems
        if status != "production":
            if verbose:
                print(f"\n[{status.upper()}] {name}: Not enforced (status={status})")
            continue

        production_count += 1
        print(f"\n[PROD] {name}:", end=" ")

        if not test_paths:
            print("WARNING: No test paths defined")
            continue

        # Run tests for all defined paths
        path_results = []
        for test_path in test_paths:
            success, output = run_tests_for_path(test_path, verbose)
            path_results.append((test_path, success, output))

        # Check if all paths passed
        subsystem_passed = all(result[1] for result in path_results)

        if subsystem_passed:
            passed_count += 1
            print("PASS")
            if verbose:
                for test_path, _success, output in path_results:
                    print(f"  ✓ {test_path}: {output}")
        else:
            all_passed = False
            print("FAIL")
            for test_path, success, output in path_results:
                status_symbol = "✓" if success else "✗"
                print(f"  {status_symbol} {test_path}: {output}")

    print("\n" + "=" * 60)
    print(f"Summary: {passed_count}/{production_count} production subsystems passing")
    print("=" * 60)

    return all_passed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check that production subsystems have passing tests"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed test output",
    )
    parser.add_argument(
        "--subsystem",
        type=str,
        default=None,
        help="Check only this specific subsystem",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="docs/status/SUBSYSTEM_STATUS.yaml",
        help="Path to subsystem registry YAML file",
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.parent
    os.chdir(project_root)

    # Load registry
    subsystems = load_subsystem_registry(args.registry)

    if not subsystems:
        print("ERROR: No subsystems defined in registry")
        return 2

    # Check status
    all_passed = check_subsystem_status(
        subsystems,
        verbose=args.verbose,
        target_subsystem=args.subsystem,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
