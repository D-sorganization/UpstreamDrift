#!/usr/bin/env python3
"""
Physics Validation Runner
-------------------------
Automated script to run physics validation tests within the Golf Modeling Suite.
Designed to be run from CI/CD or inside a Docker container.

Refactored to use shared script utilities (DRY principle).

Usage:
    python scripts/validate_physics.py \
        [--engine {mujoco,drake,pinocchio,all}] \
        [--type {analytical,complex,all}]
"""

import argparse
import sys
from pathlib import Path

from scripts.script_utils import get_repo_root, run_pytest, setup_script_logging

logger = setup_script_logging("PhysicsValidator")

# Test file configuration - maps test types to relative paths
TEST_FILES: dict[str, list[str]] = {
    "analytical": [
        "tests/physics_validation/test_energy_conservation.py",
        "tests/physics_validation/test_pendulum_accuracy.py",
    ],
    "complex": [
        "tests/physics_validation/test_complex_models.py",
    ],
}


def get_test_files(test_type: str) -> list[Path]:
    """Return list of test files based on filter.

    Args:
        test_type: Filter for test type (analytical, complex, or all).

    Returns:
        List of test file paths.
    """
    repo_root = get_repo_root()
    files: list[Path] = []

    if test_type == "all":
        types_to_include = list(TEST_FILES.keys())
    else:
        types_to_include = [test_type]

    for t in types_to_include:
        if t in TEST_FILES:
            files.extend(repo_root / f for f in TEST_FILES[t])

    return files


def run_tests(engine_filter: str, test_type: str) -> bool:
    """Run pytest on selected tests.

    Args:
        engine_filter: Engine to target (tests auto-skip if missing).
        test_type: Type of validation tests to run.

    Returns:
        True if tests passed, False otherwise.
    """
    logger.info(
        f"Starting Physics Validation (Engine: {engine_filter}, Type: {test_type})"
    )

    test_files = get_test_files(test_type)

    if not test_files:
        logger.warning("No test files found for the given criteria.")
        return False

    # Engine filtering is handled by skips within pytest files
    # based on 'is_engine_available' checks
    return run_pytest(test_files, verbose=True, logger=logger)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Physics Validation Suite")
    parser.add_argument(
        "--engine",
        choices=["mujoco", "drake", "pinocchio", "all"],
        default="all",
        help="Target specific engine (tests will auto-skip if engine is missing)",
    )
    parser.add_argument(
        "--type",
        choices=["analytical", "complex", "all"],
        default="all",
        help="Type of validation (analytical baselines or complex model stability)",
    )

    args = parser.parse_args()

    success = run_tests(args.engine, args.type)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
