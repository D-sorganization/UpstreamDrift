#!/usr/bin/env python3
"""
Physics Validation Runner
-------------------------
Automated script to run physics validation tests within the Golf Modeling Suite.
Designed to be run from CI/CD or inside a Docker container.

Usage:
    python scripts/validate_physics.py \
        [--engine {mujoco,drake,pinocchio,all}] \
        [--type {analytical,complex,all}]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PhysicsValidator")

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_test_files(test_type: str) -> list[str]:
    """Return list of test files based on filter."""
    base_dir = REPO_ROOT / "tests" / "physics_validation"

    files = []

    if test_type in ["analytical", "all"]:
        files.append(str(base_dir / "test_energy_conservation.py"))
        files.append(str(base_dir / "test_pendulum_accuracy.py"))

    if test_type in ["complex", "all"]:
        files.append(str(base_dir / "test_complex_models.py"))

    return files


def run_tests(engine_filter: str, test_type: str) -> bool:
    """Run pytest on selected tests."""
    logger.info(
        f"Starting Physics Validation (Engine: {engine_filter}, Type: {test_type})"
    )

    # Ensure PYTHONPATH includes repo root
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{pythonpath}"

    test_files = get_test_files(test_type)

    if not test_files:
        logger.warning("No test files found for the given criteria.")
        return False

    # Build command
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(test_files)
    cmd.append("-v")  # Verbose

    # If engine filter is specific, we might skip tests
    # (handled by valid skipping in pytest files)
    # But checking 'is_engine_available' inside tests is the robust way.
    # We can pass markers if we had them, but for now we rely on the internal skips.

    logger.info(f"Executing: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=REPO_ROOT,
            capture_output=False,  # Stream directly to stdout/stderr
            check=False
        )

        if result.returncode == 0:
            logger.info("OK Physics Validation PASSED")
            return True
        else:
            logger.error(
                f"X Physics Validation FAILED (Exit Code: {result.returncode})"
            )
            return False

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False


def main():
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
