#!/usr/bin/env python3

"""Golf Modeling Suite - Comprehensive Validation Script.



This script validates that all components of the Golf Modeling Suite

are properly migrated and functional.

"""

import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_ROOT = _PROJECT_ROOT / "src"

# Ensure the project root is on sys.path so ``src`` is importable.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Some src/ modules do a bare ``import bunkershot3d`` (not ``src.bunkershot3d``),
# which only resolves when src/ itself -- not just the project root -- is on
# sys.path. Without this, validate_launchers() fails with
# "No module named 'bunkershot3d'" even though the launcher code is fine.
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.shared.python.data_io.path_utils import get_src_root  # noqa: E402

SUITE_ROOT = get_src_root()


from src.shared.python.data_io.common_utils import setup_logging  # noqa: E402

logger = setup_logging(__name__)


def validate_directory_structure() -> bool:
    """Validate that all expected directories exist."""

    logger.info("Validating directory structure...")

    suite_root = SUITE_ROOT

    # Paths under src/ (the migrated layout).
    expected_src_dirs = [
        "engines/Simscape_Multibody_Models/2D_Golf_Model",
        "engines/Simscape_Multibody_Models/3D_Golf_Model",
        "engines/physics_engines/mujoco",
        "engines/physics_engines/drake",
        "engines/physics_engines/pinocchio",
        "engines/pendulum_models",
        "launchers",
        "shared/python",
        "shared/matlab",
        "tools",
    ]
    # docs/ and output/ live at the project root, not under src/ -- they
    # were never part of the migrated src/ layout (issue #8834).
    expected_root_dirs = [
        "docs",
        "output",
    ]

    missing_dirs = []

    for dir_path in expected_src_dirs:
        full_path = suite_root / dir_path

        if not full_path.exists():
            missing_dirs.append(dir_path)

        else:
            logger.info(f"  ✅ {dir_path}")

    for dir_path in expected_root_dirs:
        full_path = _PROJECT_ROOT / dir_path

        if not full_path.exists():
            missing_dirs.append(dir_path)

        else:
            logger.info(f"  ✅ {dir_path}")

    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")

        return False

    logger.info("Directory structure validation: PASSED")

    return True


def validate_launchers() -> bool:
    """Validate that launchers can be imported and have required functions."""

    logger.info("Validating launchers...")

    try:
        # Test classic GUI launcher

        from src.launchers.upstream_drift_launcher import main as classic_main

        _ = classic_main

        logger.info("  ✅ Classic GUI launcher imports successfully")

        # Test canonical launcher

        spec = importlib.util.spec_from_file_location(
            "launch_upstream_drift", _PROJECT_ROOT / "launch_upstream_drift.py"
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                "Could not load spec or loader for launch_upstream_drift.py"
            )

        launch_module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(launch_module)

        logger.info("  ✅ Main launcher script loads successfully")

    except ImportError as e:
        logger.error(f"Launcher import error: {e}")

        return False

    except (OSError, ValueError, RuntimeError, AttributeError, TypeError) as e:
        logger.error(f"Launcher validation error: {e}")

        return False

    logger.info("Launcher validation: PASSED")

    return True


def validate_shared_components() -> bool:
    """Validate shared Python and MATLAB components."""

    logger.info("Validating shared components...")

    try:
        # Test shared Python utilities

        from src.shared.python.data_io.common_utils import (
            convert_units,
            ensure_output_dir,
            load_golf_data,
            setup_logging,
            standardize_joint_angles,
        )

        # Silence unused import warnings by using them

        _ = (
            convert_units,
            ensure_output_dir,
            load_golf_data,
            setup_logging,
            standardize_joint_angles,
        )

        logger.info("  ✅ Shared Python utilities import successfully")

        # Test shared constants

        # NOTE: aliased to avoid shadowing the module-level SUITE_ROOT (this
        # repo's src/ root) with core.constants.SUITE_ROOT, which names a
        # different, more narrowly-scoped path (src/shared) -- shadowing it
        # here previously broke the MATLAB file check below (issue #8834).
        from src.shared.python.core import (
            DRAKE_ROOT,
            ENGINES_ROOT,
            MUJOCO_ROOT,
            PINOCCHIO_ROOT,
            SUITE_ROOT as _CORE_SUITE_ROOT,
        )

        _ = (DRAKE_ROOT, ENGINES_ROOT, MUJOCO_ROOT, PINOCCHIO_ROOT, _CORE_SUITE_ROOT)

        logger.info("  ✅ Shared constants available")

        # Check MATLAB files exist

        matlab_files = [
            "shared/matlab/setup_golf_suite.m",
            "shared/matlab/golf_suite_help.m",
        ]

        suite_root = SUITE_ROOT

        for matlab_file in matlab_files:
            if not (suite_root / matlab_file).exists():
                logger.error(f"Missing MATLAB file: {matlab_file}")

                return False

            logger.info(f"  ✅ {matlab_file}")

    except ImportError as e:
        logger.error(f"Shared component import error: {e}")

        return False

    except (OSError, ValueError, RuntimeError, AttributeError, TypeError) as e:
        logger.error(f"Shared component validation error: {e}")

        return False

    logger.info("Shared components validation: PASSED")

    return True


def validate_engine_structure() -> bool:
    """Validate that all physics engines have expected structure."""

    logger.info("Validating engine structure...")

    suite_root = SUITE_ROOT

    engines = {
        "MuJoCo": "engines/physics_engines/mujoco",
        "Drake": "engines/physics_engines/drake",
        "Pinocchio": "engines/physics_engines/pinocchio",
        "2D MATLAB": "engines/Simscape_Multibody_Models/2D_Golf_Model",
        "3D MATLAB": "engines/Simscape_Multibody_Models/3D_Golf_Model",
        "Pendulum": "engines/pendulum_models",
    }

    for engine_name, engine_path in engines.items():
        full_path = suite_root / engine_path

        if not full_path.exists():
            logger.error(f"Missing engine: {engine_name} at {engine_path}")

            return False

        # Check for key files/directories

        if "matlab" in engine_path.lower():
            # MATLAB engines should have matlab/ subdirectory

            if not (full_path / "matlab").exists():
                logger.warning(f"MATLAB engine {engine_name} missing matlab/ directory")

        else:
            # Python engines should have python/ subdirectory

            if not (full_path / "python").exists():
                logger.warning(f"Python engine {engine_name} missing python/ directory")

        logger.info(f"  ✅ {engine_name}: {engine_path}")

    logger.info("Engine structure validation: PASSED")

    return True


def validate_git_repository() -> bool:
    """Validate that this is a proper Git repository."""

    logger.info("Validating Git repository...")

    # .git/ lives at the project root, not under src/ (issue #8834).
    project_root = _PROJECT_ROOT

    git_dir = project_root / ".git"

    if not git_dir.exists():
        logger.error("Not a Git repository - missing .git directory")

        return False

    # Check for key Git files

    git_files = [".git/config", ".git/HEAD", ".gitignore"]

    for git_file in git_files:
        if not (project_root / git_file).exists():
            logger.error(f"Missing Git file: {git_file}")

            return False

    logger.info("  ✅ Git repository structure valid")

    logger.info("Git repository validation: PASSED")

    return True


def validate_configuration_files() -> bool:
    """Validate that all configuration files are present."""

    logger.info("Validating configuration files...")

    # .gitignore, README.md, and docs/ live at the project root, not under
    # src/ (issue #8834).
    project_root = _PROJECT_ROOT

    config_files = [
        ".gitignore",
        "README.md",
        "docs/plans/REMEDIATION_PLAN.md",
    ]

    for config_file in config_files:
        if not (project_root / config_file).exists():
            logger.error(f"Missing configuration file: {config_file}")

            return False

        logger.info(f"  ✅ {config_file}")

    logger.info("Configuration files validation: PASSED")

    return True


def run_comprehensive_validation() -> bool:
    """Run all validation tests."""

    logger.info("Starting comprehensive Golf Modeling Suite validation...")

    logger.info("=" * 60)

    validations = [
        ("Directory Structure", validate_directory_structure),
        ("Launchers", validate_launchers),
        ("Shared Components", validate_shared_components),
        ("Engine Structure", validate_engine_structure),
        ("Git Repository", validate_git_repository),
        ("Configuration Files", validate_configuration_files),
    ]

    results = {}

    for name, validation_func in validations:
        try:
            results[name] = validation_func()

        except (OSError, ValueError, RuntimeError, AttributeError, TypeError) as e:
            logger.error(f"Validation {name} failed with exception: {e}")

            results[name] = False

        logger.info("-" * 40)

    # Summary

    logger.info("VALIDATION SUMMARY")

    logger.info("=" * 60)

    passed = 0

    total = len(results)

    for name, result in results.items():
        status = "PASSED" if result else "FAILED"

        emoji = "✅" if result else "❌"

        logger.info(f"{emoji} {name}: {status}")

        if result:
            passed += 1

    logger.info("-" * 40)

    logger.info(f"Overall: {passed}/{total} validations passed")

    if passed == total:
        logger.info("🎉 Golf Modeling Suite validation: ALL TESTS PASSED!")

        logger.info("The suite is ready for development use (Beta)!")

        return True

    logger.error(f"❌ {total - passed} validation(s) failed")

    logger.error("Please address the issues above before using the suite")

    return False


if __name__ == "__main__":
    success = run_comprehensive_validation()

    sys.exit(0 if success else 1)
