#!/usr/bin/env python3
"""Test script to run inside Docker container to diagnose dependency issues."""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def test_python_environment() -> None:
    """Test the Python environment and paths."""
    logger.info("🐍 Python Environment Diagnostics")
    logger.info("%s", "=" * 50)

    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version)
    logger.info("Python path: %s", sys.path)

    # Check if we're using the venv
    if "/opt/mujoco-env" in sys.executable:
        logger.info("✓ Using MuJoCo virtual environment")
    else:
        logger.info("⚠️  NOT using MuJoCo virtual environment!")
        logger.info(
            "   This might be the issue - dependencies installed in /opt/mujoco-env"
        )

    logger.info("")


def test_pip_list() -> None:
    """Show all installed packages."""
    logger.info("📦 Installed Packages")
    logger.info("%s", "=" * 50)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("%s", result.stdout)
    except (OSError, ValueError) as e:
        logger.error("❌ Failed to get pip list: %s", e)

    logger.info("")


def test_specific_imports() -> bool:
    """Test specific problematic imports."""
    logger.info("🔍 Testing Specific Imports")
    logger.info("%s", "=" * 50)

    # Test defusedxml specifically
    try:
        import defusedxml

        logger.info("✓ defusedxml available at: %s", defusedxml.__file__)
        logger.info("  Version: %s", getattr(defusedxml, "__version__", "unknown"))
    except ImportError as e:
        logger.info("❌ defusedxml missing: %s", e)
        return False

    # Test defusedxml.ElementTree specifically
    try:
        import importlib.util

        if importlib.util.find_spec("defusedxml.ElementTree"):
            logger.info("✓ defusedxml.ElementTree available")
        else:
            raise ImportError("defusedxml.ElementTree not found")
    except ImportError as e:
        logger.info("❌ defusedxml.ElementTree missing: %s", e)
        return False

    # Test the problematic urdf_io module
    try:
        # Add the python directory to path
        python_dir = "/workspace/python"
        if python_dir not in sys.path:
            logger.info("  Added %s to Python path", python_dir)

        if importlib.util.find_spec("mujoco_humanoid_golf.urdf_io"):
            logger.info("✓ mujoco_humanoid_golf.urdf_io imported successfully")
        else:
            raise ImportError("mujoco_humanoid_golf.urdf_io not found")
    except ImportError as e:
        logger.error("❌ mujoco_humanoid_golf.urdf_io failed: %s", e)
        return False

    # Test the main module
    try:
        if importlib.util.find_spec("mujoco_humanoid_golf"):
            logger.info("✓ mujoco_humanoid_golf module imported successfully")
        else:
            raise ImportError("mujoco_humanoid_golf not found")
    except ImportError as e:
        logger.error("❌ mujoco_humanoid_golf module failed: %s", e)
        return False

    return True


def test_environment_activation() -> None:
    """Test if the virtual environment is properly activated."""
    logger.info("🔧 Environment Activation Test")
    logger.info("%s", "=" * 50)

    # Check environment variables
    virtual_env = os.environ.get("VIRTUAL_ENV")
    path = os.environ.get("PATH", "")

    logger.info("VIRTUAL_ENV: %s", virtual_env)
    logger.info("PATH contains /opt/mujoco-env: %s", "/opt/mujoco-env" in path)

    # Check if the venv python is first in PATH
    which_python = subprocess.run(["which", "python3"], capture_output=True, text=True)
    logger.info("which python3: %s", which_python.stdout.strip())

    # Test if we can import from the venv
    venv_python = "/opt/mujoco-env/bin/python"
    if os.path.exists(venv_python):
        logger.info("✓ Virtual environment python exists: %s", venv_python)

        # Test importing defusedxml with the venv python
        try:
            result = subprocess.run(
                [venv_python, "-c", "import defusedxml; print('defusedxml OK')"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(
                "✓ defusedxml works with venv python: %s", result.stdout.strip()
            )
        except subprocess.CalledProcessError as e:
            logger.info("❌ defusedxml fails with venv python: %s", e.stderr)
    else:
        logger.info("❌ Virtual environment python not found: %s", venv_python)


def main() -> int:
    """Main diagnostic function."""
    logger.info("🔬 Docker Container Dependency Diagnostics")
    logger.info("%s", "=" * 60)
    logger.info("")

    test_python_environment()
    test_environment_activation()
    test_pip_list()
    success = test_specific_imports()

    logger.info("%s", "=" * 60)
    if success:
        logger.info("✅ All tests passed! Dependencies should work.")
    else:
        logger.error("❌ Some tests failed. Check the output above for issues.")
        logger.info("")
        logger.info("💡 Possible solutions:")
        logger.info("   1. Make sure Docker container uses: /opt/mujoco-env/bin/python")
        logger.info(
            "   2. Activate virtual environment: source /opt/mujoco-env/bin/activate"
        )
        logger.info("   3. Rebuild Docker image if dependencies are missing")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
