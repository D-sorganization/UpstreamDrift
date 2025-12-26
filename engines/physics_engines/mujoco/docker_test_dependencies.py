#!/usr/bin/env python3
"""Test script to run inside Docker container to diagnose dependency issues."""

import os
import subprocess
import sys


def test_python_environment():
    """Test the Python environment and paths."""
    print("🐍 Python Environment Diagnostics")
    print("=" * 50)

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")

    # Check if we're using the venv
    if "/opt/mujoco-env" in sys.executable:
        print("✓ Using MuJoCo virtual environment")
    else:
        print("⚠️  NOT using MuJoCo virtual environment!")
        print("   This might be the issue - dependencies installed in /opt/mujoco-env")

    print()


def test_pip_list():
    """Show all installed packages."""
    print("📦 Installed Packages")
    print("=" * 50)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except Exception as e:
        print(f"❌ Failed to get pip list: {e}")

    print()


def test_specific_imports():
    """Test specific problematic imports."""
    print("🔍 Testing Specific Imports")
    print("=" * 50)

    # Test defusedxml specifically
    try:
        import defusedxml

        print(f"✓ defusedxml available at: {defusedxml.__file__}")
        print(f"  Version: {getattr(defusedxml, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"❌ defusedxml missing: {e}")
        return False

    # Test defusedxml.ElementTree specifically
    try:
        import importlib.util

        if importlib.util.find_spec("defusedxml.ElementTree"):
            print("✓ defusedxml.ElementTree available")
        else:
            raise ImportError("defusedxml.ElementTree not found")
    except ImportError as e:
        print(f"❌ defusedxml.ElementTree missing: {e}")
        return False

    # Test the problematic urdf_io module
    try:
        # Add the python directory to path
        python_dir = "/workspace/python"
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
            print(f"  Added {python_dir} to Python path")

        if importlib.util.find_spec("mujoco_humanoid_golf.urdf_io"):
            print("✓ mujoco_humanoid_golf.urdf_io imported successfully")
        else:
            raise ImportError("mujoco_humanoid_golf.urdf_io not found")
    except ImportError as e:
        print(f"❌ mujoco_humanoid_golf.urdf_io failed: {e}")
        return False

    # Test the main module
    try:
        if importlib.util.find_spec("mujoco_humanoid_golf"):
            print("✓ mujoco_humanoid_golf module imported successfully")
        else:
            raise ImportError("mujoco_humanoid_golf not found")
    except ImportError as e:
        print(f"❌ mujoco_humanoid_golf module failed: {e}")
        return False

    return True


def test_environment_activation():
    """Test if the virtual environment is properly activated."""
    print("🔧 Environment Activation Test")
    print("=" * 50)

    # Check environment variables
    virtual_env = os.environ.get("VIRTUAL_ENV")
    path = os.environ.get("PATH", "")

    print(f"VIRTUAL_ENV: {virtual_env}")
    print(f"PATH contains /opt/mujoco-env: {'/opt/mujoco-env' in path}")

    # Check if the venv python is first in PATH
    which_python = subprocess.run(["which", "python3"], capture_output=True, text=True)
    print(f"which python3: {which_python.stdout.strip()}")

    # Test if we can import from the venv
    venv_python = "/opt/mujoco-env/bin/python"
    if os.path.exists(venv_python):
        print(f"✓ Virtual environment python exists: {venv_python}")

        # Test importing defusedxml with the venv python
        try:
            result = subprocess.run(
                [venv_python, "-c", "import defusedxml; print('defusedxml OK')"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"✓ defusedxml works with venv python: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ defusedxml fails with venv python: {e.stderr}")
    else:
        print(f"❌ Virtual environment python not found: {venv_python}")


def main():
    """Main diagnostic function."""
    print("🔬 Docker Container Dependency Diagnostics")
    print("=" * 60)
    print()

    test_python_environment()
    test_environment_activation()
    test_pip_list()
    success = test_specific_imports()

    print("=" * 60)
    if success:
        print("✅ All tests passed! Dependencies should work.")
    else:
        print("❌ Some tests failed. Check the output above for issues.")
        print()
        print("💡 Possible solutions:")
        print("   1. Make sure Docker container uses: /opt/mujoco-env/bin/python")
        print("   2. Activate virtual environment: source /opt/mujoco-env/bin/activate")
        print("   3. Rebuild Docker image if dependencies are missing")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
