#!/usr/bin/env python3
"""Test script to check if all required dependencies are available."""

import sys


def test_imports():
    """Test all critical imports for the MuJoCo golf model."""
    print("Testing critical dependencies...")

    # Test basic dependencies
    import importlib.util

    if importlib.util.find_spec("numpy") is not None:
        print("✓ numpy available")
    else:
        print("✗ numpy missing")
        return False

    if importlib.util.find_spec("mujoco") is not None:
        print("✓ mujoco available")
    else:
        print("✗ mujoco missing")
        return False

    # Test the problematic dependency
    if importlib.util.find_spec("defusedxml") is not None:
        print("✓ defusedxml available")
    else:
        print("✗ defusedxml missing")
        print("  This is likely the cause of the launch failure!")
        return False

    # Test PyQt6
    if importlib.util.find_spec("PyQt6.QtWidgets") is not None:
        print("✓ PyQt6 available")
    else:
        print("✗ PyQt6 missing")
        return False

    # Test the module that uses defusedxml
    if importlib.util.find_spec("mujoco_humanoid_golf.urdf_io") is not None:
        print("✓ mujoco_humanoid_golf.urdf_io available")
    else:
        print("✗ mujoco_humanoid_golf.urdf_io missing")
        return False

    # Test the main module
    if importlib.util.find_spec("mujoco_humanoid_golf") is not None:
        print("✓ mujoco_humanoid_golf module available")
    else:
        print("✗ mujoco_humanoid_golf module missing")
        return False

    print("\n✓ All dependencies are available!")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
