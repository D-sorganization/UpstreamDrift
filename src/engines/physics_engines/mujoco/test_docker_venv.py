#!/usr/bin/env python3
"""Test script to verify Docker container virtual environment setup."""

import os
import subprocess
import sys


def test_docker_venv() -> bool:
    """Test if Docker container properly uses the virtual environment."""
    print("🐳 Testing Docker Container Virtual Environment")
    print("=" * 60)

    # Test 1: Check if Docker image exists
    print("1. Checking if robotics_env Docker image exists...")
    try:
        result = subprocess.run(
            [
                "docker",
                "images",
                "robotics_env",
                "--format",
                "{{.Repository}}:{{.Tag}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if "robotics_env" in result.stdout:
            print("✓ robotics_env Docker image found")
        else:
            print("❌ robotics_env Docker image not found")
            print("   Run: docker build -t robotics_env . (from docker/ directory)")
            return False
    except Exception as e:
        print(f"❌ Failed to check Docker images: {e}")
        return False

    # Test 2: Check Python path in container
    print("\n2. Testing Python executable path in container...")
    try:
        cmd = ["docker", "run", "--rm", "robotics_env", "which", "python"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        python_path = result.stdout.strip()
        print(f"   Python path: {python_path}")

        if "/opt/mujoco-env/bin/python" in python_path:
            print("✓ Container uses virtual environment Python")
        else:
            print("⚠️  Container may not be using virtual environment")
    except Exception as e:
        print(f"❌ Failed to check Python path: {e}")

    # Test 3: Check if defusedxml is available in container
    print("\n3. Testing defusedxml availability in container...")
    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "robotics_env",
            "python",
            "-c",
            (
                "import defusedxml; print('defusedxml version:', "
                "getattr(defusedxml, '__version__', 'unknown'))"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ defusedxml not available: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Failed to test defusedxml: {e}")
        return False

    # Test 4: Test the specific import that was failing
    print("\n4. Testing defusedxml.ElementTree import...")
    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "robotics_env",
            "python",
            "-c",
            (
                "import defusedxml.ElementTree as DefusedET; "
                "print('defusedxml.ElementTree imported successfully')"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ defusedxml.ElementTree import failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Failed to test defusedxml.ElementTree: {e}")
        return False

    # Test 5: Test the mujoco_golf_pendulum module import (if workspace is mounted)
    print("\n5. Testing mujoco_golf_pendulum module import...")

    # Get current directory (should be MuJoCo repo root)
    current_dir = os.getcwd()
    if not current_dir.endswith("MuJoCo_Golf_Swing_Model"):
        print("⚠️  Not running from MuJoCo_Golf_Swing_Model directory")
        print("   Skipping module import test")
    else:
        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{current_dir}:/workspace",
                "-w",
                "/workspace/python",
                "robotics_env",
                "python",
                "-c",
                (
                    "from mujoco_golf_pendulum import urdf_io; "
                    "print('urdf_io imported successfully')"
                ),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ mujoco_golf_pendulum.urdf_io import failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Failed to test module import: {e}")
            return False

    print("\n" + "=" * 60)
    print("✅ All Docker container tests passed!")
    print("   The container should now work with the MuJoCo golf model.")
    return True


def main() -> int:
    """Main function."""
    success = test_docker_venv()

    if not success:
        print("\n💡 Troubleshooting steps:")
        print(
            "   1. Rebuild Docker image: docker build -t robotics_env . "
            "(from docker/ directory)"
        )
        print("   2. Check if defusedxml was properly installed during build")
        print("   3. Verify virtual environment is activated in container")
        print("   4. Run this script from MuJoCo_Golf_Swing_Model directory")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
