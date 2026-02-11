#!/usr/bin/env python3
"""Test script to verify Docker container virtual environment setup."""

import os
import subprocess
import sys
import logging


logger = logging.getLogger(__name__)

def test_docker_venv() -> bool:
    """Test if Docker container properly uses the virtual environment."""
    logger.info("🐳 Testing Docker Container Virtual Environment")
    logger.info("=" * 60)

    # Test 1: Check if Docker image exists
    logger.info("1. Checking if robotics_env Docker image exists...")
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
            logger.info("✓ robotics_env Docker image found")
        else:
            logger.info("❌ robotics_env Docker image not found")
            logger.info("   Run: docker build -t robotics_env . (from docker/ directory)")
            return False
    except Exception as e:
        logger.info(f"❌ Failed to check Docker images: {e}")
        return False

    # Test 2: Check Python path in container
    logger.info("\n2. Testing Python executable path in container...")
    try:
        cmd = ["docker", "run", "--rm", "robotics_env", "which", "python"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        python_path = result.stdout.strip()
        logger.info(f"   Python path: {python_path}")

        if "/opt/mujoco-env/bin/python" in python_path:
            logger.info("✓ Container uses virtual environment Python")
        else:
            logger.info("⚠️  Container may not be using virtual environment")
    except Exception as e:
        logger.info(f"❌ Failed to check Python path: {e}")

    # Test 3: Check if defusedxml is available in container
    logger.info("\n3. Testing defusedxml availability in container...")
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
        logger.info(f"✓ {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logger.info(f"❌ defusedxml not available: {e.stderr}")
        return False
    except Exception as e:
        logger.info(f"❌ Failed to test defusedxml: {e}")
        return False

    # Test 4: Test the specific import that was failing
    logger.info("\n4. Testing defusedxml.ElementTree import...")
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
        logger.info(f"✓ {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logger.info(f"❌ defusedxml.ElementTree import failed: {e.stderr}")
        return False
    except Exception as e:
        logger.info(f"❌ Failed to test defusedxml.ElementTree: {e}")
        return False

    # Test 5: Test the mujoco_golf_pendulum module import (if workspace is mounted)
    logger.info("\n5. Testing mujoco_golf_pendulum module import...")

    # Get current directory (should be MuJoCo repo root)
    current_dir = os.getcwd()
    if not current_dir.endswith("MuJoCo_Golf_Swing_Model"):
        logger.info("⚠️  Not running from MuJoCo_Golf_Swing_Model directory")
        logger.info("   Skipping module import test")
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
            logger.info(f"✓ {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logger.info(f"❌ mujoco_golf_pendulum.urdf_io import failed: {e.stderr}")
            return False
        except Exception as e:
            logger.info(f"❌ Failed to test module import: {e}")
            return False

    logger.info("\n" + "=" * 60)
    logger.info("✅ All Docker container tests passed!")
    logger.info("   The container should now work with the MuJoCo golf model.")
    return True


def main() -> int:
    """Main function."""
    success = test_docker_venv()

    if not success:
        logger.info("\n💡 Troubleshooting steps:")
        logger.info(
            "   1. Rebuild Docker image: docker build -t robotics_env . "
            "(from docker/ directory)"
        )
        logger.info("   2. Check if defusedxml was properly installed during build")
        logger.info("   3. Verify virtual environment is activated in container")
        logger.info("   4. Run this script from MuJoCo_Golf_Swing_Model directory")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
