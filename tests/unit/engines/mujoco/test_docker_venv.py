#!/usr/bin/env python3
"""Test script to verify Docker container virtual environment setup."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _check_docker_image_exists() -> bool:
    """Check whether the upstream-drift Docker image is available."""
    logger.debug("1. Checking if upstream-drift Docker image exists...")
    try:
        result = subprocess.run(
            [
                "docker",
                "images",
                "upstream-drift:engine",
                "--format",
                "{{.Repository}}:{{.Tag}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if "upstream-drift:engine" in result.stdout:
            logger.info("✓ upstream-drift Docker image found")
            return True
        logger.warning("❌ upstream-drift Docker image not found")
        logger.info(
            "   Run: docker build -t upstream-drift:engine . (from docker/ directory)",
        )
        return False
    except (subprocess.CalledProcessError, OSError) as e:
        logger.error(f"❌ Failed to check Docker images: {e}")
        return False


def _check_python_path() -> None:
    """Verify the container uses the virtual environment Python."""
    logger.info("\n2. Testing Python executable path in container...")
    try:
        cmd = ["docker", "run", "--rm", "upstream-drift:engine", "which", "python"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        python_path = result.stdout.strip()
        logger.info(f"   Python path: {python_path}")

        if "/opt/mujoco-env/bin/python" in python_path:
            logger.info("✓ Container uses virtual environment Python")
        else:
            logger.info("⚠️  Container may not be using virtual environment")
    except (subprocess.CalledProcessError, OSError) as e:
        logger.error(f"❌ Failed to check Python path: {e}")


def _check_defusedxml() -> bool:
    """Test that the defusedxml package is importable inside the container."""
    logger.info("\n3. Testing defusedxml availability in container...")
    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "upstream-drift:engine",
            "python",
            "-c",
            (
                "import defusedxml; print('defusedxml version:', "
                "getattr(defusedxml, '__version__', 'unknown'))"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"❌ defusedxml not available: {e.stderr}")
        return False
    except OSError as e:
        logger.error(f"❌ Failed to test defusedxml: {e}")
        return False


def _check_defusedxml_elementtree() -> bool:
    """Test that defusedxml.ElementTree can be imported in the container."""
    logger.info("\n4. Testing defusedxml.ElementTree import...")
    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "upstream-drift:engine",
            "python",
            "-c",
            (
                "import defusedxml.ElementTree as DefusedET; "
                "print('defusedxml.ElementTree imported successfully')"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ defusedxml.ElementTree import failed: {e.stderr}")
        return False
    except OSError as e:
        logger.error(f"❌ Failed to test defusedxml.ElementTree: {e}")
        return False


def _check_module_import() -> bool:
    """Test that the mujoco_golf_pendulum module can be imported in the container."""
    logger.info("\n5. Testing mujoco_golf_pendulum module import...")
    current_dir = os.getcwd()
    if not current_dir.endswith("MuJoCo_Golf_Swing_Model"):
        logger.info("⚠️  Not running from MuJoCo_Golf_Swing_Model directory")
        logger.warning("   Skipping module import test")
        return True

    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{current_dir}:/workspace",
            "-w",
            "/workspace/python",
            "upstream-drift:engine",
            "python",
            "-c",
            (
                "from mujoco_golf_pendulum import urdf_io; "
                "print('urdf_io imported successfully')"
            ),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ mujoco_golf_pendulum.urdf_io import failed: {e.stderr}")
        return False
    except OSError as e:
        logger.error(f"❌ Failed to test module import: {e}")
        return False


def test_docker_venv() -> bool:
    """Test if Docker container properly uses the virtual environment."""
    logger.info("🐳 Testing Docker Container Virtual Environment")
    logger.info("=" * 60)

    if not _check_docker_image_exists():
        return False

    _check_python_path()

    if not _check_defusedxml():
        return False

    if not _check_defusedxml_elementtree():
        return False

    if not _check_module_import():
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
            "   1. Rebuild Docker image: docker build -t upstream-drift:engine . "
            "(from docker/ directory)",
        )
        logger.info("   2. Check if defusedxml was properly installed during build")
        logger.info("   3. Verify virtual environment is activated in container")
        logger.info("   4. Run this script from MuJoCo_Golf_Swing_Model directory")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
