#!/usr/bin/env python3
"""Script to add Qt system dependencies to robotics_env."""

import logging
import os
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)


def create_qt_dockerfile() -> str:
    """Create Dockerfile to add Qt system dependencies."""
    dockerfile_content = """# Add Qt system dependencies to robotics_env
FROM robotics_env:latest

# Install Qt system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libxrender1 \\
    libxrandr2 \\
    libxss1 \\
    libxcursor1 \\
    libxcomposite1 \\
    libasound2 \\
    libxi6 \\
    libxtst6 \\
    libqt6gui6 \\
    libqt6widgets6 \\
    libqt6core6 \\
    qt6-qpa-plugins \\
    && rm -rf /var/lib/apt/lists/*

# Set Qt platform plugin path
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt6/plugins
ENV QT_QPA_PLATFORM=offscreen

# Install PyQt6 with all components
RUN /opt/mujoco-env/bin/pip install "PyQt6>=6.6.0" "PyQt6-Qt6>=6.6.0"
"""
    return dockerfile_content


def update_robotics_env_qt() -> bool:
    """Update robotics_env with Qt dependencies."""
    logger.info("🎨 Adding Qt dependencies to robotics_env...")

    with tempfile.TemporaryDirectory() as temp_dir:
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")

        with open(dockerfile_path, "w") as f:
            f.write(create_qt_dockerfile())

        logger.info("📝 Created Qt Dockerfile: %s", dockerfile_path)

        cmd = ["docker", "build", "-t", "robotics_env", "."]

        try:
            logger.info("🚀 Running: %s", " ".join(cmd))
            logger.info("📦 Installing Qt system libraries and PyQt6...")

            subprocess.run(cmd, cwd=temp_dir, check=True, text=True)

            logger.info("✅ Successfully added Qt dependencies to robotics_env!")
            return True

        except subprocess.CalledProcessError as e:
            logger.error("❌ Failed to update robotics_env: %s", e)
            return False


def test_qt_environment() -> bool:
    """Test Qt functionality in the updated environment."""
    logger.info("\n🧪 Testing Qt environment...")

    try:
        # Test PyQt6 import (headless)
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                "QT_QPA_PLATFORM=offscreen",
                "robotics_env",
                "python",
                "-c",
                "from PyQt6 import QtWidgets, QtCore; "
                "print('✅ PyQt6 imports successfully')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info("%s", result.stdout.strip())

        # Test creating a QApplication (headless)
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                "QT_QPA_PLATFORM=offscreen",
                "robotics_env",
                "python",
                "-c",
                "from PyQt6.QtWidgets import QApplication; "
                "app = QApplication([]); "
                "print('✅ QApplication created successfully')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info("%s", result.stdout.strip())

        return True

    except subprocess.CalledProcessError as e:
        logger.error("❌ Qt test failed: %s", e.stderr)
        return False


def main() -> int:
    """Main function."""
    logger.info("🤖 Qt Dependencies Installer for Robotics Environment")
    logger.info("%s", "=" * 60)

    success = update_robotics_env_qt()

    if success:
        test_success = test_qt_environment()

        if test_success:
            logger.info("\n🎉 Success! PyQt6 is now fully functional in robotics_env.")
            logger.info("💡 MuJoCo GUI simulations should now work properly!")
        else:
            logger.error("\n⚠️  Qt installed but tests failed. May work in GUI mode.")
    else:
        logger.error("\n💥 Failed to install Qt dependencies.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
