#!/usr/bin/env python3
"""Script to rebuild the Docker image with updated dependencies.

Refactored to use shared script utilities (DRY principle).
"""

import sys
from pathlib import Path

# Add repo root to path for imports
_repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_repo_root))

from scripts.script_utils import run_command, setup_script_logging  # noqa: E402

logger = setup_script_logging("DockerRebuilder")


def rebuild_docker_image() -> bool:
    """Rebuild the robotics_env Docker image."""
    logger.info("Rebuilding Docker image with updated dependencies...")

    # Get the directory containing this script (should be mujoco engine root)
    script_dir = Path(__file__).parent
    docker_dir = script_dir / "docker"

    if not docker_dir.exists():
        logger.error(f"Docker directory not found at {docker_dir}")
        return False

    logger.info(f"Building from: {docker_dir}")

    # Build command
    cmd = ["docker", "build", "-t", "robotics_env", "."]

    logger.info("This may take several minutes...")

    result = run_command(cmd, cwd=docker_dir, logger=logger)

    if result.returncode == 0:
        logger.info("Docker image rebuilt successfully!")
        logger.info("You can now run simulations with the updated dependencies.")
        return True
    else:
        logger.error(f"Docker build failed with exit code {result.returncode}")
        return False


def main() -> int:
    """Main entry point."""
    logger.info("MuJoCo Golf Model - Docker Image Rebuilder")
    logger.info("=" * 50)

    success = rebuild_docker_image()

    if success:
        logger.info("Build completed successfully!")
        logger.info("You can now launch simulations from the GUI.")
    else:
        logger.error("Build failed! Check the error messages above for details.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
