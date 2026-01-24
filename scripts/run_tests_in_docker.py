"""
Dockerized Physics Verification Runner.

This script builds the Golf Modeling Suite development container (if needed)
and runs the physics verification suite inside it to ensure environment consistency.

Refactored to use shared script utilities (DRY principle).
"""

import sys

from scripts.script_utils import get_repo_root, run_command, setup_script_logging

logger = setup_script_logging("DockerRunner")


def main() -> None:
    """Main execution flow."""
    root_dir = get_repo_root()
    dockerfile_path = (
        root_dir / "src" / "engines" / "physics_engines" / "mujoco" / "Dockerfile"
    )

    if not dockerfile_path.exists():
        logger.error(f"Dockerfile not found at {dockerfile_path}")
        sys.exit(1)

    image_name = "golf-suite-dev"

    logger.info(f"Building Docker Image: {image_name}")
    # Build image
    build_cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        str(dockerfile_path),
        str(root_dir),
    ]
    result = run_command(build_cmd, cwd=root_dir, check=True, logger=logger)
    if result.returncode != 0:
        logger.error("Docker build failed")
        sys.exit(result.returncode)

    logger.info("Running Verification Suite in Docker")
    # Run container
    # Mount root_dir to /workspace
    # Workdir /workspace
    # Command: python scripts/verify_physics.py
    run_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{root_dir}:/workspace",
        "-w",
        "/workspace",
        image_name,
        "python",
        "scripts/verify_physics.py",
    ]

    result = run_command(run_cmd, logger=logger)
    if result.returncode != 0:
        logger.error("Verification failed")
        sys.exit(result.returncode)

    logger.info("Verification completed successfully")


if __name__ == "__main__":
    main()
