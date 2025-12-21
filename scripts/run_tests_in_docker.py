"""
Dockerized Physics Verification Runner.

This script builds the Golf Modeling Suite development container (if needed)
and runs the physics verification suite inside it to ensure environment consistency.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command and stream output."""
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        sys.exit(e.returncode)


def main() -> None:
    """Main execution flow."""
    root_dir = Path(__file__).parent.parent.resolve()
    dockerfile_path = root_dir / "engines" / "physics_engines" / "mujoco" / "Dockerfile"

    if not dockerfile_path.exists():
        print(f"Error: Dockerfile not found at {dockerfile_path}")
        sys.exit(1)

    image_name = "golf-suite-dev"

    print(f"--- Building Docker Image: {image_name} ---")
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
    run_command(build_cmd, cwd=root_dir)

    print("\n--- Running Verification Suite in Docker ---")
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

    print(f"Executing: {' '.join(run_cmd)}")
    run_command(run_cmd)


if __name__ == "__main__":
    main()
