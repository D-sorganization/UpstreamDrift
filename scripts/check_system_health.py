#!/usr/bin/env python3
"""
Golf Modeling Suite - System Health Check
Verifies that all components (Local Python, Docker Images, Libraries)
are installed and functioning.
"""

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def log_result(component: str, status: str, message: str = "") -> None:
    """Log a color-coded health check result line."""
    color = "\033[92m" if status == "OK" else "\033[91m"
    reset = "\033[0m"
    line = f"{component:<30} [{color}{status}{reset}] {message}"
    logger.info(line)


def check_python_module(module_name: str) -> tuple[bool, str]:
    """Check whether a Python module can be imported."""
    try:
        __import__(module_name)
        return True, ""
    except ImportError as e:
        return False, str(e)
    except (RuntimeError, OSError) as e:
        return False, f"Error: {e}"


def check_docker_image(image_name: str) -> tuple[bool, str]:
    """Check whether a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name], capture_output=True, text=True
        )
        if result.returncode == 0:
            return True, "Found"
        return False, "Not Found"
    except FileNotFoundError:
        return False, "Docker command not found"


def check_nvidia_docker() -> tuple[bool, str]:
    """Check whether the NVIDIA Docker runtime is available."""
    try:
        # Check Docker without running a full container
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if "nvidia" in result.stdout:
                return True, "NVIDIA Docker Runtime OK"
            return False, "NVIDIA runtime not active (warning)"
        return False, "Could not query docker info"
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
        return False, f"Docker check failed: {e}"


def main() -> None:
    """Run all system health checks and log results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("System Health Check - %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # 1. Local Python Environment
    logger.info("\n--- Local Python Environment ---")
    modules = ["mujoco", "PyQt6", "numpy", "matplotlib"]

    for mod in modules:
        ok, msg = check_python_module(mod)
        log_result(f"Module: {mod}", "OK" if ok else "FAIL", msg)

    # 2. Docker Health
    logger.info("\n--- Docker Environment ---")
    docker_ok, docker_msg = check_docker_image("robotics_env")
    log_result("Image: robotics_env", "OK" if docker_ok else "FAIL", docker_msg)

    # 3. Pinocchio / Docker Libraries Check
    # We verify if the critical libraries we added (libEGL) are present in the image
    if docker_ok:
        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                "robotics_env",
                "sh",
                "-c",
                "dpkg -l libegl1 libxkbcommon-x11-0 libxcb-cursor0",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                log_result("Docker Libs (libEGL/XCB)", "OK", "Installed")
            else:
                log_result(
                    "Docker Libs (libEGL/XCB)", "FAIL", "Missing dependencies in image"
                )
                logger.warning(res.stderr)
        except (subprocess.SubprocessError, OSError) as e:
            log_result("Docker Libs (libEGL/XCB)", "FAIL", str(e))

    logger.info("\n--- File Integrity ---")
    # Check for critical launcher files
    files = [
        "launchers/golf_launcher.py",
        "engines/physics_engines/mujoco/python/humanoid_launcher.py",
        "engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui.py",
    ]

    root = Path(__file__).parent.resolve()
    for f in files:
        path = root / f
        exists = path.exists()
        log_result(
            f"File: {f}", "OK" if exists else "FAIL", "Found" if exists else "Missing"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Check complete.")


if __name__ == "__main__":
    main()
