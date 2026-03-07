"""Centralized Docker configuration for UpstreamDrift."""

from __future__ import annotations

import os
import shutil
import subprocess

# Base Image Family
DOCKER_IMAGE_FAMILY = os.environ.get("UPSTREAM_DRIFT_IMAGE_FAMILY", "upstream-drift")

# Primary Images
DOCKER_IMAGE_ENGINE = f"{DOCKER_IMAGE_FAMILY}:engine"
DOCKER_IMAGE_RUNTIME = f"{DOCKER_IMAGE_FAMILY}:runtime"
DOCKER_IMAGE_DEV = f"{DOCKER_IMAGE_FAMILY}:dev"
DOCKER_IMAGE_TRAINING = f"{DOCKER_IMAGE_FAMILY}:training"

# Legacy images for fallback/cleanup validation
LEGACY_DOCKER_ALIASES = (
    "robotics_env:latest",
    "golf-suite:latest",
    "golf-suite-dev:latest",
)


def detect_gpu_support() -> dict:
    """Detect GPU availability on the host system.

    Returns a dict with:
        available (bool): Whether an NVIDIA GPU is accessible.
        device_name (str): GPU name, or empty string.
        driver_version (str): NVIDIA driver version, or empty string.
        cuda_version (str): CUDA version reported by nvidia-smi, or empty string.
        container_toolkit (bool): Whether nvidia-container-toolkit appears installed.
    """
    result = {
        "available": False,
        "device_name": "",
        "driver_version": "",
        "cuda_version": "",
        "container_toolkit": False,
    }

    # Check for nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return result

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split(", ")
            result["available"] = True
            result["device_name"] = parts[0] if len(parts) > 0 else ""
            result["driver_version"] = parts[1] if len(parts) > 1 else ""

        # Get CUDA version
        proc2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc2.returncode == 0:
            result["cuda_version"] = proc2.stdout.strip()

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Check for NVIDIA Container Toolkit
    result["container_toolkit"] = shutil.which("nvidia-container-cli") is not None

    return result
