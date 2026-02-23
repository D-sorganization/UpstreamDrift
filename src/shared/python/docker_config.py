"""Centralized Docker configuration for UpstreamDrift."""

from __future__ import annotations

import os

# Base Image Family
DOCKER_IMAGE_FAMILY = os.environ.get("UPSTREAM_DRIFT_IMAGE_FAMILY", "upstream-drift")

# Primary Images
DOCKER_IMAGE_ENGINE = f"{DOCKER_IMAGE_FAMILY}:engine"
DOCKER_IMAGE_RUNTIME = f"{DOCKER_IMAGE_FAMILY}:runtime"
DOCKER_IMAGE_DEV = f"{DOCKER_IMAGE_FAMILY}:dev"

# Legacy images for fallback/cleanup validation
LEGACY_DOCKER_ALIASES = (
    "robotics_env:latest",
    "golf-suite:latest",
    "golf-suite-dev:latest",
)
