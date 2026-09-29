"""Smoke tests for the API Docker image artifact."""

from __future__ import annotations

import os
import subprocess

import pytest

IMAGE_ENV = "UPSTREAM_DRIFT_API_IMAGE"
pytestmark = pytest.mark.smoke


def test_image_artifact_declares_non_root_user() -> None:
    """The release Docker image must run with a non-root default user."""
    image_name = os.environ.get(IMAGE_ENV)
    if not image_name:
        raise AssertionError(f"Set {IMAGE_ENV} to the built API image tag")

    result = subprocess.run(
        ["docker", "image", "inspect", image_name, "--format", "{{.Config.User}}"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() not in {"", "0", "root"}
