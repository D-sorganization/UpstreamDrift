"""Smoke tests for the Tauri desktop bundle artifact."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

BUNDLE_ENV = "UPSTREAM_DRIFT_TAURI_BUNDLE"
pytestmark = pytest.mark.smoke


def test_bundle_artifact_exists() -> None:
    """Release smoke tests must point at a real Tauri bundle."""
    bundle_value = os.environ.get(BUNDLE_ENV)
    if not bundle_value:
        raise AssertionError(f"Set {BUNDLE_ENV} to the built Tauri bundle path")

    assert Path(bundle_value).is_file()
