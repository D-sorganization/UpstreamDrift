from __future__ import annotations

import os
from pathlib import Path

import pytest

from installer.windows.packaging_profiles import (
    build_profile_environment,
    get_packaging_profile,
    iter_packaging_profile_ids,
)


def test_get_packaging_profile_defaults_to_hybrid() -> None:
    profile = get_packaging_profile(None)

    assert profile.profile_id == "hybrid"
    assert profile.discovery_mode == "hybrid"
    assert profile.include_api_executable is True


def test_get_packaging_profile_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unknown packaging profile"):
        get_packaging_profile("invalid")


def test_iter_packaging_profile_ids_is_stable() -> None:
    assert iter_packaging_profile_ids() == ("core", "hybrid", "full")


def test_build_profile_environment_sets_profile_and_provider_roots() -> None:
    profile = get_packaging_profile("full")
    env = build_profile_environment(
        profile,
        provider_roots=("C:/repos/MuJoCo_Models", "C:/repos/Drake_Models"),
        base_env={"PATH": "x", "UPSTREAM_DRIFT_PROVIDER_ROOTS": "stale"},
    )

    assert env["UPSTREAM_DRIFT_INSTALL_PROFILE"] == "full"
    assert env["UPSTREAM_DRIFT_DISCOVERY_MODE"] == "provider-first"
    assert env["UPSTREAM_DRIFT_PROVIDER_ROOTS"] == os.pathsep.join(
        (str(Path("C:/repos/MuJoCo_Models")), str(Path("C:/repos/Drake_Models")))
    )


def test_build_profile_environment_clears_provider_roots_for_core() -> None:
    profile = get_packaging_profile("core")
    env = build_profile_environment(
        profile,
        base_env={"UPSTREAM_DRIFT_PROVIDER_ROOTS": "stale", "PATH": "x"},
    )

    assert env["UPSTREAM_DRIFT_INSTALL_PROFILE"] == "core"
    assert env["UPSTREAM_DRIFT_DISCOVERY_MODE"] == "local-only"
    assert "UPSTREAM_DRIFT_PROVIDER_ROOTS" not in env
