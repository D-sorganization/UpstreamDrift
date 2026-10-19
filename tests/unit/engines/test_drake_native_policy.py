"""Native Drake version policy for torque replay (#13019)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from src.engines.physics_engines.drake.python.drake_native_policy import (
    DRAKE_OPTIONAL_DEPENDENCY_SPEC,
    DRAKE_REVIEWED_VERSIONS,
    is_reviewed_native_drake_version,
    require_reviewed_native_drake_version,
    skip_unless_reviewed_native_drake,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pyproject_drake_extra_matches_policy_constant() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    drake_entries = extras["drake"]
    assert len(drake_entries) == 1
    assert drake_entries[0] == f"drake{DRAKE_OPTIONAL_DEPENDENCY_SPEC}"


def test_each_reviewed_version_satisfies_optional_dependency_spec() -> None:
    spec = SpecifierSet(DRAKE_OPTIONAL_DEPENDENCY_SPEC)
    for version in DRAKE_REVIEWED_VERSIONS:
        assert Version(version) in spec


def test_require_reviewed_native_drake_version_accepts_reviewed_builds() -> None:
    for version in DRAKE_REVIEWED_VERSIONS:
        assert require_reviewed_native_drake_version(version) == version


def test_require_reviewed_native_drake_version_rejects_unreviewed() -> None:
    with pytest.raises(ValueError, match="reviewed Drake"):
        require_reviewed_native_drake_version("1.59.0")


def test_skip_unless_reviewed_native_drake_skips_unreviewed_distribution() -> None:
    with (
        patch(
            "src.engines.physics_engines.drake.python.drake_native_policy.installed_drake_distribution_version",
            return_value="1.59.0",
        ),
        patch(
            "pytest.importorskip",
            return_value=None,
        ),
    ):
        with pytest.raises(pytest.skip.Exception) as exc:
            skip_unless_reviewed_native_drake()
    assert "1.59.0" in str(exc.value)
    assert "1.57.0" in str(exc.value)


def test_is_reviewed_native_drake_version_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        is_reviewed_native_drake_version("")
