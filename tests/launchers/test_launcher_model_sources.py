"""Tests for provider-aware launcher model source helpers."""

from pathlib import Path

import pytest

from src.launchers.launcher_model_sources import (
    get_model_python_paths,
    get_model_source_root,
    get_model_working_directory,
    resolve_model_artifact_path,
)


class ProviderBackedModel:
    path = "models/humanoid.urdf"
    source_root = "../Drake_Models"
    working_dir = "python"
    python_paths = ["src", "bindings", "src"]


def test_get_model_source_root_uses_override() -> None:
    root = get_model_source_root(ProviderBackedModel(), Path("/repos/UpstreamDrift"))
    assert root == Path("/repos/UpstreamDrift/../Drake_Models")


def test_resolve_model_artifact_path_uses_source_root() -> None:
    artifact = resolve_model_artifact_path(
        ProviderBackedModel(), Path("/repos/UpstreamDrift")
    )
    assert artifact == Path("/repos/UpstreamDrift/../Drake_Models/models/humanoid.urdf")


def test_get_model_working_directory_uses_override() -> None:
    working_dir = get_model_working_directory(
        ProviderBackedModel(), Path("/repos/UpstreamDrift")
    )
    assert working_dir == Path("/repos/UpstreamDrift/../Drake_Models/python")


def test_get_model_python_paths_deduplicates_entries() -> None:
    paths = get_model_python_paths(ProviderBackedModel(), Path("/repos/UpstreamDrift"))
    assert paths == (
        Path("/repos/UpstreamDrift/../Drake_Models/src"),
        Path("/repos/UpstreamDrift/../Drake_Models/bindings"),
    )


def test_resolve_model_artifact_path_requires_model_path() -> None:
    with pytest.raises(ValueError, match="model.path must be a non-empty string"):
        resolve_model_artifact_path(object(), Path("/repos/UpstreamDrift"))
