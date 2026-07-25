"""Tests for provider-aware launcher model source helpers."""

from pathlib import Path

import pytest
from src.launchers.launcher_model_sources import (
    get_model_python_paths,
    get_model_source_root,
    get_model_working_directory,
    resolve_model_artifact_path,
)
from src.shared.python.config.model_registry import ModelRegistry
from src.shared.python.config.model_source_providers import resolve_model_source


class ProviderBackedModel:
    path = "models/humanoid.urdf"
    source_root = "../Drake_Models"
    working_dir = "python"
    python_paths = ["src", "bindings", "src"]


def test_get_model_source_root_uses_override() -> None:
    root = get_model_source_root(ProviderBackedModel(), Path("/repos/UpstreamDrift"))
    assert root == (Path("/repos/UpstreamDrift") / "../Drake_Models").resolve()


def test_resolve_model_artifact_path_uses_source_root() -> None:
    artifact = resolve_model_artifact_path(
        ProviderBackedModel(), Path("/repos/UpstreamDrift")
    )
    assert (
        artifact
        == (
            Path("/repos/UpstreamDrift") / "../Drake_Models/models/humanoid.urdf"
        ).resolve()
    )


def test_get_model_working_directory_uses_override() -> None:
    working_dir = get_model_working_directory(
        ProviderBackedModel(), Path("/repos/UpstreamDrift")
    )
    assert (
        working_dir
        == (Path("/repos/UpstreamDrift") / "../Drake_Models/python").resolve()
    )


def test_get_model_python_paths_deduplicates_entries() -> None:
    paths = get_model_python_paths(ProviderBackedModel(), Path("/repos/UpstreamDrift"))
    assert paths == (
        (Path("/repos/UpstreamDrift") / "../Drake_Models/src").resolve(),
        (Path("/repos/UpstreamDrift") / "../Drake_Models/bindings").resolve(),
    )


def test_resolve_model_artifact_path_requires_model_path() -> None:
    with pytest.raises(ValueError, match="path must be a non-empty string"):
        resolve_model_artifact_path(object(), Path("/repos/UpstreamDrift"))


def test_resolve_model_source_preserves_local_provider_parity(tmp_path: Path) -> None:
    class LocalModel:
        path = "src/engines/physics_engines/mujoco/python/main.py"

    resolved = resolve_model_source(LocalModel(), tmp_path)

    assert resolved.provider_id == "local-repo"
    assert resolved.source_root == tmp_path.resolve()
    assert (
        resolved.artifact_path
        == (tmp_path / "src/engines/physics_engines/mujoco/python/main.py").resolve()
    )
    assert resolved.working_directory == tmp_path.resolve()


def test_resolve_model_source_rejects_paths_outside_approved_roots() -> None:
    class InvalidModel:
        path = "../../outside.py"

    with pytest.raises(ValueError, match="path must resolve within approved roots"):
        resolve_model_source(InvalidModel(), Path("/repos/UpstreamDrift"))


def test_movement_optimizer_targets_sibling_public_entry_point() -> None:
    """The ready tile must resolve to Movement-Optimizer's public CLI module."""
    repo_root = Path(__file__).resolve().parents[2]
    model = ModelRegistry().get_model("movement_optimizer")

    assert model is not None
    assert (
        resolve_model_artifact_path(model, repo_root)
        == (
            repo_root.parent
            / "Movement_Optimizer"
            / "src/movement_optimizer/__main__.py"
        ).resolve()
    )
    assert (
        get_model_working_directory(model, repo_root)
        == (repo_root.parent / "Movement_Optimizer").resolve()
    )
    assert get_model_python_paths(model, repo_root) == (
        (repo_root.parent / "Movement_Optimizer" / "src").resolve(),
    )


def test_visible_registry_tiles_have_unique_display_name_and_artifact() -> None:
    """Visible launcher cards must not send users to the same entry point.

    Hidden aliases are intentionally excluded: they preserve saved layouts
    without presenting duplicate cards in the PyQt6 launcher.
    """
    repo_root = Path(__file__).resolve().parents[2]
    registry = ModelRegistry()
    visible_targets: dict[tuple[str, Path], list[str]] = {}

    for model in registry.get_all_models():
        if model.hidden:
            continue
        key = (model.name.casefold(), resolve_model_artifact_path(model, repo_root))
        visible_targets.setdefault(key, []).append(model.id)

    duplicates = {
        target: model_ids
        for target, model_ids in visible_targets.items()
        if len(model_ids) > 1
    }
    assert not duplicates, f"Duplicate visible launcher targets: {duplicates}"


def test_putting_green_hidden_alias_is_documented() -> None:
    """Saved layouts retain the alias without creating a second visible card."""
    registry = ModelRegistry()
    canonical = registry.get_model("putting_green")
    alias = registry.get_model("putting_green_gui")

    assert canonical is not None
    assert canonical.hidden is False
    assert alias is not None
    assert alias.hidden is True
    assert alias.hidden_reason
    assert alias.hidden_owner
    assert alias.name == canonical.name
