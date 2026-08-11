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


TOOLS_MODEL_IDS = (
    "video_analyzer",
    "video_processor",
    "data_explorer",
    "data_processor",
    "rate_of_closure",
)


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


def test_tools_provider_prefers_pinned_vendor_over_mutable_sibling(
    tmp_path: Path,
) -> None:
    class ToolsModel:
        provider = "tools"
        source_root = "../Tools"
        path = "src/tool.py"

    vendor_root = tmp_path / "vendor" / "ud-tools"
    sibling_root = tmp_path.parent / "Tools"
    (vendor_root / "src").mkdir(parents=True)
    (sibling_root / "src").mkdir(parents=True, exist_ok=True)

    resolved = resolve_model_source(ToolsModel(), tmp_path)

    assert resolved.provider_id == "tools-vendor"
    assert resolved.source_root == vendor_root.resolve()
    assert resolved.artifact_path == (vendor_root / "src" / "tool.py").resolve()


def test_tools_provider_missing_vendor_does_not_fall_back_to_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ToolsModel:
        provider = "tools"
        path = "src/tool.py"

    sibling_root = tmp_path.parent / "Tools"
    (sibling_root / "src").mkdir(parents=True, exist_ok=True)
    override_root = tmp_path.parent / "ExplicitTools"
    (override_root / "src").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(override_root))

    resolved = resolve_model_source(ToolsModel(), tmp_path)

    assert resolved.provider_id == "tools-vendor"
    assert resolved.source_root == (tmp_path / "vendor" / "ud-tools").resolve()
    assert not resolved.source_root.exists()


def test_all_tools_launchers_resolve_from_pinned_vendor(tmp_path: Path) -> None:
    vendor_root = tmp_path / "vendor" / "ud-tools"
    sibling_root = tmp_path.parent / "Tools"
    (vendor_root / "src").mkdir(parents=True)
    (sibling_root / "src").mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()

    for model_id in TOOLS_MODEL_IDS:
        model = registry.get_model(model_id)
        assert model is not None
        resolved = resolve_model_source(model, tmp_path)
        assert resolved.provider_id == "tools-vendor"
        assert resolved.source_root == vendor_root.resolve()


def test_generic_sibling_provider_resolution_is_unchanged(tmp_path: Path) -> None:
    class GenericSiblingModel:
        provider = "custom"
        source_root = "SiblingModels"
        path = "models/model.urdf"

    sibling_root = tmp_path.parent / "SiblingModels"
    sibling_root.mkdir(parents=True, exist_ok=True)

    resolved = resolve_model_source(GenericSiblingModel(), tmp_path)

    assert resolved.provider_id == "sibling-repo"
    assert resolved.source_root == sibling_root.resolve()


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
