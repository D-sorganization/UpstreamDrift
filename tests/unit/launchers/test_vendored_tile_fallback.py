"""Tiles resolve from vendor/ud-tools when no sibling checkout exists (#12993)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.shared.python.config import model_source_providers as msp
from src.shared.python.config.model_registry import ModelRegistry
from src.shared.python.config.model_source_providers import (
    ModelSourcePathPolicy,
    SourceRootNotFoundError,
    normalize_dir_name,
    resolve_model_artifact,
    tools_subproject_from_source_root,
)
from src.shared.python.config.tile_target_resolution import resolve_tile_target
from src.shared.python.config.tools_vendor_authority import (
    ToolsVendorAuthority,
    inspect_tools_vendor_authority,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TILES = ("pendulum_simulator", "swing_objective_lab", "movement_optimizer")


def _vendor_only_layout(tmp_path: Path) -> Path:
    """tmp workspace: UpstreamDrift + vendor/ud-tools only, no siblings."""
    repo = tmp_path / "work" / "UpstreamDrift"
    vendor = repo / "vendor" / "ud-tools" / "src"
    (vendor / "movement_optimizer").mkdir(parents=True)
    (vendor / "movement_optimizer" / "__main__.py").write_text("")
    pend = vendor / "pendulum_simulator" / "src" / "double_pendulum_golf"
    (pend / "swing_objectives").mkdir(parents=True)
    (pend / "__main__.py").write_text("")
    (pend / "swing_objectives" / "__main__.py").write_text("")
    return repo


@pytest.fixture
def vendor_authority(monkeypatch):
    """Treat the git-less tmp vendor tree as a valid pinned gitlink."""

    def fake(repo_root: Path) -> ToolsVendorAuthority:
        root = repo_root / "vendor" / "ud-tools"
        return ToolsVendorAuthority(root=root, expected_sha="x", available=True)

    monkeypatch.setattr(msp, "inspect_tools_vendor_authority", fake)


@pytest.mark.unit
@pytest.mark.parametrize("tile", TILES)
def test_tile_artifact_resolves_with_only_vendor(tile, tmp_path, vendor_authority):
    repo = _vendor_only_layout(tmp_path)
    model = ModelRegistry().get_model(tile)
    assert model is not None
    artifact = resolve_model_artifact(model, repo)
    assert artifact.is_file(), artifact
    assert "ud-tools" in artifact.parts


@pytest.mark.unit
@pytest.mark.parametrize("tile", TILES)
def test_real_checkout_tile_launch_target_exists(tile):
    if not inspect_tools_vendor_authority(REPO_ROOT).available:
        pytest.skip("vendor/ud-tools not materialised")
    model = ModelRegistry().get_model(tile)
    assert model is not None
    assert resolve_model_artifact(model, REPO_ROOT).is_file()
    assert resolve_tile_target(model, REPO_ROOT).resolvable


@pytest.mark.unit
def test_movement_optimizer_policy_falls_back_to_vendor(tmp_path):
    repo = _vendor_only_layout(tmp_path)
    root = ModelSourcePathPolicy(repo).resolve_source_root("Movement_Optimizer")
    assert root == (repo / "vendor" / "ud-tools").resolve()


@pytest.mark.unit
def test_hyphenated_sibling_matches_underscored_name(tmp_path):
    repo = tmp_path / "UpstreamDrift"
    repo.mkdir()
    sibling = tmp_path / "Movement-Optimizer"
    sibling.mkdir()
    root = ModelSourcePathPolicy(repo).resolve_source_root("Movement_Optimizer")
    assert root == sibling.resolve()


@pytest.mark.unit
def test_missing_source_root_lists_every_searched_path(tmp_path):
    repo = tmp_path / "UpstreamDrift"
    repo.mkdir()
    policy = ModelSourcePathPolicy(repo)
    with pytest.raises(SourceRootNotFoundError) as info:
        policy.require_source_root("Nowhere_Repo")
    message = str(info.value)
    for candidate in policy.source_root_candidates(Path("Nowhere_Repo")):
        assert str(candidate) in message
    assert "git submodule update --init vendor/ud-tools" in message


@pytest.mark.unit
def test_missing_sibling_tile_reports_unresolvable_with_paths(tmp_path):
    repo = tmp_path / "UpstreamDrift"
    repo.mkdir()
    tile = SimpleNamespace(
        id="x", path="src/x/__main__.py", type="special_app", source_root="Gone_Repo"
    )
    result = resolve_tile_target(tile, repo)
    assert not result.resolvable
    assert str(repo / "Gone_Repo") in result.reason


@pytest.mark.unit
@pytest.mark.parametrize(
    "a,b",
    [
        ("Movement_Optimizer", "Movement-Optimizer"),
        ("movement optimizer", "MovementOptimizer"),
    ],
)
def test_normalize_dir_name_is_style_insensitive(a, b):
    assert normalize_dir_name(a) == normalize_dir_name(b)


@pytest.mark.unit
def test_tools_subproject_extraction():
    assert (
        tools_subproject_from_source_root("../Tools/src/pendulum_simulator")
        == "src/pendulum_simulator"
    )
    assert tools_subproject_from_source_root(None) == ""
