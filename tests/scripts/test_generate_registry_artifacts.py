"""The registry artifacts are generated, committed, and drift-checked (#9412)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.registry import generate_registry_artifacts as gen

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


def test_committed_artifacts_are_up_to_date() -> None:
    stale = gen.check(REPO_ROOT)
    assert not stale, (
        f"stale registry artifacts: {[str(p.relative_to(REPO_ROOT)) for p in stale]}; "
        "run: python -m scripts.registry.generate_registry_artifacts"
    )


def test_check_cli_exit_codes() -> None:
    assert gen.main(["--check", "--repo-root", str(REPO_ROOT)]) == 0


def test_generation_is_deterministic() -> None:
    first = gen.project_artifacts(REPO_ROOT)
    second = gen.project_artifacts(REPO_ROOT)
    assert {p: r for p, (_, r) in first.items()} == {
        p: r for p, (_, r) in second.items()
    }


def _scratch_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "config").mkdir(parents=True)
    for rel in (
        "src/config/models.yaml",
        "src/config/launcher_manifest.json",
        "src/config/feature_parity.json",
        "README.md",
    ):
        shutil.copy(REPO_ROOT / rel, root / rel)
    # The help-coverage gate (#9413) reads the pages the registry points at,
    # so the scratch repo needs the real docs/help tree.
    shutil.copytree(REPO_ROOT / "docs" / "help", root / "docs" / "help")
    return root


def test_help_gate_rejects_a_declared_page_that_does_not_exist(
    tmp_path: Path,
) -> None:
    """A tile may not point at a help page that is not in the tree (#9413)."""
    root = _scratch_repo(tmp_path)
    assert gen.help_violations(root) == []
    (root / "docs" / "help" / "engine_selection.md").unlink()
    problems = gen.help_violations(root)
    assert problems
    assert any("engine_selection.md" in problem for problem in problems)
    assert gen.main(["--check", "--repo-root", str(root)]) == 1


def test_check_detects_drift_and_generate_repairs_it(tmp_path: Path) -> None:
    root = _scratch_repo(tmp_path)
    manifest = root / "src/config/launcher_manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert gen.main(["--check", "--repo-root", str(root)]) == 1
    assert gen.main(["--repo-root", str(root)]) == 0
    assert gen.main(["--check", "--repo-root", str(root)]) == 0


def test_manifest_projection_has_every_registry_tile() -> None:
    from src.config.tile_registry import load_tile_registry

    registry = load_tile_registry()
    manifest = registry.web_manifest_dict()
    assert [t["id"] for t in manifest["tiles"]] == [
        t.id for t in sorted(registry.tiles, key=lambda t: (t.order, t.id))
    ]
    assert manifest["tiles"][0]["id"] == "model_explorer"
    for tile in manifest["tiles"]:
        assert set(tile) >= {"maturity", "surfaces", "help", "feature_id", "web"}


def test_unknown_feature_binding_is_refused(tmp_path: Path) -> None:
    root = _scratch_repo(tmp_path)
    models = root / "src/config/models.yaml"
    text = models.read_text(encoding="utf-8")
    assert 'feature_id: "engines.load_and_simulate"' in text
    models.write_text(
        text.replace(
            'feature_id: "engines.load_and_simulate"', 'feature_id: "no.such"', 1
        ),
        encoding="utf-8",
    )
    assert gen.main(["--check", "--repo-root", str(root)]) == 2
