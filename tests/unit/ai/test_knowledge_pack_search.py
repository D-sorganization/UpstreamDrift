from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.ai.knowledge.manifest import PackManifest, SourceSpec
from src.shared.python.ai.knowledge.pack import KnowledgePack, build_pack

pytestmark = pytest.mark.unit


def _multi_repo_pack(tmp_path: Path) -> KnowledgePack:
    roots: dict[str, Path] = {}
    for repo in ("UpstreamDrift", "Tools"):
        root = tmp_path / repo.lower()
        docs = root / "docs"
        docs.mkdir(parents=True)
        (docs / "topic.md").write_text(
            f"# {repo} Topic\n\nShared quasar diagnostic wording for {repo}.\n",
            encoding="utf-8",
        )
        roots[repo] = root
    manifest = PackManifest(
        id="multi_repo_pack",
        title="Multi Repo Pack",
        sources=(
            SourceSpec("UpstreamDrift", "product", ("docs/**/*.md",)),
            SourceSpec("Tools", "reference", ("docs/**/*.md",)),
        ),
    )
    pack_path = tmp_path / "multi.sqlite"
    build_pack(manifest, roots, pack_path)
    return KnowledgePack.open(pack_path)


def _pack(tmp_path: Path) -> KnowledgePack:
    root = tmp_path / "source"
    docs = root / "docs"
    docs.mkdir(parents=True)
    contents = {
        "title-match.md": "# Quasar Diagnostics\n\nA focused procedure describes telemetry and calibration.\n",
        "body-match.md": "# General Reference\n\nThe quasar diagnostic process explains telemetry and calibration.\n",
        "old.md": "---\nstatus: superseded\n---\n# Retired Quasar\n\nOld quasar procedures are obsolete.\n",
    }
    for name, text in contents.items():
        (docs / name).write_text(text, encoding="utf-8")
    manifest = PackManifest(
        id="test_pack",
        title="Test Pack",
        sources=(SourceSpec("UpstreamDrift", "product", ("docs/**/*.md",)),),
        status_overrides={"docs/old.md": "superseded"},
    )
    pack_path = tmp_path / "pack.sqlite"
    build_pack(manifest, {"UpstreamDrift": root}, pack_path)
    return KnowledgePack.open(pack_path)


def test_search_preserves_title_weighted_bm25_order(tmp_path: Path) -> None:
    results = _pack(tmp_path).search("quasar diagnostic telemetry", k=2)
    assert [passage.title for passage in results] == [
        "Quasar Diagnostics",
        "General Reference",
    ]


def test_search_returns_empty_for_blank_query(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    assert pack.search("") == []
    assert pack.search("   \t") == []


def test_search_preserves_repository_scope(tmp_path: Path) -> None:
    results = _multi_repo_pack(tmp_path).search("quasar diagnostic", k=4)
    assert {passage.repo for passage in results} == {"UpstreamDrift", "Tools"}
    assert {passage.title for passage in results} == {
        "UpstreamDrift Topic",
        "Tools Topic",
    }


def test_search_hides_superseded_and_sanitizes_fts(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    assert all(passage.status != "superseded" for passage in pack.search("quasar"))
    assert any(
        passage.status == "superseded"
        for passage in pack.search("quasar", include_superseded=True)
    )
    assert pack.search('quasar OR "diagnostic"*', k=3)
    assert pack.search('"*() :', k=3) == []


@pytest.mark.parametrize(
    ("query", "k", "error"),
    [(None, 1, TypeError), ("quasar", 0, ValueError), ("quasar", True, ValueError)],
)
def test_search_rejects_invalid_inputs(
    tmp_path: Path, query: object, k: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _pack(tmp_path).search(query, k=k)  # type: ignore[arg-type]
