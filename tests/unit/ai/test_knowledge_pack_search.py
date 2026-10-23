from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.ai.knowledge.manifest import PackManifest, SourceSpec
from src.shared.python.ai.knowledge.pack import KnowledgePack, build_pack


def _pack(tmp_path: Path) -> KnowledgePack:
    root = tmp_path / "source"
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "title-match.md").write_text(
        "# Quasar Diagnostics\n\nA focused procedure describes telemetry and calibration.\n",
        encoding="utf-8",
    )
    (docs / "body-match.md").write_text(
        "# General Reference\n\nThe quasar diagnostic process explains telemetry and calibration.\n",
        encoding="utf-8",
    )
    (docs / "old.md").write_text(
        "---\nstatus: superseded\n---\n# Retired Quasar\n\nOld quasar procedures are obsolete.\n",
        encoding="utf-8",
    )
    manifest = PackManifest(
        id="test_pack",
        title="Test Pack",
        sources=(
            SourceSpec(
                repo="UpstreamDrift",
                authority="product",
                include=("docs/**/*.md",),
            ),
        ),
        status_overrides={"docs/old.md": "superseded"},
    )
    pack_path = tmp_path / "pack.sqlite"
    build_pack(manifest, {"UpstreamDrift": root}, pack_path)
    return KnowledgePack.open(pack_path)


@pytest.mark.unit
def test_search_preserves_title_weighted_bm25_order(tmp_path: Path) -> None:
    results = _pack(tmp_path).search("quasar diagnostic telemetry", k=2)

    assert [passage.title for passage in results] == [
        "Quasar Diagnostics",
        "General Reference",
    ]


@pytest.mark.unit
def test_search_hides_superseded_by_default_and_can_include_it(
    tmp_path: Path,
) -> None:
    pack = _pack(tmp_path)

    assert all(passage.status != "superseded" for passage in pack.search("quasar"))
    assert any(
        passage.status == "superseded"
        for passage in pack.search("quasar", include_superseded=True)
    )


@pytest.mark.unit
def test_search_sanitizes_fts_operators_and_returns_empty_for_punctuation(
    tmp_path: Path,
) -> None:
    pack = _pack(tmp_path)

    assert pack.search('quasar OR "diagnostic"*', k=3)
    assert pack.search('"*() :', k=3) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "k", "error"),
    [(None, 1, TypeError), ("quasar", 0, ValueError), ("quasar", True, ValueError)],
)
def test_search_rejects_invalid_inputs(
    tmp_path: Path, query: object, k: object, error: type[Exception]
) -> None:
    pack = _pack(tmp_path)

    with pytest.raises(error):
        pack.search(query, k=k)  # type: ignore[arg-type]
