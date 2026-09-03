"""Tests for the documentation-map generator (issues #8839, #8850)."""

from __future__ import annotations

from importlib import util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "generate_docs_map.py"

INDEX_TEMPLATE = """# Docs

## Documentation Map

<!-- BEGIN GENERATED: docs-map (scripts/generate_docs_map.py) -->

placeholder

<!-- END GENERATED: docs-map -->

## Directory Catalog

| Directory | Owner    | Stability | Description                        |
| --------- | -------- | --------- | ---------------------------------- |
| `alpha/`  | @team-a  | stable    | Alpha guidance for the alpha area. |
| `beta/`   | @team-b  | archived  | Beta records preserved for later.  |
"""

README_TEMPLATE = """# Hub

## Documentation Structure

<!-- BEGIN GENERATED: docs-structure (scripts/generate_docs_map.py) -->

placeholder

<!-- END GENERATED: docs-structure -->
"""


def _load_module():
    spec = util.spec_from_file_location("generate_docs_map_under_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generate_docs_map script")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return the generator module pointed at a small synthetic docs tree."""
    mod = _load_module()
    docs = tmp_path / "docs"
    (docs / "alpha").mkdir(parents=True)
    (docs / "beta").mkdir()
    (docs / "alpha" / "README.md").write_text("# alpha", encoding="utf-8")
    (docs / "beta" / "one.md").write_text("# one", encoding="utf-8")
    (docs / "beta" / "two.md").write_text("# two", encoding="utf-8")
    index = docs / "index.md"
    readme = docs / "README.md"
    index.write_text(INDEX_TEMPLATE, encoding="utf-8")
    readme.write_text(README_TEMPLATE, encoding="utf-8")

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "DOCS_DIR", docs)
    monkeypatch.setattr(mod, "INDEX_PATH", index)
    monkeypatch.setattr(mod, "README_PATH", readme)
    return mod, index, readme


def test_stability_is_read_from_the_catalog(sandbox) -> None:
    mod, _, _ = sandbox
    assert mod.read_stability() == {"alpha": "stable", "beta": "archived"}


def test_entry_target_prefers_a_landing_page(sandbox) -> None:
    mod, _, _ = sandbox
    assert mod.entry_target(mod.DOCS_DIR / "alpha") == "alpha/README.md"


def test_entry_target_falls_back_to_the_directory(sandbox) -> None:
    """Two pages and no landing page means the directory is the entry point."""
    mod, _, _ = sandbox
    assert mod.entry_target(mod.DOCS_DIR / "beta") == "beta/"


def test_generation_groups_by_stability(sandbox) -> None:
    mod, index, _ = sandbox
    assert mod.apply(check_only=False) == 0
    text = index.read_text(encoding="utf-8")
    assert "### Stable" in text
    assert "### Archived" in text
    assert "[`alpha/`](alpha/README.md)" in text
    assert text.index("### Stable") < text.index("### Archived")


def test_readme_structure_counts_real_pages(sandbox) -> None:
    mod, _, readme = sandbox
    assert mod.apply(check_only=False) == 0
    text = readme.read_text(encoding="utf-8")
    assert "|-- beta/" in text
    assert "# 2 pages" in text
    assert "# 1 page" in text


def test_check_mode_detects_drift(sandbox) -> None:
    mod, _, _ = sandbox
    assert mod.apply(check_only=True) == 1
    assert mod.apply(check_only=False) == 0
    assert mod.apply(check_only=True) == 0


def test_new_directory_makes_the_block_stale(sandbox) -> None:
    """A directory added after generation must be reported by --check."""
    mod, _, _ = sandbox
    assert mod.apply(check_only=False) == 0
    (mod.DOCS_DIR / "gamma").mkdir()
    assert mod.apply(check_only=True) == 1


def test_missing_markers_are_a_hard_error(sandbox) -> None:
    mod, index, _ = sandbox
    index.write_text("# Docs\n\nno markers here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        mod.apply(check_only=False)


def test_uncatalogued_directory_is_still_listed(sandbox) -> None:
    mod, index, _ = sandbox
    (mod.DOCS_DIR / "delta").mkdir()
    assert mod.apply(check_only=False) == 0
    text = index.read_text(encoding="utf-8")
    assert "### Not yet catalogued" in text
    assert "[`delta/`](delta/)" in text
