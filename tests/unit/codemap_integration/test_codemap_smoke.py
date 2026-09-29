"""Smoke test: build a tiny code-map and round-trip a search.

This is the only test in the integration suite that actually exercises the
codemap *indexer*. It uses a 3-file fixture tree under ``tmp_path`` so it
never depends on the real repo's index. Skips cleanly when the codemap
extras (tree-sitter etc.) aren't installed.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

# Skip the whole module if the optional codemap extras are not present.
_REQUIRED = ("tree_sitter",)
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
if _MISSING:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"codemap extras not installed (missing: {_MISSING}); "
        "install with `pip install tree-sitter tree-sitter-python blake3`",
        allow_module_level=True,
    )


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """Create a 3-file fixture repo and chdir into it."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "alpha.py").write_text(
        textwrap.dedent('''
            """Alpha module."""

            def widget_factory(name: str) -> str:
                """Build a widget named ``name``."""
                return f"widget:{name}"


            class ChatDockWidget:
                """A canary class our chat test searches for."""

                def render(self) -> None: ...
            ''').strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "beta.py").write_text(
        textwrap.dedent('''
            """Beta module."""
            from .alpha import widget_factory


            def use_widget() -> str:
                return widget_factory("beta")
            ''').strip()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_rebuild_then_search_finds_canary_symbol(
    tiny_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold rebuild on a tiny fixture must surface ``ChatDockWidget``."""
    monkeypatch.chdir(tiny_repo)

    # Prefer the top-level ``codemap`` distribution (Tools installed as a
    # sibling) if importable; otherwise use the in-tree byte-identical copy.
    try:
        from codemap.indexer import rebuild as _rebuild  # type: ignore[import-not-found]
        from codemap.api import search_code as _search  # type: ignore[import-not-found]

        flavor = "tools"
    except ImportError:
        from src.shared.python.codemap.indexer import rebuild as _rebuild
        from src.shared.python.codemap.api import search_code as _search

        flavor = "local"

    _rebuild(tiny_repo)

    hits = _search("ChatDockWidget", k=10, repo_root=tiny_repo)
    assert hits, f"expected at least one hit for ChatDockWidget (flavor={flavor})"
    # Hit / SymbolRow both expose a path; assert it points back at our fixture.
    first = hits[0]
    sym = getattr(first, "symbol", first)
    path = getattr(sym, "path", None) or getattr(sym, "path", "")
    assert "alpha" in str(path).lower()


@pytest.mark.perf
def test_real_repo_index_within_budgets() -> None:
    """Perf budget: real-repo index must be reasonably populated yet compact.

    Skips when no `.codemap/index.db` exists — this guard runs in CI jobs
    that explicitly opt in to ``-m perf`` after rebuilding the index.
    """
    db = Path(".codemap/index.db").resolve()
    if not db.exists():
        pytest.skip("no .codemap/index.db; run `make codemap` first")

    try:
        from codemap.api import repo_summary  # type: ignore[import-not-found]
    except ImportError:
        from src.shared.python.codemap.api import repo_summary

    stats = repo_summary()
    symbols = stats.symbols
    size = stats.db_size_bytes

    assert symbols > 1000, f"only {symbols} symbols indexed; budget requires >1000"
    size_mb = size / (1024 * 1024)
    # UpstreamDrift indexes ~8k files / ~95k symbols → ~55 MB.
    # Budget includes headroom; bump only when the repo materially grows.
    assert size_mb < 100, f".codemap/index.db is {size_mb:.1f} MB; budget is <100 MB"
