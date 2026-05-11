"""Unit tests for the code-map indexer (shared.python.codemap).

Covers:
- DB schema creation (FTS5 + fallback)
- SymbolRow persistence round-trip
- Python parser: module, class, function, method, constant extraction
- Rust parser: function, struct, enum extraction (regex)
- TypeScript parser: function, class, arrow extraction (regex)
- Markdown parser: section headings
- CodeMapIndex cold rebuild on a tiny fixture tree
- Incremental update (add / delete)
- API functions: search, get_symbol, who_calls, imports_of
- CLI smoke test: rebuild + search
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.shared.python.codemap.api import (
    get_symbol,
    imports_of,
    search,
    who_calls,
)
from src.shared.python.codemap.cli import main as cli_main
from src.shared.python.codemap.db import (
    SymbolRow,
    get_manifest,
    open_db,
    search_fts,
    set_manifest,
    upsert_symbols,
    who_calls_db,
)
from src.shared.python.codemap.indexer import CodeMapIndex
from src.shared.python.codemap.parsers import (
    parse_markdown,
    parse_python,
    parse_rust,
    parse_typescript,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path: Path):
    """Open a temporary in-memory code-map database."""
    db_path = tmp_path / "test_index.db"
    conn = open_db(db_path)
    yield conn, db_path
    conn.close()


@pytest.fixture()
def simple_repo(tmp_path: Path) -> Path:
    """Create a minimal fake repo tree for integration tests."""
    # Python file
    (tmp_path / "mymod.py").write_text(
        textwrap.dedent('''\
            """Module docstring."""

            CONSTANT = 42

            class MyClass:
                """A class."""

                def method(self) -> None:
                    """A method."""
                    pass

            def top_func(x: int) -> int:
                """A top-level function."""
                return x + 1
            '''),
        encoding="utf-8",
    )

    # Rust file
    (tmp_path / "lib.rs").write_text(
        textwrap.dedent("""\
            pub struct Solver {}
            pub fn solve(x: f64) -> f64 { x }
            """),
        encoding="utf-8",
    )

    # Markdown file
    (tmp_path / "README.md").write_text(
        "# Title\n\n## Section One\n\n### Sub-section\n",
        encoding="utf-8",
    )

    return tmp_path


# ── DB layer ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_open_db_creates_schema(tmp_db) -> None:
    conn, _ = tmp_db
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','shadow')"
        ).fetchall()
    }
    assert "manifest" in tables


@pytest.mark.unit
def test_manifest_round_trip(tmp_db) -> None:
    conn, _ = tmp_db
    set_manifest(conn, "key1", {"a": 1})
    conn.commit()
    val = get_manifest(conn, "key1")
    assert val == {"a": 1}


@pytest.mark.unit
def test_manifest_missing_key_returns_none(tmp_db) -> None:
    conn, _ = tmp_db
    assert get_manifest(conn, "nonexistent") is None


@pytest.mark.unit
def test_upsert_and_search(tmp_db) -> None:
    conn, _ = tmp_db
    rows = [
        SymbolRow(
            kind="function",
            qualified_name="mymod.parse_spec",
            path="mymod.py",
            line_start=10,
            line_end=20,
            signature="def parse_spec(path: Path) -> dict",
        )
    ]
    upsert_symbols(conn, rows)
    conn.commit()
    results = search_fts(conn, "parse_spec")
    assert len(results) == 1
    assert results[0].qualified_name == "mymod.parse_spec"


@pytest.mark.unit
def test_upsert_replaces_existing(tmp_db) -> None:
    conn, _ = tmp_db
    row1 = SymbolRow("function", "mod.fn", "mod.py", 1, 5, "def fn(): pass")
    row2 = SymbolRow("function", "mod.fn", "mod.py", 3, 8, "def fn(x): pass")
    upsert_symbols(conn, [row1])
    conn.commit()
    upsert_symbols(conn, [row2])
    conn.commit()
    results = search_fts(conn, "fn")
    # Only one entry should exist (upsert deletes by path first)
    assert len(results) == 1
    assert results[0].line_start == 3


@pytest.mark.unit
def test_who_calls_db(tmp_db) -> None:
    conn, _ = tmp_db
    rows = [
        SymbolRow(
            kind="function",
            qualified_name="caller",
            path="a.py",
            line_start=1,
            line_end=5,
            calls_out="parse_spec validate_spec",
        )
    ]
    upsert_symbols(conn, rows)
    conn.commit()
    results = who_calls_db(conn, "parse_spec")
    assert any(r.qualified_name == "caller" for r in results)


# ── Python parser ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_python_module_symbol(simple_repo: Path) -> None:
    rows = parse_python(simple_repo / "mymod.py", simple_repo)
    kinds = {r.kind for r in rows}
    assert "module" in kinds


@pytest.mark.unit
def test_parse_python_class(simple_repo: Path) -> None:
    rows = parse_python(simple_repo / "mymod.py", simple_repo)
    classes = [r for r in rows if r.kind == "class"]
    assert any("MyClass" in r.qualified_name for r in classes)


@pytest.mark.unit
def test_parse_python_function(simple_repo: Path) -> None:
    rows = parse_python(simple_repo / "mymod.py", simple_repo)
    fns = [r for r in rows if r.kind == "function"]
    assert any("top_func" in r.qualified_name for r in fns)


@pytest.mark.unit
def test_parse_python_method(simple_repo: Path) -> None:
    rows = parse_python(simple_repo / "mymod.py", simple_repo)
    methods = [r for r in rows if r.kind == "method"]
    assert any("method" in r.qualified_name for r in methods)


@pytest.mark.unit
def test_parse_python_constant(simple_repo: Path) -> None:
    rows = parse_python(simple_repo / "mymod.py", simple_repo)
    consts = [r for r in rows if r.kind == "constant"]
    assert any("CONSTANT" in r.qualified_name for r in consts)


@pytest.mark.unit
def test_parse_python_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def (broken syntax:", encoding="utf-8")
    rows = parse_python(bad, tmp_path)
    # Should return empty list, not raise
    assert rows == []


# ── Rust parser ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_rust_function(simple_repo: Path) -> None:
    rows = parse_rust(simple_repo / "lib.rs", simple_repo)
    fns = [r for r in rows if r.kind == "function"]
    assert any("solve" in r.qualified_name for r in fns)


@pytest.mark.unit
def test_parse_rust_struct(simple_repo: Path) -> None:
    rows = parse_rust(simple_repo / "lib.rs", simple_repo)
    structs = [r for r in rows if r.kind == "class"]
    assert any("Solver" in r.qualified_name for r in structs)


# ── TypeScript parser ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_typescript(tmp_path: Path) -> None:
    ts = tmp_path / "app.ts"
    ts.write_text(
        "export function parseInput(x: string): number { return 0; }\n"
        "export class Parser {}\n",
        encoding="utf-8",
    )
    rows = parse_typescript(ts, tmp_path)
    kinds = {r.kind for r in rows}
    assert "function" in kinds
    assert "class" in kinds


# ── Markdown parser ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_markdown(simple_repo: Path) -> None:
    rows = parse_markdown(simple_repo / "README.md", simple_repo)
    assert len(rows) >= 3  # Title + Section One + Sub-section
    assert all(r.kind == "section" for r in rows)


# ── CodeMapIndex integration ───────────────────────────────────────────────────


@pytest.mark.unit
def test_rebuild_produces_symbols(simple_repo: Path) -> None:
    with CodeMapIndex(simple_repo) as idx:
        stats = idx.rebuild()
    assert stats.files_processed >= 3
    assert stats.symbols_indexed > 0
    assert stats.duration_s > 0
    assert stats.errors == 0


@pytest.mark.unit
def test_rebuild_manifest_written(simple_repo: Path) -> None:
    with CodeMapIndex(simple_repo) as idx:
        idx.rebuild()
        schema = idx.get_manifest_value("schema_version")
    assert schema == "1.0"


@pytest.mark.unit
def test_incremental_update_add_file(simple_repo: Path) -> None:
    with CodeMapIndex(simple_repo) as idx:
        idx.rebuild()
        new_file = simple_repo / "newmod.py"
        new_file.write_text("def new_function(): pass\n", encoding="utf-8")
        n = idx.update_file(new_file)
    assert n > 0


@pytest.mark.unit
def test_incremental_update_delete_file(simple_repo: Path) -> None:
    with CodeMapIndex(simple_repo) as idx:
        idx.rebuild()
        target = simple_repo / "mymod.py"
        target.unlink()
        n = idx.update_file(target)
    assert n == 0


# ── API layer ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_api_search(simple_repo: Path) -> None:
    db_path = simple_repo / ".codemap" / "index.db"
    with CodeMapIndex(simple_repo, db_path=db_path) as idx:
        idx.rebuild()
    results = search("top_func", db_path=db_path)
    assert any("top_func" in r.qualified_name for r in results)


@pytest.mark.unit
def test_api_get_symbol(simple_repo: Path) -> None:
    db_path = simple_repo / ".codemap" / "index.db"
    with CodeMapIndex(simple_repo, db_path=db_path) as idx:
        idx.rebuild()
        # Find what qualified name was used
        from src.shared.python.codemap.db import open_db, search_fts

        conn = open_db(db_path)
        hits = search_fts(conn, "top_func", limit=1)
        conn.close()

    if hits:
        qn = hits[0].qualified_name
        sym = get_symbol(qn, db_path=db_path)
        assert sym is not None
        assert sym.qualified_name == qn


@pytest.mark.unit
def test_api_search_missing_db(tmp_path: Path) -> None:
    """search() should return [] when index doesn't exist."""
    results = search("anything", db_path=tmp_path / "nonexistent.db")
    assert results == []


@pytest.mark.unit
def test_api_who_calls_missing_db(tmp_path: Path) -> None:
    results = who_calls("something", db_path=tmp_path / "nonexistent.db")
    assert results == []


@pytest.mark.unit
def test_api_imports_of_missing_db(tmp_path: Path) -> None:
    results = imports_of("something", db_path=tmp_path / "nonexistent.db")
    assert results == []


# ── CLI smoke tests ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_cli_rebuild(simple_repo: Path) -> None:
    exit_code = cli_main(
        [
            "--db",
            str(simple_repo / ".codemap" / "index.db"),
            "rebuild",
            "--repo",
            str(simple_repo),
        ]
    )
    assert exit_code == 0


@pytest.mark.unit
def test_cli_search_no_results(simple_repo: Path, capsys) -> None:
    db = simple_repo / ".codemap" / "index.db"
    cli_main(["--db", str(db), "rebuild", "--repo", str(simple_repo)])
    exit_code = cli_main(["--db", str(db), "search", "xyzzy_nonexistent_symbol_abc"])
    assert exit_code == 1  # No results → exit 1


# ── Performance and Size Budgets ───────────────────────────────────────────────


@pytest.mark.benchmark
def test_rebuild_perf(benchmark: pytest.FixtureRequest, tmp_path: Path) -> None:
    repo = tmp_path / "perf_repo"
    repo.mkdir()
    for i in range(10):
        (repo / f"mod_{i}.py").write_text(f"def func_{i}(): pass\n", encoding="utf-8")

    def do_rebuild() -> None:
        with CodeMapIndex(repo) as idx:
            idx.rebuild()

    benchmark(do_rebuild)


@pytest.mark.benchmark
def test_search_perf(benchmark: pytest.FixtureRequest, simple_repo: Path) -> None:
    db_path = simple_repo / ".codemap" / "index.db"
    with CodeMapIndex(simple_repo, db_path=db_path) as idx:
        idx.rebuild()

    def do_search() -> None:
        search("top_func", db_path=db_path)

    benchmark(do_search)


@pytest.mark.benchmark
def test_who_calls_perf(benchmark: pytest.FixtureRequest, simple_repo: Path) -> None:
    db_path = simple_repo / ".codemap" / "index.db"
    with CodeMapIndex(simple_repo, db_path=db_path) as idx:
        idx.rebuild()

    def do_who_calls() -> None:
        who_calls("top_func", db_path=db_path)

    benchmark(do_who_calls)


@pytest.mark.unit
def test_index_size_budget(simple_repo: Path) -> None:
    db_path = simple_repo / ".codemap" / "index.db"
    with CodeMapIndex(simple_repo, db_path=db_path) as idx:
        idx.rebuild()

    size_mb = db_path.stat().st_size / (1024 * 1024)
    assert size_mb < 30.0, f"Index size {size_mb:.2f} MB exceeds 30 MB budget"
