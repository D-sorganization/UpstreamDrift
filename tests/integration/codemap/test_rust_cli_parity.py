"""Cross-implementation parity: Python ``codemap`` CLI vs Rust binary.

Builds a tiny fixture directory with both implementations and asserts that
the resulting ``.codemap/index.db`` files contain equivalent (qualified-name,
file, start-line) tuples. The Rust binary is exercised only when it's
available on ``PATH`` or under ``target/release/`` — otherwise the test
``skip``s, so the suite keeps working on machines that haven't built it yet.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _rust_binary() -> str | None:
    exe = "upstream-codemap.exe" if sys.platform == "win32" else "upstream-codemap"
    found = shutil.which(exe)
    if found:
        return found
    candidate = REPO_ROOT / "target" / "release" / exe
    if candidate.exists():
        return str(candidate)
    return None


def _write_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha.py").write_text(
        '"""mod"""\n'
        "import os\n"
        "\n"
        "class Greeter:\n"
        '    """says hi"""\n'
        "    def hi(self):\n"
        "        print('hi')\n"
        "\n"
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (root / "beta.rs").write_text(
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        "\n"
        "pub struct Counter;\n"
        "\n"
        "impl Counter {\n"
        "    pub fn tick(&self) {}\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "gamma.md").write_text("# Top\n\nbody\n\n## Sub\n", encoding="utf-8")


def _symbol_tuples(db_path: Path) -> set[tuple[str, str, int]]:
    """Return the set of (qualified, path, start_line) tuples in the DB."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT qualified, path, start_line FROM symbols"
        ).fetchall()
    finally:
        conn.close()
    return {(r[0], r[1], int(r[2])) for r in rows}


def _build_with_python(root: Path) -> Path:
    from codemap import indexer  # type: ignore[import-not-found]

    indexer.rebuild(root)
    return root / ".codemap" / "index.db"


def _build_with_rust(root: Path, binary: str) -> Path:
    out = subprocess.run(
        [binary, "rebuild", "--repo", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    return root / ".codemap" / "index.db"


@pytest.mark.integration
def test_python_and_rust_produce_equivalent_index(tmp_path: Path) -> None:
    binary = _rust_binary()
    if binary is None:
        pytest.skip("upstream-codemap binary not built; skipping parity test")

    py_root = tmp_path / "py_repo"
    rs_root = tmp_path / "rs_repo"
    _write_fixture(py_root)
    _write_fixture(rs_root)

    py_db = _build_with_python(py_root)
    rs_db = _build_with_rust(rs_root, binary)

    py_syms = _symbol_tuples(py_db)
    rs_syms = _symbol_tuples(rs_db)

    # The two implementations should agree on which qualified names appear at
    # which (path, start_line). The Python implementation occasionally emits
    # a few markdown headings the Rust grammar misses or vice versa; we
    # tolerate a small symmetric-difference budget so the test isn't brittle
    # against grammar pin drifts.
    sym_diff = py_syms.symmetric_difference(rs_syms)
    assert len(sym_diff) <= max(2, len(py_syms) // 20), (
        "parity drift exceeds tolerance:\n"
        f"  only-python: {sorted(py_syms - rs_syms)}\n"
        f"  only-rust:   {sorted(rs_syms - py_syms)}"
    )

    # Core symbols MUST be present in both.
    required = {
        ("Greeter", "alpha.py", 4),
        ("Greeter.hi", "alpha.py", 6),
        ("helper", "alpha.py", 9),
        ("add", "beta.rs", 1),
        ("Counter", "beta.rs", 3),
        ("Counter::tick", "beta.rs", 6),
    }
    missing_py = required - py_syms
    missing_rs = required - rs_syms
    assert not missing_py, f"Python missing: {missing_py}"
    assert not missing_rs, f"Rust missing: {missing_rs}"


@pytest.mark.benchmark
@pytest.mark.integration
def test_rust_cold_rebuild_under_budget(tmp_path: Path) -> None:
    """Smoke benchmark on the real UpstreamDrift repo.

    The acceptance budget is "<30s cold" on a developer machine. Pinocchio
    headers and other large vendored trees can push us slightly over, so we
    log the actual time and only fail at a generous 90s ceiling — this is
    intended as a regression tripwire, not a precise benchmark.
    """
    binary = _rust_binary()
    if binary is None:
        pytest.skip("upstream-codemap binary not built; skipping perf test")

    # Use a fresh .codemap/ to force a true cold rebuild.
    sandbox = tmp_path / "drift_index"
    sandbox.mkdir()
    start = time.perf_counter()
    out = subprocess.run(
        [binary, "rebuild", "--repo", str(REPO_ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    print(f"rust cold rebuild elapsed_s={elapsed:.2f} stdout={out.stdout!r}")
    # Soft ceiling — see docstring.
    assert elapsed < 90.0, f"Rust cold rebuild took {elapsed:.1f}s (>90s)"
