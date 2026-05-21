"""Tests for scripts/ci/check_lod.py."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.ci import check_lod as mod


def test_attr_chain_basic() -> None:
    tree = ast.parse("a.b.c.d", mode="eval")
    assert mod._attr_chain(tree.body) == ["a", "b", "c", "d"]


def test_attr_chain_returns_none_for_call_root() -> None:
    tree = ast.parse("f().x.y", mode="eval")
    assert mod._attr_chain(tree.body) is None


def test_is_library_chain_leaf() -> None:
    assert mod._is_library_chain(["self", "data", "tolist"])
    assert mod._is_library_chain(["self", "btn", "clicked", "connect"])


def test_is_library_chain_intermediate() -> None:
    assert mod._is_library_chain(["self", "btn", "clicked", "x"])


def test_is_library_chain_namespace_root() -> None:
    assert mod._is_library_chain(["np", "linalg", "norm", "x"])
    assert mod._is_library_chain(["QtCore", "Qt", "Orientation", "Horizontal"])


def test_is_library_chain_navigation_not_library() -> None:
    assert not mod._is_library_chain(["self", "a", "b", "c"])


def test_is_library_chain_empty() -> None:
    assert mod._is_library_chain([])


def test_check_file_detects_violation(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("self.a.b.c.d\n")
    violations = mod.check_file(p)
    assert violations
    assert violations[0][1] == "self.a.b.c.d"


def test_check_file_ignores_2_hops(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("self.a.b\n")
    assert mod.check_file(p) == []


def test_check_file_ignores_library(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("self.btn.clicked.connect(handler)\n")
    assert mod.check_file(p) == []


def test_check_file_syntax_error(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("def f(:\n")
    assert mod.check_file(p) == []


def test_check_file_unicode_error(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_bytes(b"\xff\xfe\x00bogus")
    assert mod.check_file(p) == []


def test_iter_python_files_skips_tests_and_examples(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x=1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "b.py").write_text("x=1\n")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "c.py").write_text("x=1\n")
    files = mod.iter_python_files(tmp_path)
    names = {p.name for p in files}
    assert "a.py" in names
    assert "b.py" not in names
    assert "c.py" not in names


def test_main_missing_root(capsys: pytest.CaptureFixture[str]) -> None:
    ret = mod.main(["/definitely/does/not/exist/xyz123"])
    assert ret == 2
    assert "does not exist" in capsys.readouterr().err


def test_main_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    assert mod.main([str(tmp_path)]) == 0
    assert "clean" in capsys.readouterr().out


def test_main_finds_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.py").write_text("self.a.b.c.d\n")
    assert mod.main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "LOD chain" in captured.out


def test_main_advisory_returns_zero(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("self.a.b.c.d\n")
    assert mod.main([str(tmp_path), "--advisory"]) == 0
