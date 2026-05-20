"""Comprehensive tests for matlab_quality_check.py — file analysis, no MATLAB engine."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.matlab_utilities.scripts.matlab_quality_check import (
    MATLABQualityChecker,
    main,
)


# ─── Constructor / DbC ──────────────────────────────────────────


def test_init_sets_paths(tmp_path: Path) -> None:
    c = MATLABQualityChecker(tmp_path)
    assert c.project_root == tmp_path
    assert c.matlab_dir == tmp_path / "matlab"
    assert c.results["passed"] is True
    assert c.results["total_files"] == 0
    assert c.results["issues"] == []


def test_init_requires_project_root() -> None:
    with pytest.raises(ValueError, match="project_root"):
        MATLABQualityChecker(None)  # type: ignore[arg-type]


# ─── _check_docstring_and_args ──────────────────────────────────


def _checker(tmp_path: Path) -> MATLABQualityChecker:
    return MATLABQualityChecker(tmp_path)


def test_docstring_present(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    lines = [
        "function out = foo(x)",
        "% This is a docstring describing foo",
        "    arguments",
        "        x double",
        "    end",
        "    out = x;",
        "end",
    ]
    issues: list[str] = []
    c._check_docstring_and_args(lines, 1, "foo.m", issues)
    assert issues == []


def test_docstring_missing(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    lines = ["function out = foo(x)", "out = x;", "end"]
    issues: list[str] = []
    c._check_docstring_and_args(lines, 1, "foo.m", issues)
    assert any("Missing function docstring" in i for i in issues)
    assert any("arguments validation" in i for i in issues)


def test_docstring_short_does_not_count(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    lines = ["function foo()", "%a", "end"]
    issues: list[str] = []
    c._check_docstring_and_args(lines, 1, "f.m", issues)
    assert any("Missing function docstring" in i for i in issues)


def test_docstring_check_requires_lines(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    with pytest.raises(ValueError, match="lines"):
        c._check_docstring_and_args(None, 1, "f.m", [])  # type: ignore[arg-type]


# ─── _check_banned_patterns ─────────────────────────────────────


@pytest.mark.parametrize(
    "line,expected",
    [
        ("% TO" + "DO: fix later", "Placeholder marker (TO-DO)"),
        ("% FIX" + "ME urgently", "Placeholder marker (FIX-ME)"),
        ("% HACK around bug", "HACK"),
        ("% XXX dangerous", "XXX"),
        ("name = <PLACEHOLDER>;", "Angle bracket placeholder"),
        ("text = '{{value}}';", "Template placeholder"),
    ],
)
def test_banned_patterns_detected(tmp_path: Path, line: str, expected: str) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_banned_patterns(line, 5, "f.m", issues)
    assert any(expected in i for i in issues)


def test_banned_patterns_clean(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_banned_patterns("y = sin(x);", 1, "f.m", issues)
    assert issues == []


def test_banned_patterns_requires_line(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    with pytest.raises(ValueError, match="line"):
        c._check_banned_patterns(None, 1, "f.m", [])  # type: ignore[arg-type]


# ─── _check_anti_patterns ───────────────────────────────────────


@pytest.mark.parametrize(
    "line,expected",
    [
        ("result = eval('1+1');", "eval()"),
        ("assignin('base', 'x', 1);", "assignin()"),
        ("evalin('caller', 'x');", "evalin()"),
        ("global myvar", "Global variable"),
        ("if exist('x', 'var')", "exist()"),
    ],
)
def test_anti_patterns_detected(tmp_path: Path, line: str, expected: str) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_anti_patterns(line, 1, "f.m", issues)
    assert any(expected in i for i in issues)


def test_load_without_output_detected(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_anti_patterns("load mydata", 1, "f.m", issues)
    assert any("load without output" in i for i in issues)
    issues = []
    c._check_anti_patterns("load('data.mat')", 1, "f.m", issues)
    assert any("load without output" in i for i in issues)


def test_load_with_output_ok(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_anti_patterns("data = load('x.mat')", 1, "f.m", issues)
    assert not any("load without output" in i for i in issues)


def test_anti_patterns_requires_line(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    with pytest.raises(ValueError, match="line"):
        c._check_anti_patterns(None, 1, "f.m", [])  # type: ignore[arg-type]


# ─── _check_magic_numbers ───────────────────────────────────────


def test_magic_number_detected(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_magic_numbers("threshold = 42.7;", 1, "f.m", issues)
    assert any("Magic number 42.7" in i for i in issues)


def test_magic_number_acceptable_ignored(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_magic_numbers("x = 0; y = 1; z = 2.0;", 1, "f.m", issues)
    assert issues == []


def test_magic_number_with_comment_explanation(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    # Number before the % — treated as having a trailing-comment explanation, not flagged
    c._check_magic_numbers("x = 42; % explanation", 1, "f.m", issues)
    assert issues == []
    # Number inside the comment portion — flagged (number comes after %)
    issues = []
    c._check_magic_numbers("x = 1; % see RFC 42 for details", 1, "f.m", issues)
    assert any("Magic number 42" in i for i in issues)


def test_magic_numbers_requires_line(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    with pytest.raises(ValueError, match="line"):
        c._check_magic_numbers(None, 1, "f.m", [])  # type: ignore[arg-type]


# ─── _check_function_unsafe ─────────────────────────────────────


@pytest.mark.parametrize(
    "line,expected",
    [
        ("clear all", "clear all/global"),
        ("clear global", "clear all/global"),
        ("clear", "Avoid 'clear' in functions"),
        ("clc", "clc"),
        ("close all", "close all"),
        ("addpath('foo')", "addpath"),
    ],
)
def test_function_unsafe_detected(tmp_path: Path, line: str, expected: str) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_function_unsafe(line, 1, "f.m", issues)
    assert any(expected in i for i in issues)


def test_function_unsafe_clear_with_var_ok(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    issues: list[str] = []
    c._check_function_unsafe("clear x", 1, "f.m", issues)
    # 'clear x' should not trigger the bare 'clear' rule
    assert not any("Avoid 'clear' in functions" in i for i in issues)


def test_function_unsafe_requires_line(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    with pytest.raises(ValueError, match="line"):
        c._check_function_unsafe(None, 1, "f.m", [])  # type: ignore[arg-type]


# ─── _analyze_matlab_file ───────────────────────────────────────


def test_analyze_clean_file(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    f = tmp_path / "clean.m"
    f.write_text(
        "function out = clean(x)\n"
        "% Compute clean result from input x.\n"
        "    arguments\n"
        "        x double\n"
        "    end\n"
        "    out = x;\n"
        "end\n"
    )
    issues = c._analyze_matlab_file(f)
    assert issues == []


def test_analyze_dirty_file(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    f = tmp_path / "dirty.m"
    f.write_text(
        "function out = dirty(x)\n"
        "    eval('disp(1)');\n"
        "    clear all\n"
        "    out = 42.7;\n"
        "end\n"
    )
    issues = c._analyze_matlab_file(f)
    text = "\n".join(issues)
    assert "Missing function docstring" in text
    assert "eval()" in text
    assert "clear all/global" in text
    assert "Magic number 42.7" in text


def test_analyze_file_handles_oserror(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    bad = tmp_path / "missing.m"
    with patch.object(Path, "read_text", side_effect=OSError("boom")):
        issues = c._analyze_matlab_file(bad)
    assert any("Failed analysis" in i for i in issues)


def test_analyze_requires_file_path(tmp_path: Path) -> None:
    c = _checker(tmp_path)
    with pytest.raises(ValueError, match="file_path"):
        c._analyze_matlab_file(None)  # type: ignore[arg-type]


def test_analyze_tracks_function_scope(tmp_path: Path) -> None:
    """Outside function, 'clear all' should NOT be flagged by function-unsafe check."""
    c = _checker(tmp_path)
    f = tmp_path / "script.m"
    # Script-level (no `function` keyword) — clear all at top should not be flagged
    f.write_text("clear all\nx = 1;\n")
    issues = c._analyze_matlab_file(f)
    assert not any("clear all/global" in i for i in issues)


# ─── run_all_checks ─────────────────────────────────────────────


def test_run_all_checks_no_matlab_dir(tmp_path: Path) -> None:
    c = MATLABQualityChecker(tmp_path)
    out = c.run_all_checks()
    assert out["passed"] is True
    assert "No MATLAB dir" in out["summary"]


def test_run_all_checks_pass(tmp_path: Path) -> None:
    mdir = tmp_path / "matlab"
    mdir.mkdir()
    (mdir / "ok.m").write_text(
        "function y = ok(x)\n"
        "% Identity function.\n"
        "    arguments\n"
        "        x double\n"
        "    end\n"
        "    y = x;\n"
        "end\n"
    )
    c = MATLABQualityChecker(tmp_path)
    out = c.run_all_checks()
    assert out["total_files"] == 1
    assert out["passed"] is True
    assert "[PASS]" in out["summary"]


def test_run_all_checks_fail(tmp_path: Path) -> None:
    mdir = tmp_path / "matlab"
    mdir.mkdir()
    (mdir / "bad.m").write_text("function bad()\n    eval('1');\nend\n")
    c = MATLABQualityChecker(tmp_path)
    out = c.run_all_checks()
    assert out["passed"] is False
    assert "[FAIL]" in out["summary"]
    assert len(out["issues"]) > 0


def test_run_all_checks_recursive(tmp_path: Path) -> None:
    mdir = tmp_path / "matlab" / "sub"
    mdir.mkdir(parents=True)
    (mdir / "x.m").write_text("function x()\n% doc text\n    arguments\n    end\nend\n")
    c = MATLABQualityChecker(tmp_path)
    out = c.run_all_checks()
    assert out["total_files"] == 1


# ─── main / CLI ─────────────────────────────────────────────────


def test_main_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--project-root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_main_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mdir = tmp_path / "matlab"
    mdir.mkdir()
    (mdir / "bad.m").write_text("function bad()\n    eval('1');\nend\n")
    monkeypatch.setattr(sys, "argv", ["prog", "--project-root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
