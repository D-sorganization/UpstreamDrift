"""Comprehensive tests for code_quality_check.py — full coverage including DbC and edge cases."""

from pathlib import Path

import pytest

from src.shared.python.contracts import PreconditionError
from src.tools.code_quality_check import (
    check_ast_issues,
    check_banned_patterns,
    check_file,
    check_magic_numbers,
    is_legitimate_pass_context,
)

# ─── is_legitimate_pass_context ────────────────────────────────


def test_is_legitimate_pass_context():
    lines = [
        "def foo():",
        "    try:",
        "        pass",
        "    except Exception:",
        "        pass",
        "class Bar:",
        "    pass",
    ]
    assert is_legitimate_pass_context(lines, 3) is True  # inside try
    assert is_legitimate_pass_context(lines, 5) is True  # inside except
    assert is_legitimate_pass_context(lines, 7) is True  # inside class


def test_legitimate_pass_in_with_block():
    lines = ["with open('f') as f:", "    pass"]
    assert is_legitimate_pass_context(lines, 2) is True


def test_legitimate_pass_line_too_large():
    lines = ["pass"]
    assert is_legitimate_pass_context(lines, 99) is False


def test_legitimate_pass_line_zero():
    lines = ["pass"]
    assert is_legitimate_pass_context(lines, 0) is False


def test_non_pass_line_returns_false():
    lines = ["x = 1", "y = 2"]
    assert is_legitimate_pass_context(lines, 1) is False


def test_pass_inside_function_is_not_legitimate():
    lines = ["def foo():", "    pass"]
    # Inside a function def → NOT legitimate (a stub function)
    assert is_legitimate_pass_context(lines, 2) is False


def test_is_legitimate_pass_dbc_non_list():
    with pytest.raises(PreconditionError):
        is_legitimate_pass_context("not a list", 1)  # type: ignore[arg-type]


def test_is_legitimate_pass_dbc_non_int():
    with pytest.raises(PreconditionError):
        is_legitimate_pass_context(["pass"], "1")  # type: ignore[arg-type]


# ─── check_banned_patterns ─────────────────────────────────────


def test_check_banned_patterns():
    lines = ["# TODO: fix this", "def test():", "    ...  ", "    pass"]
    issues = check_banned_patterns(lines, Path("test_file.py"))
    assert len(issues) >= 2
    types = [issue[1] for issue in issues]
    assert any("TODO placeholder" in t for t in types)
    assert any("Ellipsis placeholder" in t for t in types)


def test_check_banned_patterns_fixme():
    lines = ["# FIXME: broken logic"]
    issues = check_banned_patterns(lines, Path("test.py"))
    assert any("FIXME" in i[1] for i in issues)


def test_check_banned_patterns_not_implemented_error():
    lines = ["    raise NotImplementedError"]
    issues = check_banned_patterns(lines, Path("test.py"))
    assert any("NotImplementedError" in i[1] for i in issues)


def test_check_banned_patterns_template_placeholder():
    lines = ["    # Insert your code here"]
    issues = check_banned_patterns(lines, Path("test.py"))
    assert any("placeholder" in i[1].lower() for i in issues)


def test_check_banned_patterns_skips_quality_scripts():
    """Self-referential quality check scripts must be excluded."""
    lines = ["# TODO: internal marker"]
    issues = check_banned_patterns(lines, Path("code_quality_check.py"))
    assert issues == []


def test_check_banned_patterns_clean():
    lines = ["def add(x, y):", '    """Add two numbers."""', "    return x + y"]
    issues = check_banned_patterns(lines, Path("math_utils.py"))
    assert issues == []


def test_check_banned_patterns_dbc_non_list():
    with pytest.raises(PreconditionError):
        check_banned_patterns("not a list", Path("test.py"))  # type: ignore[arg-type]


def test_check_banned_patterns_dbc_non_path():
    with pytest.raises(PreconditionError):
        check_banned_patterns([], "/not/a/path")  # type: ignore[arg-type]


# ─── check_magic_numbers ───────────────────────────────────────


def test_check_magic_numbers():
    lines = ["x = 3.141 * r", "y = 9.8 * m", "z = 6.67 * x"]
    issues = check_magic_numbers(lines, Path("test_file.py"))
    assert len(issues) == 3


def test_check_magic_numbers_in_comment_ignored():
    """Magic numbers only inside comments should not be flagged."""
    lines = ["# constant is 3.141 for pi"]
    issues = check_magic_numbers(lines, Path("test.py"))
    # The number is entirely in the comment, should not be flagged
    assert issues == []


def test_check_magic_numbers_skips_quality_scripts():
    lines = ["x = 3.141 * r"]
    issues = check_magic_numbers(lines, Path("code_quality_check.py"))
    assert issues == []


def test_check_magic_numbers_gravity():
    lines = ["g = 9.81"]
    issues = check_magic_numbers(lines, Path("physics.py"))
    assert any("GRAVITY" in i[1] for i in issues)


def test_check_magic_numbers_dbc_non_list():
    with pytest.raises(PreconditionError):
        check_magic_numbers("not a list", Path("test.py"))  # type: ignore[arg-type]


def test_check_magic_numbers_dbc_non_path():
    with pytest.raises(PreconditionError):
        check_magic_numbers([], "/not/a/path")  # type: ignore[arg-type]


# ─── check_ast_issues ──────────────────────────────────────────


def test_check_ast_issues():
    content = "def missing_docstring():\n    return 1"
    issues = check_ast_issues(content, Path("test_file.py"))
    assert len(issues) == 1
    assert "missing docstring" in issues[0][1]


def test_check_ast_issues_with_docstring():
    content = 'def good():\n    """Has docstring."""\n    return 1'
    issues = check_ast_issues(content, Path("test.py"))
    assert not any("missing docstring" in i[1] for i in issues)


def test_check_ast_issues_syntax_error():
    content = "def :"
    issues = check_ast_issues(content, Path("bad.py"))
    assert any("Syntax error" in i[1] for i in issues)


def test_check_ast_issues_skips_quality_scripts():
    content = "def no_docstring():\n    pass"
    issues = check_ast_issues(content, Path("code_quality_check.py"))
    assert issues == []


def test_check_ast_issues_dbc_non_string():
    with pytest.raises(PreconditionError):
        check_ast_issues(123, Path("test.py"))  # type: ignore[arg-type]


def test_check_ast_issues_dbc_non_path():
    with pytest.raises(PreconditionError):
        check_ast_issues("x = 1", "/not/a/path")  # type: ignore[arg-type]


# ─── check_file ────────────────────────────────────────────────


def test_check_file_clean(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text('def add(x, y):\n    """Add two numbers."""\n    return x + y\n')
    issues = check_file(f)
    assert issues == []


def test_check_file_with_todo(tmp_path):
    f = tmp_path / "todo.py"
    f.write_text("# TODO: fix this\n")
    issues = check_file(f)
    assert any("TODO" in i[1] for i in issues)


def test_check_file_dbc_non_path():
    with pytest.raises(PreconditionError):
        check_file("/not/a/path/object")  # type: ignore[arg-type]


def test_check_file_dbc_missing():
    with pytest.raises(PreconditionError):
        check_file(Path("/nonexistent/file.py"))


def test_check_file_dbc_directory(tmp_path):
    with pytest.raises(PreconditionError):
        check_file(tmp_path)  # directory, not a file
