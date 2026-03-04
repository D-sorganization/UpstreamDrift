"""Tests for code_quality_check.py"""
from pathlib import Path
from src.tools.code_quality_check import (
    is_legitimate_pass_context,
    check_banned_patterns,
    check_magic_numbers,
    check_ast_issues,
)

def test_is_legitimate_pass_context():
    lines = [
        "def foo():",
        "    try:",
        "        pass",
        "    except Exception:",
        "        pass",
        "class Bar:",
        "    pass"
    ]
    # Line 3 'pass' inside try:
    assert is_legitimate_pass_context(lines, 3) is True
    # Line 5 'pass' inside except:
    assert is_legitimate_pass_context(lines, 5) is True
    # Line 7 'pass' inside class:
    assert is_legitimate_pass_context(lines, 7) is True

def test_check_banned_patterns():
    lines = [
        "# TODO: fix this",
        "def test():",
        "    ...  ",
        "    pass"
    ]
    path = Path("test_file.py")
    issues = check_banned_patterns(lines, path)
    
    assert len(issues) >= 2
    types = [issue[1] for issue in issues]
    assert any("TODO placeholder" in t for t in types)
    assert any("Ellipsis placeholder" in t for t in types)

def test_check_magic_numbers():
    lines = [
        "x = 3.141 * r",
        "y = 9.8 * m",
        "z = 6.67 * x"
    ]
    path = Path("test_file.py")
    issues = check_magic_numbers(lines, path)
    assert len(issues) == 3
    
def test_check_ast_issues():
    content = "def missing_docstring():\n    return 1"
    path = Path("test_file.py")
    issues = check_ast_issues(content, path)
    assert len(issues) == 1
    assert "missing docstring" in issues[0][1]
