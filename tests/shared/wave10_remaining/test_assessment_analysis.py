"""Tests for src.shared.python.assessment.analysis."""

from __future__ import annotations

import pytest

from src.shared.python.assessment.analysis import (
    assess_error_handling_content,
    assess_logging_content,
    calculate_complexity,
    classify_assessment_category,
    count_files,
    get_detailed_function_metrics,
    get_python_metrics,
    grep_count,
)


# ---------------------------------------------------------------------------
# get_python_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_python_metrics_counts(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(
        '"""Mod doc."""\n'
        "def f(x: int) -> int:\n"
        '    """fn doc."""\n'
        "    if x > 0:\n"
        "        return x\n"
        "    return 0\n"
        "class C:\n"
        '    """class doc."""\n'
        "    def m(self):\n"
        "        for i in []: pass\n",
        encoding="utf-8",
    )
    m = get_python_metrics(src)
    assert m["functions"] == 2
    assert m["classes"] == 1
    assert m["docstrings"] >= 2
    assert m["typed_returns"] == 1
    assert m["branches"] >= 2


@pytest.mark.unit
def test_get_python_metrics_bad_syntax(tmp_path):
    src = tmp_path / "bad.py"
    src.write_text("def def def", encoding="utf-8")
    m = get_python_metrics(src)
    assert m == {
        "functions": 0,
        "classes": 0,
        "docstrings": 0,
        "typed_returns": 0,
        "branches": 0,
    }


@pytest.mark.unit
def test_get_python_metrics_missing_file(tmp_path):
    m = get_python_metrics(tmp_path / "nope.py")
    assert m["functions"] == 0


# ---------------------------------------------------------------------------
# calculate_complexity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_calculate_complexity_zero_functions():
    assert calculate_complexity({"functions": 0, "branches": 5}) == 0.0


@pytest.mark.unit
def test_calculate_complexity_normal():
    assert calculate_complexity({"functions": 2, "branches": 10}) == 5.0


@pytest.mark.unit
def test_calculate_complexity_missing_keys():
    assert calculate_complexity({}) == 0.0


# ---------------------------------------------------------------------------
# assess_error_handling_content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assess_error_handling_counts():
    content = "try:\n    pass\nexcept:\n    pass\ntry:\n    pass\nexcept ValueError:\n    pass"
    r = assess_error_handling_content(content)
    assert r["try_count"] == 2
    assert r["bare_except_count"] == 1


@pytest.mark.unit
def test_assess_error_handling_empty():
    r = assess_error_handling_content("")
    assert r == {"try_count": 0, "bare_except_count": 0}


# ---------------------------------------------------------------------------
# assess_logging_content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assess_logging_counts_logger_and_print():
    content = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "logger.info('hi')\n"
        "print('debug')\n"
        "print(1, 2)\n"
    )
    r = assess_logging_content(content)
    assert r["print_usage"] == 2
    assert r["logging_usage"] >= 2


@pytest.mark.unit
def test_assess_logging_falls_back_on_syntax_error():
    content = "def def\nprint('x')\nprint(1)"
    r = assess_logging_content(content)
    assert r["print_usage"] == 2


@pytest.mark.unit
def test_assess_logging_ignores_print_attribute():
    # `obj.print(...)` should not be counted by the AST path
    content = "obj.print('x')\nprint('y')"
    r = assess_logging_content(content)
    assert r["print_usage"] == 1


# ---------------------------------------------------------------------------
# get_detailed_function_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_detailed_function_metrics():
    content = (
        "def alpha(a, b):\n"
        '    """alpha"""\n'
        "    return a + b\n"
        "async def beta():\n"
        "    return 1\n"
    )
    out = get_detailed_function_metrics(content)
    names = {f["name"] for f in out}
    assert names == {"alpha", "beta"}
    alpha = next(f for f in out if f["name"] == "alpha")
    assert alpha["args"] == 2
    assert alpha["has_docstring"] is True
    beta = next(f for f in out if f["name"] == "beta")
    assert beta["has_docstring"] is False


@pytest.mark.unit
def test_get_detailed_function_metrics_bad_syntax():
    out = get_detailed_function_metrics("def def def")
    assert out == []


# ---------------------------------------------------------------------------
# count_files / grep_count
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_count_files(tmp_path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "c.txt").write_text("c")
    assert count_files(tmp_path, "*.py") == 2
    assert count_files(tmp_path, "*.md") == 0


@pytest.mark.unit
def test_grep_count_finds_pattern(tmp_path):
    (tmp_path / "a.py").write_text("hello world")
    (tmp_path / "b.py").write_text("nothing here")
    (tmp_path / "c.py").write_text("Hello there")
    assert grep_count(tmp_path, r"hello") == 1
    assert grep_count(tmp_path, r"(?i)hello") == 2


@pytest.mark.unit
def test_grep_count_default_exclude_parts_no_crash(tmp_path):
    """Regression: ``exclude_parts=None`` (the default) used to raise TypeError."""
    (tmp_path / "a.py").write_text("hello")
    assert grep_count(tmp_path, r"hello") == 1


@pytest.mark.unit
def test_grep_count_with_excluded_dir(tmp_path):
    (tmp_path / "a.py").write_text("needle")
    sub = tmp_path / "skip_me"
    sub.mkdir()
    (sub / "b.py").write_text("needle")
    n = grep_count(tmp_path, r"needle", exclude_parts=["skip_me"])
    assert n == 1


@pytest.mark.unit
def test_grep_count_requires_root():
    with pytest.raises(ValueError, match="root must be provided"):
        grep_count(None, "x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# classify_assessment_category
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,desc,expected",
    [
        ("A", "", "Architecture"),
        ("B", "", "Code Quality"),
        ("C", "", "Documentation"),
        ("D", "", "User Experience"),
        ("E", "", "Performance"),
        ("F", "", "Installation"),
        ("G", "", "Testing"),
        ("H", "", "Error Handling"),
        ("I", "", "Security"),
        ("J", "", "Extensibility"),
        ("K", "", "Reproducibility"),
        ("L", "", "Maintainability"),
        ("N", "", "Visualization"),
        ("O", "", "CI/CD"),
        ("unknown", "", "General"),
    ],
)
def test_classify_assessment_category_letter_codes(name, desc, expected):
    assert classify_assessment_category(name, desc) == expected


@pytest.mark.unit
def test_classify_assessment_category_keyword_match():
    assert (
        classify_assessment_category("system", "implementation review")
        == "Architecture"
    )
    assert classify_assessment_category("ux audit", "") == "User Experience"
    assert (
        classify_assessment_category("perf", "performance bottleneck") == "Performance"
    )


@pytest.mark.unit
def test_classify_assessment_category_requires_source_name():
    with pytest.raises(ValueError, match="source_name must be provided"):
        classify_assessment_category(None)  # type: ignore[arg-type]
