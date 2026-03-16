"""Unit tests for shared/python/assessment/analysis.py and assessment/reporting.py."""

from __future__ import annotations

from pathlib import Path

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
# Helpers / fixtures
# ---------------------------------------------------------------------------

SIMPLE_PYTHON = """\
def add(x: int, y: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return x + y


def silently() -> None:
    pass


class Foo:
    \"\"\"A class.\"\"\"
    def method(self) -> str:
        if True:
            return "yes"
        return "no"
"""

ERROR_PYTHON = """\
def danger():
    try:
        pass
    except:
        pass
    try:
        pass
    except ValueError:
        pass
"""

LOGGING_PYTHON = """\
import logging as logging
logger = logging.getLogger(__name__)
logger.info("hello")
print("debug msg")
"""


@pytest.fixture
def tmp_py_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.py"
    f.write_text(SIMPLE_PYTHON, encoding="utf-8")
    return f


@pytest.fixture
def tmp_dir_with_files(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")
    (tmp_path / "c.txt").write_text("z = 3", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# get_python_metrics
# ---------------------------------------------------------------------------


class TestGetPythonMetrics:
    def test_counts_functions(self, tmp_py_file: Path) -> None:
        m = get_python_metrics(tmp_py_file)
        assert m["functions"] == 3  # add, silently, method

    def test_counts_classes(self, tmp_py_file: Path) -> None:
        m = get_python_metrics(tmp_py_file)
        assert m["classes"] == 1  # Foo

    def test_counts_docstrings(self, tmp_py_file: Path) -> None:
        m = get_python_metrics(tmp_py_file)
        # add(1), silently(0), Foo(1), method(0) → 2
        assert m["docstrings"] == 2

    def test_counts_typed_returns(self, tmp_py_file: Path) -> None:
        m = get_python_metrics(tmp_py_file)
        assert m["typed_returns"] == 3  # add, silently, method

    def test_counts_branches(self, tmp_py_file: Path) -> None:
        m = get_python_metrics(tmp_py_file)
        assert m["branches"] >= 1  # at least the if True

    def test_nonexistent_file_returns_zero_metrics(self) -> None:
        m = get_python_metrics(Path("/nonexistent/path.py"))
        assert m["functions"] == 0
        assert m["classes"] == 0

    def test_invalid_syntax_returns_zero_metrics(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.py"
        bad.write_text("def (syntax error:", encoding="utf-8")
        m = get_python_metrics(bad)
        assert m["functions"] == 0


# ---------------------------------------------------------------------------
# calculate_complexity
# ---------------------------------------------------------------------------


class TestCalculateComplexity:
    def test_zero_functions_returns_zero(self) -> None:
        assert calculate_complexity({"functions": 0, "branches": 10}) == 0.0

    def test_correct_ratio(self) -> None:
        result = calculate_complexity({"functions": 4, "branches": 12})
        assert abs(result - 3.0) < 1e-9

    def test_no_branches(self) -> None:
        assert calculate_complexity({"functions": 5, "branches": 0}) == 0.0


# ---------------------------------------------------------------------------
# assess_error_handling_content
# ---------------------------------------------------------------------------


class TestAssessErrorHandlingContent:
    def test_counts_try(self) -> None:
        d = assess_error_handling_content(ERROR_PYTHON)
        assert d["try_count"] == 2

    def test_counts_bare_except(self) -> None:
        d = assess_error_handling_content(ERROR_PYTHON)
        assert d["bare_except_count"] == 1

    def test_no_patterns(self) -> None:
        d = assess_error_handling_content("x = 1 + 2")
        assert d["try_count"] == 0
        assert d["bare_except_count"] == 0


# ---------------------------------------------------------------------------
# assess_logging_content
# ---------------------------------------------------------------------------


class TestAssessLoggingContent:
    def test_counts_logging(self) -> None:
        d = assess_logging_content(LOGGING_PYTHON)
        assert d["logging_usage"] >= 2  # logging.getLogger + logger.info

    def test_counts_print(self) -> None:
        d = assess_logging_content(LOGGING_PYTHON)
        assert d["print_usage"] == 1

    def test_no_logging(self) -> None:
        d = assess_logging_content("x = 42")
        assert d["logging_usage"] == 0
        assert d["print_usage"] == 0


# ---------------------------------------------------------------------------
# get_detailed_function_metrics
# ---------------------------------------------------------------------------


class TestGetDetailedFunctionMetrics:
    def test_returns_list(self) -> None:
        result = get_detailed_function_metrics(SIMPLE_PYTHON)
        assert isinstance(result, list)

    def test_finds_add_function(self) -> None:
        result = get_detailed_function_metrics(SIMPLE_PYTHON)
        names = [f["name"] for f in result]
        assert "add" in names

    def test_function_has_correct_args_count(self) -> None:
        result = get_detailed_function_metrics(SIMPLE_PYTHON)
        add = next(f for f in result if f["name"] == "add")
        assert add["args"] == 2

    def test_has_docstring_flag(self) -> None:
        result = get_detailed_function_metrics(SIMPLE_PYTHON)
        add = next(f for f in result if f["name"] == "add")
        assert add["has_docstring"] is True
        silently = next(f for f in result if f["name"] == "silently")
        assert silently["has_docstring"] is False

    def test_invalid_syntax_returns_empty(self) -> None:
        result = get_detailed_function_metrics("def (: broken")
        assert result == []


# ---------------------------------------------------------------------------
# count_files
# ---------------------------------------------------------------------------


class TestCountFiles:
    def test_counts_py_files(self, tmp_dir_with_files: Path) -> None:
        n = count_files(tmp_dir_with_files, "*.py")
        assert n == 2

    def test_counts_txt_files(self, tmp_dir_with_files: Path) -> None:
        n = count_files(tmp_dir_with_files, "*.txt")
        assert n == 1

    def test_no_match(self, tmp_dir_with_files: Path) -> None:
        n = count_files(tmp_dir_with_files, "*.rb")
        assert n == 0


# ---------------------------------------------------------------------------
# grep_count
# ---------------------------------------------------------------------------


class TestGrepCount:
    def test_finds_pattern(self, tmp_dir_with_files: Path) -> None:
        n = grep_count(tmp_dir_with_files, r"x = 1", "**/*.py")
        assert n == 1

    def test_no_match(self, tmp_dir_with_files: Path) -> None:
        n = grep_count(tmp_dir_with_files, r"NOT_IN_ANY_FILE", "**/*.py")
        assert n == 0

    def test_asserts_root(self) -> None:
        with pytest.raises(AssertionError):
            grep_count(None, "x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# classify_assessment_category
# ---------------------------------------------------------------------------


class TestClassifyAssessmentCategory:
    @pytest.mark.parametrize(
        "source, expected",
        [
            ("A", "Architecture"),
            ("B", "Code Quality"),
            ("C", "Documentation"),
            ("D", "User Experience"),
            ("E", "Performance"),
            ("F", "Installation"),
            ("G", "Testing"),
            ("H", "Error Handling"),
            ("I", "Security"),
            ("J", "Extensibility"),
            ("K", "Reproducibility"),
            ("L", "Maintainability"),
            ("N", "Visualization"),
            ("O", "CI/CD"),
        ],
    )
    def test_single_letter_codes(self, source: str, expected: str) -> None:
        assert classify_assessment_category(source) == expected

    def test_keyword_architecture(self) -> None:
        assert (
            classify_assessment_category("some_source", "architecture issues")
            == "Architecture"
        )

    def test_keyword_quality(self) -> None:
        assert (
            classify_assessment_category("X", "code quality concern") == "Code Quality"
        )

    def test_keyword_testing(self) -> None:
        assert classify_assessment_category("test_suite") == "Testing"

    def test_keyword_security(self) -> None:
        assert classify_assessment_category("X", "security vulnerability") == "Security"

    def test_keyword_ci(self) -> None:
        assert classify_assessment_category("X", "ci pipeline failure") == "CI/CD"

    def test_unknown_returns_general(self) -> None:
        assert classify_assessment_category("Z", "completely random") == "General"

    def test_requires_source_name(self) -> None:
        with pytest.raises(AssertionError):
            classify_assessment_category(None)  # type: ignore[arg-type]
