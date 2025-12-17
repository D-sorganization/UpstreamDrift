"""Tests for quality check script."""

import importlib.util
import sys
import types
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
scripts_path = project_root / "scripts"
sys.path.insert(0, str(scripts_path))

# Import the quality check module
# Use importlib to import from scripts directory

spec = importlib.util.spec_from_file_location(
    "quality_check", project_root / "scripts" / "quality-check.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load quality_check module")

quality_check: types.ModuleType = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality_check)


class TestContextDetection:
    """Tests for context detection functions."""

    def test_is_in_class_definition_detects_class(self) -> None:
        """Test detection of pass in class definition."""
        lines = [
            "class MyClass:\n",
            "    '''Docstring'''\n",
            "    pass\n",
        ]
        # Line 3 (index 2) is the pass statement
        assert quality_check.is_legitimate_pass_context(lines, 3)

    def test_is_in_class_definition_rejects_function(self) -> None:
        """Test that pass in function is not considered class context."""
        lines = [
            "def my_function():\n",
            "    pass\n",
        ]
        # Line 2 (index 1) is the pass statement
        assert not quality_check.is_legitimate_pass_context(lines, 2)

    def test_is_in_try_except_block(self) -> None:
        """Test detection of pass in try/except block."""
        lines = [
            "try:\n",
            "    something()\n",
            "except Exception:\n",
            "    pass\n",
        ]
        # Line 4 is the pass statement
        assert quality_check.is_legitimate_pass_context(lines, 4)

    def test_is_in_context_manager(self) -> None:
        """Test detection of pass in context manager."""
        lines = [
            "with open('file.txt') as f:\n",
            "    pass\n",
        ]
        # Line 2 is the pass statement
        assert quality_check.is_legitimate_pass_context(lines, 2)

    def test_pass_not_in_legitimate_context(self) -> None:
        """Test that standalone pass is not legitimate."""
        lines = [
            "def some_function():\n",
            "    x = 1\n",
            "    pass\n",
        ]
        # Line 3 is the pass statement
        assert not quality_check.is_legitimate_pass_context(lines, 3)

    def test_is_legitimate_pass_context_with_main_block(self) -> None:
        """Test that pass in if __name__ block is legitimate."""
        lines = [
            "if __name__ == '__main__':\n",
            "    pass\n",
        ]
        assert quality_check.is_legitimate_pass_context(lines, 2)


class TestQualityCheckScriptDetection:
    """Tests for quality check script detection."""

    def test_is_quality_check_script_exact_match(self) -> None:
        """Test exact filename match."""
        assert quality_check.is_quality_check_script(Path("quality-check.py"))
        assert quality_check.is_quality_check_script(Path("quality_check.py"))
        assert quality_check.is_quality_check_script(Path("quality_check_script.py"))

    def test_is_quality_check_script_in_path(self) -> None:
        """Test script detection in path."""
        assert quality_check.is_quality_check_script(
            Path("/path/to/quality_check_file.py")
        )

    def test_is_quality_check_script_negative(self) -> None:
        """Test non-quality-check files."""
        assert not quality_check.is_quality_check_script(Path("test_something.py"))
        assert not quality_check.is_quality_check_script(Path("main.py"))


class TestBannedPatterns:
    """Tests for banned pattern detection."""

    def test_check_banned_patterns_todo(self) -> None:
        """Test detection of TODO placeholder."""
        lines = ["# TODO: implement this\n", "def function():\n", "    pass\n"]
        filepath = Path("test.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        assert len(issues) > 0
        assert any("TODO" in issue[1] for issue in issues)
        assert issues[0][0] == 1  # Line number

    def test_check_banned_patterns_fixme(self) -> None:
        """Test detection of FIXME placeholder."""
        lines = ["# FIXME: broken code\n", "x = 1\n"]
        filepath = Path("test.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        assert len(issues) > 0
        assert any("FIXME" in issue[1] for issue in issues)

    def test_check_banned_patterns_not_implemented(self) -> None:
        """Test detection of NotImplementedError."""
        lines = ["def function():\n", "    raise NotImplementedError\n"]
        filepath = Path("test.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        assert len(issues) > 0
        assert any("NotImplementedError" in issue[1] for issue in issues)

    def test_check_banned_patterns_skips_quality_check_script(self) -> None:
        """Test that quality check script is not checked for its own patterns."""
        lines = ["# TODO: this is ok in quality check script\n"]
        filepath = Path("quality-check.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        assert len(issues) == 0

    def test_check_banned_patterns_illegitimate_pass(self) -> None:
        """Test detection of illegitimate pass statement."""
        lines = ["def function():\n", "    x = 1\n", "    pass\n"]
        filepath = Path("test.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        assert len(issues) > 0
        assert any("pass" in issue[1].lower() for issue in issues)

    def test_check_banned_patterns_legitimate_pass_ignored(self) -> None:
        """Test that legitimate pass statements are not flagged."""
        lines = ["class MyClass:\n", "    pass\n"]
        filepath = Path("test.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        # Should not flag pass in class definition
        pass_issues = [issue for issue in issues if "pass" in issue[1].lower()]
        assert len(pass_issues) == 0

    def test_check_banned_patterns_template_placeholders(self) -> None:
        """Test detection of template placeholders."""
        lines = [
            "# <your code here>\n",
            "# <insert logic here>\n",
            "# your code here\n",
        ]
        filepath = Path("test.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        assert len(issues) >= 2  # At least the explicit template patterns


class TestMagicNumbers:
    """Tests for magic number detection."""

    def test_check_magic_numbers_pi(self) -> None:
        """Test detection of hardcoded pi value."""
        lines = ["area = radius * radius * 3.14159\n"]
        filepath = Path("test.py")
        issues = quality_check.check_magic_numbers(lines, filepath)

        assert len(issues) > 0
        assert any("math.pi" in issue[1] for issue in issues)

    def test_check_magic_numbers_gravity(self) -> None:
        """Test detection of hardcoded gravity value."""
        lines = ["force = mass * 9.8\n"]
        filepath = Path("test.py")
        issues = quality_check.check_magic_numbers(lines, filepath)

        assert len(issues) > 0
        assert any("GRAVITY" in issue[1] for issue in issues)

    def test_check_magic_numbers_skips_quality_check_script(self) -> None:
        """Test that quality check script is not checked for magic numbers."""
        lines = ["x = 3.14159\n"]
        filepath = Path("quality-check.py")
        issues = quality_check.check_magic_numbers(lines, filepath)

        assert len(issues) == 0

    def test_check_magic_numbers_ignores_comments(self) -> None:
        """Test that magic numbers in comments are ignored."""
        lines = ["# This uses pi = 3.14159 for calculation\n", "x = 1\n"]
        filepath = Path("test.py")
        issues = quality_check.check_magic_numbers(lines, filepath)

        # Should not flag the pi in the comment
        assert len(issues) == 0


class TestASTIssues:
    """Tests for AST-based quality checks."""

    def test_check_ast_issues_missing_docstring(self) -> None:
        """Test detection of missing function docstring."""
        content = """
def my_function():
    return 42
"""
        issues = quality_check.check_ast_issues(content)

        assert len(issues) > 0
        assert any("docstring" in issue[1].lower() for issue in issues)

    def test_check_ast_issues_missing_return_type(self) -> None:
        """Test detection of missing return type hint."""
        content = """
def my_function():
    '''This has a docstring.'''
    return 42
"""
        issues = quality_check.check_ast_issues(content)

        assert len(issues) > 0
        assert any("return type" in issue[1].lower() for issue in issues)

    def test_check_ast_issues_init_no_return_type_required(self) -> None:
        """Test that __init__ doesn't require return type hint."""
        content = """
class MyClass:
    def __init__(self):
        '''Initialize.'''
        pass
"""
        issues = quality_check.check_ast_issues(content)

        # Should not complain about missing return type for __init__
        return_type_issues = [
            issue for issue in issues if "return type" in issue[1].lower()
        ]
        assert len(return_type_issues) == 0

    def test_check_ast_issues_syntax_error(self) -> None:
        """Test detection of syntax errors."""
        content = "def my_function(\n"  # Incomplete function
        issues = quality_check.check_ast_issues(content)

        assert len(issues) > 0
        assert any("syntax" in issue[1].lower() for issue in issues)

    def test_check_ast_issues_well_formed_function(self) -> None:
        """Test that well-formed function passes checks."""
        content = """
def my_function() -> int:
    '''This is a well-formed function.'''
    return 42
"""
        issues = quality_check.check_ast_issues(content)

        # Should have no issues
        assert len(issues) == 0


class TestCheckFile:
    """Tests for check_file function."""

    def test_check_file_with_issues(self, tmp_path: Path) -> None:
        """Test checking a file with quality issues."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
# TODO: implement this
def my_function():
    pass
"""
        )

        issues = quality_check.check_file(test_file)

        assert len(issues) > 0
        # Should have TODO and missing docstring/return type
        assert any("TODO" in issue[1] for issue in issues)

    def test_check_file_clean_code(self, tmp_path: Path) -> None:
        """Test checking a file with no issues."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def my_function() -> int:
    '''A well-formed function.'''
    return 42
"""
        )

        issues = quality_check.check_file(test_file)

        # Clean code should have no issues
        assert len(issues) == 0

    def test_check_file_unicode_error(self, tmp_path: Path) -> None:
        """Test handling of files with encoding errors."""
        test_file = tmp_path / "test.py"
        # Write binary content that's not valid UTF-8
        test_file.write_bytes(b"\x80\x81\x82")

        issues = quality_check.check_file(test_file)

        assert len(issues) > 0
        assert any("error reading file" in issue[1].lower() for issue in issues)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_lines_list(self) -> None:
        """Test handling of empty lines list."""
        lines: list[str] = []
        filepath = Path("test.py")

        issues = quality_check.check_banned_patterns(lines, filepath)
        assert len(issues) == 0

        issues = quality_check.check_magic_numbers(lines, filepath)
        assert len(issues) == 0

    def test_is_legitimate_pass_context_invalid_line_number(self) -> None:
        """Test handling of invalid line numbers."""
        lines = ["class MyClass:\n", "    pass\n"]

        # Line number out of range
        assert not quality_check.is_legitimate_pass_context(lines, 10)
        assert not quality_check.is_legitimate_pass_context(lines, 0)
        assert not quality_check.is_legitimate_pass_context(lines, -1)

    def test_multiple_issues_per_line(self) -> None:
        """Test detection of multiple issues on same line."""
        lines = ["# TODO: FIXME: implement this\n"]
        filepath = Path("test.py")
        issues = quality_check.check_banned_patterns(lines, filepath)

        # Should detect both TODO and FIXME
        assert len(issues) >= 2

    def test_check_ast_issues_empty_content(self) -> None:
        """Test AST checking with empty content."""
        issues = quality_check.check_ast_issues("")
        assert len(issues) == 0

    def test_check_ast_issues_only_comments(self) -> None:
        """Test AST checking with only comments."""
        content = "# Just a comment\n# Another comment\n"
        issues = quality_check.check_ast_issues(content)
        assert len(issues) == 0
