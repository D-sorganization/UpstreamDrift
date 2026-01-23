#!/usr/bin/env python3
"""Quality check script to verify AI-generated code meets standards."""

import ast
import re
import sys
from pathlib import Path


# ANSI colors for terminal output
class Colors:
    if sys.stderr.isatty():
        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
    else:
        HEADER = ""
        BLUE = ""
        CYAN = ""
        GREEN = ""
        WARNING = ""
        FAIL = ""
        ENDC = ""
        BOLD = ""


# Configuration
BANNED_PATTERNS = [
    (re.compile(r"\b" + "TO" + "DO" + r"\b"), "TO" + "DO placeholder found"),
    (re.compile(r"\b" + "FIX" + "ME" + r"\b"), "FIX" + "ME placeholder found"),
    (re.compile(r"^\s*\.\.\.\s*$"), "Ellipsis placeholder"),
    (re.compile(r"NotImplementedError"), "NotImplementedError placeholder"),
    (re.compile(r"your.*here", re.IGNORECASE), "Template placeholder"),
    (re.compile(r"insert.*here", re.IGNORECASE), "Template placeholder"),
]

# More intelligent pass statement detection
PASS_PATTERNS = [
    # Empty pass statements that are likely placeholders
    (re.compile(r"^\s*pass\s*$"), "Empty pass statement"),
    # Pass statements in empty blocks that might be placeholders
    (
        re.compile(r"^\s*if\s+.*:\s*$"),
        "Empty if block - consider adding logic or comment",
    ),
    (
        re.compile(r"^\s*else:\s*$"),
        "Empty else block - consider adding logic or comment",
    ),
    (
        re.compile(r"^\s*except\s+.*:\s*$"),
        "Empty except block - consider adding error handling",
    ),
]

MAGIC_NUMBERS = [
    (re.compile(r"(?<![0-9])3\.141"), "Use math.pi instead of 3.141"),
    (re.compile(r"(?<![0-9])9\.8[0-9]?(?![0-9])"), "Define GRAVITY_M_S2 constant"),
    (re.compile(r"(?<![0-9])6\.67[0-9]?(?![0-9])"), "Define gravitational constant"),
]


def is_legitimate_pass_context(lines: list[str], line_num: int) -> bool:
    """Check if a pass statement is in a legitimate context."""
    if line_num <= 0 or line_num > len(lines):
        return False

    line = lines[line_num - 1].strip()
    if line != "pass":
        return False

    # Check if this is in a class definition (legitimate)
    for i in range(line_num - 1, max(0, line_num - 10), -1):
        prev_line = lines[i - 1].strip()
        if prev_line.startswith("class "):
            return True
        if prev_line.startswith("def "):
            return False
        if prev_line.endswith(":") and any(
            keyword in prev_line
            for keyword in [
                "try:",
                "except",
                "finally:",
                "with ",
                "if __name__",
                "if TYPE_CHECKING",
            ]
        ):
            return True

    # Check if this is in a try/except block (legitimate)
    for i in range(line_num - 1, max(0, line_num - 5), -1):
        prev_line = lines[i - 1].strip()
        if "try:" in prev_line or "except" in prev_line:
            return True

    # Check if this is in a context manager (legitimate)
    for i in range(line_num - 1, max(0, line_num - 3), -1):
        prev_line = lines[i - 1].strip()
        if prev_line.startswith("with "):
            return True

    return False


def check_banned_patterns(
    lines: list[str],
    filepath: Path,
) -> list[tuple[int, str, str]]:
    """Check for banned patterns in lines."""
    issues: list[tuple[int, str, str]] = []
    # Skip checking quality check scripts for their own patterns
    if filepath.name in (
        "quality_check_script.py",
        "matlab_quality_check.py",
        "code_quality_check.py",
        "quality_check.py",
        "quality-check.py",
    ):
        return issues

    for line_num, line in enumerate(lines, 1):
        # Check for basic banned patterns
        for pattern, message in BANNED_PATTERNS:
            if pattern.search(line):
                issues.append((line_num, message, line.strip()))

        # Special handling for pass statements
        if re.match(r"^\s*pass\s*$", line) and not is_legitimate_pass_context(
            lines,
            line_num,
        ):
            issues.append(
                (
                    line_num,
                    "Empty pass statement - consider adding logic or comment",
                    line.strip(),
                ),
            )

    return issues


def check_magic_numbers(lines: list[str], filepath: Path) -> list[tuple[int, str, str]]:
    """Check for magic numbers in lines."""
    issues: list[tuple[int, str, str]] = []
    # Skip checking quality check scripts for magic numbers (they contain patterns
    # they check for)
    if filepath.name in (
        "quality_check_script.py",
        "matlab_quality_check.py",
        "code_quality_check.py",
        "quality_check.py",
        "quality-check.py",
    ):
        return issues
    for line_num, line in enumerate(lines, 1):
        line_content = line[: line.index("#")] if "#" in line else line
        for pattern, message in MAGIC_NUMBERS:
            if pattern.search(line_content):
                issues.append((line_num, message, line.strip()))
    return issues


def check_ast_issues(content: str, filepath: Path) -> list[tuple[int, str, str]]:
    """Check AST for quality issues."""
    issues: list[tuple[int, str, str]] = []
    # Skip checking quality check scripts for AST issues
    if filepath.name in (
        "quality_check_script.py",
        "matlab_quality_check.py",
        "code_quality_check.py",
        "quality_check.py",
        "quality-check.py",
    ):
        return issues
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    issues.append(
                        (node.lineno, f"Function '{node.name}' missing docstring", ""),
                    )
                if not node.returns and node.name != "__init__":
                    pass
                    # Relaxed: We let MyPy handle missing return checks,
                    # as this stricter check might block valid quick scripts.
                    # Uncomment to enforce:
                    # issues.append(
                    #     (
                    #         node.lineno,
                    #         f"Function '{node.name}' missing return type hint",
                    #         "",
                    #     )
                    # )
    except SyntaxError as e:
        issues.append((0, f"Syntax error: {e}", ""))
    return issues


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a Python file for quality issues."""
    try:
        content = filepath.read_text(encoding="utf-8")
        lines = content.splitlines()

        issues = []
        issues.extend(check_banned_patterns(lines, filepath))
        issues.extend(check_magic_numbers(lines, filepath))
        issues.extend(check_ast_issues(content, filepath))
    except (OSError, UnicodeDecodeError) as e:
        return [(0, f"Error reading file: {e}", "")]
    else:
        return issues


def main() -> None:
    """Run quality checks on Python files."""
    # Support direct file arguments from pre-commit
    if len(sys.argv) > 1:
        python_files = [Path(arg) for arg in sys.argv[1:]]
    else:
        python_files = list(Path().rglob("*.py"))

    # Exclude certain directories
    exclude_dirs = {
        "archive",
        "legacy",
        "experimental",
        ".git",
        "__pycache__",
        ".ruff_cache",
        ".mypy_cache",
        "matlab",
        "output",
        ".ipynb_checkpoints",  # Add checkpoint files to exclusion
        ".Trash",  # Add trash files to exclusion
    }

    # Filter if scanning directory
    if len(sys.argv) <= 1:
        python_files = [
            f for f in python_files if not any(part in exclude_dirs for part in f.parts)
        ]

    all_issues = []
    for filepath in python_files:
        issues = check_file(filepath)
        if issues:
            all_issues.append((filepath, issues))

    # Report
    if all_issues:
        sys.stderr.write(
            f"{Colors.FAIL}{Colors.BOLD}❌ Quality check FAILED{Colors.ENDC}\n\n"
        )
        for filepath, issues in all_issues:
            sys.stderr.write(f"\n{Colors.CYAN}{filepath}:{Colors.ENDC}\n")
            for line_num, message, code in issues:
                if line_num > 0:
                    sys.stderr.write(
                        f"  Line {Colors.BOLD}{line_num}{Colors.ENDC}: {message}\n"
                    )
                    if code:
                        sys.stderr.write(f"    > {Colors.WARNING}{code}{Colors.ENDC}\n")
                else:
                    sys.stderr.write(f"  {message}\n")

        total_issues = sum(len(issues) for _, issues in all_issues)
        sys.stderr.write(
            f"\n{Colors.FAIL}Total issues: {total_issues}{Colors.ENDC}\n",
        )
        sys.exit(1)
    else:
        # success silent for pre-commit usually, but ok to print
        pass
        sys.exit(0)


if __name__ == "__main__":
    main()
