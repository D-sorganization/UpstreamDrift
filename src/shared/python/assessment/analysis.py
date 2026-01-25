"""Utilities for analyzing Python code quality and structure."""

import ast
import re
from pathlib import Path
from typing import Any


def get_python_metrics(file_path: Path) -> dict[str, Any]:
    """Extract metrics from a Python file using AST."""
    metrics = {
        "functions": 0,
        "classes": 0,
        "docstrings": 0,
        "typed_returns": 0,
        "branches": 0,
    }

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                metrics["functions"] += 1
                if ast.get_docstring(node):
                    metrics["docstrings"] += 1
                if node.returns:
                    metrics["typed_returns"] += 1
            elif isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
                if ast.get_docstring(node):
                    metrics["docstrings"] += 1
            elif isinstance(node, ast.If | ast.For | ast.While | ast.ExceptHandler):
                metrics["branches"] += 1

    except Exception:
        pass

    return metrics


def calculate_complexity(metrics: dict[str, int]) -> float:
    """Calculate average complexity (branches per function)."""
    funcs = metrics.get("functions", 0)
    branches = metrics.get("branches", 0)
    return branches / funcs if funcs > 0 else 0.0


def assess_error_handling_content(content: str) -> dict[str, int]:
    """Count try/except patterns in content."""
    return {
        "try_count": content.count("try:"),
        "bare_except_count": len(re.findall(r"except\s*:", content)),
    }


def assess_logging_content(content: str) -> dict[str, int]:
    """Count logging vs print usage in content."""
    return {
        "logging_usage": len(re.findall(r"logging\.|logger\.", content)),
        "print_usage": content.count("print("),
    }


def get_detailed_function_metrics(content: str) -> list[dict[str, Any]]:
    """Extract detailed metrics for each function in a file."""
    functions = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body_lines = (
                    node.end_lineno - node.lineno + 1
                    if hasattr(node, "end_lineno")
                    else 0
                )
                functions.append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": len(node.args.args),
                        "body_lines": body_lines,
                        "has_docstring": ast.get_docstring(node) is not None,
                    }
                )
    except Exception:
        pass
    return functions


def count_files(root: Path, pattern: str) -> int:
    """Count files matching a pattern."""
    return len(list(root.glob(pattern)))


def grep_count(root: Path, pattern: str, file_pattern: str = "**/*.py") -> int:
    """Count files where a regex pattern is found."""
    count = 0
    regex = re.compile(pattern)
    for p in root.glob(file_pattern):
        if p.is_file():
            try:
                with p.open(encoding="utf-8", errors="ignore") as f:
                    if regex.search(f.read()):
                        count += 1
            except Exception:
                pass
    return count


def classify_assessment_category(source_name: str, description: str = "") -> str:
    """Classify an assessment finding into a standard category name.

    Args:
        source_name: Name of the source report or category ID.
        description: Optional detailed description for keyword matching.

    Returns:
        A standardized category name.
    """
    text = (source_name + " " + description).lower()

    if "architecture" in text or "implementation" in text or "A" == source_name:
        return "Architecture"
    if "quality" in text or "hygiene" in text or "B" == source_name:
        return "Code Quality"
    if "documentation" in text or "C" == source_name:
        return "Documentation"
    if "user" in text or "ux" in text or "D" == source_name:
        return "User Experience"
    if "performance" in text or "E" == source_name:
        return "Performance"
    if "installation" in text or "deployment" in text or "F" == source_name:
        return "Installation"
    if "test" in text or "G" == source_name:
        return "Testing"
    if "error" in text or "H" == source_name:
        return "Error Handling"
    if "security" in text or "I" == source_name:
        return "Security"
    if "extensibility" in text or "J" == source_name:
        return "Extensibility"
    if "reproducibility" in text or "K" == source_name:
        return "Reproducibility"
    if "maintainability" in text or "L" == source_name:
        return "Maintainability"
    if "visualization" in text or "N" == source_name:
        return "Visualization"
    if "ci" in text or "cd" in text or "O" == source_name:
        return "CI/CD"

    return "General"
