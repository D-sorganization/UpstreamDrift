"""Scientific Auditor for Biomechanical Simulations.

Analyzes Python code for common scientific computing pitfalls such as:
- Potential division by zero (singularity risks).
- Trig function unit ambiguity (radians vs degrees).

Refactored to address DRY violations (Pragmatic Programmer).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class ScienceAuditor(ast.NodeVisitor):
    """AST visitor to detect scientific computing risks."""

    def __init__(self) -> None:
        self.risks: list[dict[str, object]] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Check for potentially unsafe divisions."""
        # 1. Division Safety
        if isinstance(node.op, ast.Div):
            # Check if denominator is a variable or a zero constant
            is_constant_nonzero = (
                isinstance(node.right, ast.Constant) and node.right.value != 0
            )
            if not is_constant_nonzero:
                self.risks.append(
                    {
                        "line": node.lineno,
                        "type": "Singularity Risk",
                        "msg": "Division by variable detected. Ensure denominator is non-zero.",
                    }
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for trig function unit ambiguity."""
        # 2. Trig Safety (sin, cos, tan)
        trig_functions = {"sin", "cos", "tan"}
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in trig_functions:
            # Flag if the argument is a numeric constant (likely ambiguous units)
            if any(
                isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float))
                for arg in node.args
            ):
                self.risks.append(
                    {
                        "line": node.lineno,
                        "type": "Unit Ambiguity",
                        "msg": (
                            f"{func_name}() called with a numeric constant. "
                            "Verify if argument is in radians (Python default)."
                        ),
                    }
                )
        self.generic_visit(node)


def run_audit(target_path: Path) -> list[dict[str, object]]:
    """Run scientific audit on a file or directory.

    Args:
        target_path: Path to audit.

    Returns:
        List of detected risks.
    """
    auditor = ScienceAuditor()
    files = [target_path] if target_path.is_file() else target_path.rglob("*.py")

    for py_file in files:
        if "test" in py_file.name:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            auditor.visit(tree)
        except Exception as e:
            logger.error(f"Failed to audit {py_file}: {e}")

    return auditor.risks


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python scientific_auditor.py <path>\n")
        sys.exit(1)

    target = Path(sys.argv[1])
    risks = run_audit(target)

    if risks:
        sys.stdout.write(json.dumps(risks, indent=2) + "\n")
        sys.exit(1)
    else:
        sys.stdout.write("[]\n")


if __name__ == "__main__":
    main()
