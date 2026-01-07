import ast
import json
import sys
from pathlib import Path

RISKS = []


class ScienceAuditor(ast.NodeVisitor):
    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        # 1. Division Safety
        # Use simple nested if to avoid complex boolean expression lint struggles or
        # suppress SIM102 if preferred. Actually, combining them is cleaner.
        if isinstance(node.op, ast.Div) and not (
            isinstance(node.right, ast.Constant) and node.right.value != 0
        ):
            RISKS.append(
                {
                    "line": node.lineno,
                    "type": "Singularity Risk",
                    "msg": "Division by variable detected. Check denominator.",
                },
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # 2. Trig Safety
        if isinstance(node.func, ast.Attribute) and node.func.attr in [
            "sin",
            "cos",
            "tan",
        ]:
            # Flag if the argument is a numeric constant (likely ambiguous units)
            if any(
                isinstance(arg, ast.Constant) and isinstance(arg.value, int | float)
                for arg in node.args
            ):
                RISKS.append(
                    {
                        "line": node.lineno,
                        "type": "Unit Ambiguity",
                        "msg": (
                            "Trig function called with a numeric constant. "
                            "Check if argument is in radians "
                            "(Python math module expects radians)."
                        ),
                    },
                )
        self.generic_visit(node)


def main() -> None:
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path()

    # Use rglob to recursively find .py files
    for py_file in target_dir.rglob("*.py"):
        if "test" in py_file.name:
            continue

        try:
            with py_file.open(encoding="utf-8") as source:
                ScienceAuditor().visit(ast.parse(source.read()))
        except Exception as e:  # noqa: BLE001
            # Log error but continue scanning
            # We catch generic Exception because ast.parse can raise
            # various errors and we don't want to crash the entire audit.
            sys.stderr.write(f"Error analyzing {py_file}: {e}\n")

    if RISKS:
        print(json.dumps(RISKS, indent=2))  # noqa: T201
        sys.exit(1)
    else:
        print("[]")  # noqa: T201


if __name__ == "__main__":
    main()
