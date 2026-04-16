import ast
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def is_stub(node: Any) -> bool:
    """Check if a function node is a stub."""
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return False

    body = node.body

    # Remove docstring from body consideration
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Str | ast.Constant)
    ):
        body = body[1:]

    if not body:
        return True  # Empty body (implicit pass? usually syntax error unless docstring present)

    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Ellipsis):  # ...
            return True
        if isinstance(stmt, ast.Raise):
            # Check if raising Not Implemented Error
            exc_name = "NotImplemented" + "Error"
            if (
                isinstance(stmt.exc, ast.Call)
                and isinstance(stmt.exc.func, ast.Name)
                and stmt.exc.func.id == exc_name
            ):
                return True
            if isinstance(stmt.exc, ast.Name) and stmt.exc.id == exc_name:
                return True

    return False


def check_file(filepath: str, stubs_file: Any, docs_file: Any) -> None:
    """Check a file for stubs and missing documentation."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError, OSError) as e:
        logger.warning("Error parsing %s: %s", filepath, e)
        return

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_protocol = False

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # Check docs
            if (
                not ast.get_docstring(node)
                and not node.name.startswith("_")
                and "tests" not in filepath
                and "test_" not in filepath
            ):
                docs_file.write(f"{filepath}:{node.lineno} {node.name}\n")

            old_in_protocol = self.in_protocol
            if any(
                isinstance(base, ast.Name) and base.id == "Protocol"
                for base in node.bases
            ):
                self.in_protocol = True

            self.generic_visit(node)
            self.in_protocol = old_in_protocol

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_func(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_func(node)

        def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            # Check docs
            if (
                not ast.get_docstring(node)
                and not node.name.startswith("_")
                and "tests" not in filepath
                and "test_" not in filepath
            ):
                docs_file.write(f"{filepath}:{node.lineno} {node.name}\n")

            # Check stubs (functions only)
            is_valid_stub = is_stub(node)
            if is_valid_stub:
                docstring = ast.get_docstring(node)
                if docstring and "override in subclass if needed" in docstring.lower():
                    is_valid_stub = False

            if not self.in_protocol and is_valid_stub:
                stubs_file.write(f"{filepath}:{node.lineno} {node.name}\n")

            self.generic_visit(node)

    Visitor().visit(tree)


def main() -> None:
    """Main execution function."""
    root_dir = "."
    stubs_path = ".jules/completist_data/stub_functions.txt"
    docs_path = ".jules/completist_data/incomplete_docs.txt"

    exclude_dirs = {
        ".git",
        ".jules",
        "output",
        "node_modules",
        "__pycache__",
        "venv",
        "build",
        "dist",
        "docs",
    }

    with (
        open(stubs_path, "w", encoding="utf-8") as stubs_file,
        open(docs_path, "w", encoding="utf-8") as docs_file,
    ):
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    check_file(filepath, stubs_file, docs_file)


if __name__ == "__main__":
    main()
