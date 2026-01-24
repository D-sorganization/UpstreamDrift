import ast
import os
from typing import Any


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
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            # Check docs
            if not ast.get_docstring(node):
                # Skip private members (starting with _)
                if not node.name.startswith("_"):
                    # Skip tests
                    if "tests" not in filepath and "test_" not in filepath:
                        docs_file.write(f"{filepath}:{node.lineno} {node.name}\n")

            # Check stubs (functions only)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if is_stub(node):
                    stubs_file.write(f"{filepath}:{node.lineno} {node.name}\n")


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
